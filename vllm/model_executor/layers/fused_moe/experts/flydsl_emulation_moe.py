# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlyDSL BF16 2-stage fused-MoE driver for the MXFP8-emulation path.

On devices without a native MXFP8 MoE kernel (e.g. ROCm gfx942 / MI300), the
MiniMax-M3-MXFP8 checkpoint is dequantized to BF16 at load and runs through
``Mxfp8EmulationTritonExperts`` (a plain BF16 x BF16 MoE GEMM). This module
provides an optional FlyDSL replacement for that GEMM using aiter's vendored
FlyDSL kernels (``aiter.ops.flydsl.kernels.moe_gemm_2stage``):

  stage1  g1u1:  a1[M,H] . w1[E,2I,H] -> SwiGLU -> out1[M*topk, I]
  stage2  down:  a2[M*topk,I] . w2[E,H,I] -> weighted-reduce -> out[M,H]

It is **opt-in** via ``VLLM_MINIMAX_M3_FLYDSL_MOE=1`` and **off by default**, so
no other model/path is affected. On any error it raises and the caller falls
back to the stock Triton experts. The weight non-temporal (NT) load policy is
left at the default (``weight_cache_modifier=0``); that is a separate aiter knob.

Ported from a validated standalone prototype; routing uses aiter ``moe_sorting``.
"""
import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# Lazy/cached FlyDSL handle: imports are heavy and only valid on ROCm with aiter
# + the flydsl runtime present, so resolve once on first use.
_FLY: dict | None = None
_FLY_LOAD_FAILED = False

# Caches: compiled executables per (M,E,H,inter,topk); shuffled-weight flat views
# keyed by data_ptr; one monotonically-grown scratch pool sliced per shape.
_EXE: dict = {}
_WSHUF: dict = {}
_POOL = {"rows": 0, "gemm1_raw": None, "a2": None, "scale_dummy": None}


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) == "1"


def enabled() -> bool:
    """Opt-in flag. Off by default -> zero impact on any other path/model."""
    return _env_flag("VLLM_MINIMAX_M3_FLYDSL_MOE", "0")


def _load_fly():
    global _FLY, _FLY_LOAD_FAILED
    if _FLY is not None or _FLY_LOAD_FAILED:
        return _FLY
    try:
        import flydsl.compiler as flyc  # noqa: F401

        from aiter.fused_moe import moe_sorting
        from aiter.ops.flydsl.kernels.moe_gemm_2stage import (
            compile_moe_gemm1,
            compile_moe_gemm2,
        )
        from aiter.ops.shuffle import shuffle_weight

        _FLY = dict(
            flyc=flyc,
            compile_moe_gemm1=compile_moe_gemm1,
            compile_moe_gemm2=compile_moe_gemm2,
            shuffle_weight=shuffle_weight,
            moe_sorting=moe_sorting,
        )
    except Exception as e:  # aiter / flydsl runtime not available
        _FLY_LOAD_FAILED = True
        logger.warning_once("FlyDSL MoE unavailable, staying on Triton: %s", e)
        _FLY = None
    return _FLY


def available() -> bool:
    """Enabled + gfx942 + aiter-flydsl importable. gfx942 only: NT/tile behavior
    is validated there; other archs should validate before being added."""
    if not enabled():
        return False
    from vllm.platforms.rocm import on_gfx942

    if not on_gfx942():  # validated on gfx942 only
        return False
    return _load_fly() is not None


# FlyDSL weight layout. Shared between the load-time in-place shuffle (modelopt)
# and the per-call fallback below, so the layout lives in exactly one place.
_FLY_SHUFFLE_LAYOUT = (16, 16)


def shuffle_weight_to_fly_layout(w: torch.Tensor) -> torch.Tensor:
    """Shuffle ``w`` into FlyDSL layout and tag it ``_fly_shuffled``."""
    ws = _load_fly()["shuffle_weight"](w, layout=_FLY_SHUFFLE_LAYOUT).contiguous()
    ws._fly_shuffled = True
    return ws


def shuffle_weight_inplace_ready(w: torch.Tensor):
    """Return a flat FlyDSL-layout view of weight ``w``. If the weight was already
    shuffled in-place at load (tagged ``_fly_shuffled``), this is a free view;
    otherwise it shuffles once and caches the result (bounded to one layer)."""
    key = w.data_ptr()
    hit = _WSHUF.get(key)
    if hit is None:
        if getattr(w, "_fly_shuffled", False):
            hit = w.view(-1)
        else:
            hit = shuffle_weight_to_fly_layout(w).view(-1)
        _WSHUF[key] = hit
    return hit


def _tile_cfg(M: int):
    """(tile_m, tile_n1, tile_n2, tile_k, k_batch).

    tile_n1 MUST be 128 (gate/up 384-halves alignment). k_batch defaults to 1
    (validated on aiter); split-K (decode k_batch>1) and prefill tile_k are
    tunable via env. tile_k=64 raises occupancy for prefill (autotune finding)."""
    if M <= 256:
        # decode split-K (k_batch=6) fills the GPU at tiny M; measured +8% e2e
        # throughput / -8% TPOT vs Triton at cc=2 (vs ~-5% at k_batch=1).
        kb = int(os.environ.get("FLY_DECODE_KBATCH", "6"))
        return (16, 128, 256, 128, kb)
    ptm = int(os.environ.get("FLY_PREFILL_TILE_M", "128"))
    ptk = int(os.environ.get("FLY_PREFILL_TILE_K", "64"))
    kb = int(os.environ.get("FLY_PREFILL_KBATCH", "1"))
    return (ptm, 128, 256, ptk, kb)


def _pool(rows_needed, two_inter, inter, device):
    if _POOL["rows"] < rows_needed:
        _POOL["gemm1_raw"] = torch.zeros(
            (rows_needed, two_inter), dtype=torch.bfloat16, device=device
        )
        _POOL["a2"] = torch.empty(
            (rows_needed, inter), dtype=torch.bfloat16, device=device
        )
        _POOL["rows"] = rows_needed
        if _POOL["scale_dummy"] is None:
            _POOL["scale_dummy"] = torch.empty((0,), dtype=torch.float32, device=device)
    return _POOL["gemm1_raw"][:rows_needed], _POOL["a2"][:rows_needed], _POOL["scale_dummy"]


def _build_exes(M, E, H, inter, topk):
    fly = _load_fly()
    tile_m, tn1, tn2, tk, kb = _tile_cfg(M)
    # NT (weight_cache_modifier) is a separate aiter knob that may not exist in the
    # installed aiter (it is an unmerged PR). Only pass it when explicitly enabled
    # (FLY_NT != 0); otherwise omit so this works on stock aiter. NT is off here.
    nt = int(os.environ.get("FLY_NT", "0"))
    nt_kw = {"weight_cache_modifier": nt} if nt else {}
    exe1 = fly["compile_moe_gemm1"](
        model_dim=H, inter_dim=inter, experts=E, topk=topk, in_dtype="bf16",
        group_size=-1, tile_m=tile_m, tile_n=tn1, tile_k=tk, doweight_stage1=False,
        # stage1 cshuffle epilog requires f16 out in aiter's flydsl; we use bf16
        # output, so keep it OFF for stage1 (stage2 cshuffle supports bf16).
        use_cshuffle_epilog=False, out_dtype="bf16", scale_is_bf16=False,
        k_batch=int(kb), **nt_kw,
    )
    exe2 = fly["compile_moe_gemm2"](
        model_dim=H, inter_dim=inter, experts=E, topk=topk, in_dtype="bf16",
        group_size=-1, tile_m=tile_m, tile_n=tn2, tile_k=tk, doweight_stage2=True,
        use_cshuffle_epilog=True, accumulate=True, out_dtype="bf16",
        scale_is_bf16=False, **nt_kw,
    )
    return dict(exe1=exe1, exe2=exe2, cexe1=None, cexe2=None, pool_gen=None)


def apply(
    experts,
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation,
    global_num_experts: int,
):
    """Run the BF16 2-stage FlyDSL MoE in-place into ``output``.

    Raises on any failure so the caller can fall back to Triton. ``experts`` is
    the ``Mxfp8EmulationTritonExperts`` instance (used for its exact SwiGLU-OAI
    ``activation()``)."""
    fly = _load_fly()
    flyc = fly["flyc"]
    M, H = hidden_states.shape
    E = w1.shape[0]
    inter = w1.shape[1] // 2
    topk = topk_ids.shape[1]
    if global_num_experts not in (-1, E):
        raise RuntimeError(f"EP global_num_experts={global_num_experts} != E={E}")
    dev = hidden_states.device

    key = (M, E, H, inter, topk)
    st = _EXE.get(key)
    if st is None:
        st = _build_exes(M, E, H, inter, topk)
        _EXE[key] = st

    gemm1_raw, a2, scale_dummy = _pool(M * topk, 2 * inter, inter, dev)
    pool_gen = id(_POOL["gemm1_raw"])

    hs = (
        hidden_states.to(torch.bfloat16).contiguous()
        if hidden_states.dtype != torch.bfloat16
        else hidden_states.contiguous()
    )
    w1s = shuffle_weight_inplace_ready(w1)
    w2s = shuffle_weight_inplace_ready(w2)
    stream = torch.cuda.current_stream()

    # routing depends on this call's topk_ids/weights -> recompute each call.
    sids, sw, seids, nvi, _buf = fly["moe_sorting"](
        topk_ids.to(torch.int32), topk_weights.to(torch.float32),
        E, H, torch.float16, _tile_cfg(M)[0],
    )
    if nvi.numel() > 1:
        nvi = nvi[:1].contiguous()
    sids = sids.contiguous()
    seids = seids.contiguous()
    sw1d = sw.contiguous().view(-1)
    blocks = int(seids.numel())

    if st["cexe1"] is None or st.get("pool_gen") != pool_gen:
        st["cexe1"] = flyc.compile(
            st["exe1"], gemm1_raw.view(-1), hs.view(-1), w1s,
            scale_dummy, scale_dummy, sids, seids, sw1d, nvi,
            M, inter, H, blocks, stream,
        )
        st["cexe2"] = flyc.compile(
            st["exe2"], output.view(-1), a2.view(-1), w2s,
            scale_dummy, scale_dummy, sids, seids, sw1d, nvi,
            M, H, inter, blocks, stream,
        )
        st["pool_gen"] = pool_gen

    gemm1_raw.zero_()
    st["cexe1"](
        gemm1_raw.view(-1), hs.view(-1), w1s, scale_dummy, scale_dummy,
        sids, seids, sw1d, nvi, M, inter, H, blocks, stream,
    )
    # model's own activation (M3 SwiGLU-OAI w/ exact clamp/alpha/beta) -> bit-exact
    experts.activation(activation, a2, gemm1_raw.view(-1, 2 * inter))
    output.zero_()
    st["cexe2"](
        output.view(-1), a2.view(-1), w2s, scale_dummy, scale_dummy,
        sids, seids, sw1d, nvi, M, H, inter, blocks, stream,
    )
