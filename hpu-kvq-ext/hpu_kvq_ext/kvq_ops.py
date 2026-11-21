"""Load + drive the two VALIDATED TurboQuant KV OpBackend kernels (namespace
``hpu_kvq_ext``). Mirrors hpu_dsa_ext.dsa_ops (the proven .so-loader pattern).

The two ops (built + CPU-parity-passed; on-card parity GATED via oncard_op_test.sh):

    torch.ops.hpu_kvq_ext.kv_rotate_quant(k, v, rot, v_bits) -> [k_fp8, k_scale, v_packed, v_scale]
        k,v      [N, D]   bf16   one (token,head) row per line, D == head_dim
        rot      [D, D]   bf16   shared Walsh-Hadamard rotation R (R@R^T == I)
        v_bits           int     4 (k8v4) | 2 (k8v2)
      -> k_fp8  [N, D]     float8_e4m3fn   (rotated K, fp8)
         k_scale[N, 1]     float32
         v_packed[N,D/8|D/16] int32        (rotated V, signed-int packed)
         v_scale[N, 1]     float32

    torch.ops.hpu_kvq_ext.kv_dequant_unrotate(k_fp8, k_scale, v_packed, v_scale,
                                              rot_t, v_bits, out_fp8) -> [k_out, v_out]
        rot_t    [D, D]  bf16   inverse rotation R^T (== R for the symmetric WHT)
        out_fp8         int     0 -> K,V bf16 (bf16 SDPA); 1 -> K fp8 e4m3 (fp8 SDPA)
      -> k_out [N, D]  bf16 (out_fp8==0) | float8_e4m3fn (out_fp8==1)
         v_out [N, D]  bf16

Import-safe: only ``torch`` at module top; habana_frameworks is touched lazily.
LOUD on misconfig (never a silent wrong-KV serve): if a .so is missing or the op
fails to register, we raise (the same contract as dsa_ops / moe_wna16_hpu).
"""
from __future__ import annotations

import functools
import math
import os

import torch

# --------------------------------------------------------------------------
# .so resolution. Build dir is turboquant_kv/kernel-kv/build. At serve time the
# tree mounts at /opt/turboquant_kv; in the dev tree it lives under $HOME/AI.
# HPU_KVQ_SO_DIR overrides for tests / non-standard layouts.
# --------------------------------------------------------------------------
_SO_DIR_ENV = "HPU_KVQ_SO_DIR"
_CANDIDATE_DIRS = (
    "/opt/turboquant_kv/kernel-kv/build",
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "..", "..", "mmel", "turboquant_kv", "kernel-kv", "build")),
    "$HOME/AI/mmel/turboquant_kv/kernel-kv/build",
)
# op -> (eager .so, lazy .so). BOTH ops register into the ONE namespace hpu_kvq_ext.
_SO_NAMES = {
    "kv_rotate_quant": ("libkv_rotate_quant.so", "libkv_rotate_quant_lazy.so"),
    "kv_dequant_unrotate": ("libkv_dequant_unrotate.so", "libkv_dequant_unrotate_lazy.so"),
}
_loaded = {k: False for k in _SO_NAMES}


def _is_lazy() -> bool:
    import habana_frameworks.torch.core as _htc
    return _htc.is_lazy()


def _candidate_dirs() -> list[str]:
    dirs: list[str] = []
    override = os.environ.get(_SO_DIR_ENV)
    if override:
        dirs.append(override)
    dirs.extend(_CANDIDATE_DIRS)
    seen: set[str] = set()
    out: list[str] = []
    for d in dirs:
        nd = os.path.normpath(d)
        if nd not in seen:
            seen.add(nd)
            out.append(nd)
    return out


def _so_path(op: str) -> str:
    eager_name, lazy_name = _SO_NAMES[op]
    name = lazy_name if _is_lazy() else eager_name
    searched = []
    for d in _candidate_dirs():
        cand = os.path.join(d, name)
        searched.append(cand)
        if os.path.isfile(cand):
            return cand
    raise RuntimeError(
        f"[hpu-kvq-ext] KV op library '{name}' (for '{op}', "
        f"mode={'lazy' if _is_lazy() else 'eager'}) not found. Build it CPU-only:\n"
        "  bash $HOME/AI/mmel/turboquant_kv/build_all_kv_kernels.sh\n"
        f"  or set {_SO_DIR_ENV}=<dir-with-the-.so>.\n"
        "  searched:\n    " + "\n    ".join(searched))


def _ensure(op: str):
    if not _loaded[op]:
        so = _so_path(op)
        torch.ops.load_library(so)
        if not (hasattr(torch.ops, "hpu_kvq_ext") and hasattr(torch.ops.hpu_kvq_ext, op)):
            raise RuntimeError(
                f"[hpu-kvq-ext] loaded {so}\nbut torch.ops.hpu_kvq_ext.{op} did NOT "
                "register (check the TORCH_LIBRARY_FRAGMENT def + the .so build).")
        import logging
        logging.getLogger("vllm").info(
            "[hpu-kvq-ext] loaded KV op '%s' from %s (mode=%s)",
            op, so, "lazy" if _is_lazy() else "eager")
        _loaded[op] = True
    return getattr(torch.ops.hpu_kvq_ext, op)


def load_ops() -> None:
    """Eagerly load BOTH op libraries (from register()) so a misconfig fails at
    plugin-init, not mid-first-request. Idempotent. LOUD if a .so is missing.
    In the DEFAULT torch backend the reusable int4-KV core is pure tensor ops -> NO
    .so is needed (and the k8v4 .so SIGSEGVs), so this is a no-op there."""
    if _KVQ_BACKEND == "torch":
        import logging
        logging.getLogger("vllm").info(
            "[hpu-kvq-ext] backend=torch (reusable tensor-op int4-KV core; no .so loaded)")
        return
    _ensure("kv_rotate_quant")
    _ensure("kv_dequant_unrotate")


# --------------------------------------------------------------------------
# The shared Walsh-Hadamard rotation R [D,D] bf16. Symmetric orthogonal
# involution (R @ R == I after /sqrt(D)), so R_t == R -- the read side passes
# the SAME matrix. Data-oblivious constant; built once per (head_dim, device),
# byte-identical to ref/turboquant_kv_ref.make_rotation(use_signs=False).
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=16)
def _rotation_cpu(head_dim: int) -> torch.Tensor:
    assert head_dim > 0 and (head_dim & (head_dim - 1)) == 0, \
        f"[hpu-kvq-ext] head_dim must be a power of two for the exact WHT, got {head_dim}"
    H = torch.ones(1, 1, dtype=torch.float64)
    while H.shape[0] < head_dim:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    # Plain symmetric WHT (the reference scheme, CPU-validated at 0.84% attn err on real M3 L55).
    # R is symmetric + self-inverse (R@R==I) so write x@R and read x@R.T==x@R both use this matrix.
    # (An earlier random-sign variant was added on a wrong "concentration" hypothesis — the 96%-zero
    # reading was just short-prompt PADDING, not spiky activations — and reverted to match the ref.)
    return (H / math.sqrt(head_dim)).to(torch.bfloat16)


@functools.lru_cache(maxsize=16)
def rotation(head_dim: int, device_str: str) -> torch.Tensor:
    # HPU_KVQ_NOROT=1: R=identity DIAGNOSTIC. Isolates the WHT rotation from the fetch/write
    # mechanism -- with NOROT + fp8-V the wire is ~stock fp8 (no rotation, no int packing), so if
    # it degrades the bug is in our gather/index path; if it's coherent the bug is the rotation.
    if os.environ.get("HPU_KVQ_NOROT", "0").strip().lower() in ("1", "true", "yes", "on"):
        return torch.eye(head_dim, dtype=torch.bfloat16, device=device_str)
    return _rotation_cpu(head_dim).to(device_str)


# --------------------------------------------------------------------------
# torch.compiler.disable: custom OpBackend recipes have NO shape-agnostic SIF,
# so a dynamo/torch.compile capture crashes the worker ("[SHAPE AGNOSTIC] Cache
# hit Synapse shape inference failed!"). Keep the op calls on the eager path
# (per-shape recipes); under PT_HPU_LAZY_MODE=1 there is no dynamo and the lazy
# .so captures into the HPU graph. This forces a graph break at each KV write/
# read under torch.compile (correctness-safe; a perf cost -- see report).
# --------------------------------------------------------------------------
# ==========================================================================
# REUSABLE TORCH int4-KV core (HPU_KVQ_BACKEND=torch, the DEFAULT). Pure tensor
# ops -> works on ANY head_dim (power-of-2 for the exact WHT) and ANY model class
# (GQA K/V heads here; the SAME primitives quant the MLA latent via the oot_mla
# adapter). Robust: no custom TPC .so (the k8v4 kernel SIGSEGVs on a bridge
# Double->Float shift-constant bug). Byte-goal-identical scheme to the .so /
# ref: WHT rotate -> fp8 K (absmax/448) + signed int4 V (absmax/qmax, packed
# L codes per int32). HPU_KVQ_BACKEND=tpc restores the .so path once fixed.
# ==========================================================================
_KVQ_BACKEND = os.environ.get("HPU_KVQ_BACKEND", "torch").strip().lower()
# fp8_e4m3fn's true max is 448, BUT on-card diagnosis (KVQ_RT value dump) proved Gaudi's
# float8_e4m3fn->bfloat16 conversion MISHANDLES the top exponent bin [256, 448]: any element that
# scales into it comes back inf/NaN -> nan_to_num zeros it. That silently zeroed the LARGEST (most
# attention-relevant) components of every fp8 K/V vector -> the "correct-first-token-then-garbage"
# degradation in EVERY KVQ mode (K is fp8 in all of them). Scaling so absmax maps to 224 (= 1.75*2^7,
# the top of the SECOND-highest exponent) keeps every value below the broken bin. fp8 relative
# precision is constant across exponents, so this costs ZERO accuracy -- it only avoids the HW bug.
_FP8_E4M3_MAX = 224.0

# Lloyd-Max 4-bit centroids from vLLM UPSTREAM get_centroids(128,4) — the optimal NON-uniform
# quantizer for a coordinate of an L2-normalized (~Gaussian) vector. Uniform int4 (round to [-7,7])
# is far worse on a Gaussian -> word-salad on real M3; these centroids are what makes int4-V coherent
# (this is THE difference between my first impl and the working upstream turboquant).
_TQ_CENTROIDS_4BIT = (
    -0.2413817048072815, -0.18282662332057953, -0.14300189912319183, -0.11103852838277817,
    -0.08330143988132477, -0.058060530573129654, -0.034306857734918594, -0.011351509019732475,
    0.011351509019732475, 0.034306857734918594, 0.058060530573129654, 0.08330143988132477,
    0.11103852838277817, 0.14300189912319183, 0.18282662332057953, 0.2413817048072815)
_CENT_CACHE: dict = {}


def _centroids_midpoints(v_bits: int, device):
    """(centroids[2^v_bits], midpoints[2^v_bits-1]) on device. 4-bit only for now."""
    key = (v_bits, str(device))
    cm = _CENT_CACHE.get(key)
    if cm is None:
        assert v_bits == 4, f"centroid-V only implemented for 4-bit, got {v_bits}"
        c = torch.tensor(_TQ_CENTROIDS_4BIT, dtype=torch.float32, device=device)
        _CENT_CACHE[key] = cm = (c, (c[:-1] + c[1:]) / 2)
    return cm


def _pack_subbyte(codes: torch.Tensor, v_bits: int) -> torch.Tensor:
    """codes [N,D] int (in [-qmax,qmax]) -> [N, D/L] int32, L codes/lane (L=8 for 4-bit)."""
    L = 8 if v_bits == 4 else 16
    mask = 0xF if v_bits == 4 else 0x3
    N, D = codes.shape
    low = codes.to(torch.int32) & mask                     # two's-complement low bits [N,D]
    g = low.view(N, D // L, L)                             # [N, Lv, L]
    shifts = (torch.arange(L, device=codes.device, dtype=torch.int32) * v_bits)  # [L]
    return (g << shifts).sum(dim=-1).to(torch.int32)       # disjoint fields => sum == or; [N,Lv]


def _unpack_subbyte(packed: torch.Tensor, D: int, v_bits: int, signed: bool = True) -> torch.Tensor:
    """[N, D/L] int32 -> [N,D] b-bit codes. signed=True sign-extends ([-2^(b-1),2^(b-1)-1]);
    signed=False returns the raw unsigned code [0, 2^b-1] (centroid-index path)."""
    L = 8 if v_bits == 4 else 16
    mask = 0xF if v_bits == 4 else 0x3
    N = packed.shape[0]
    shifts = (torch.arange(L, device=packed.device, dtype=torch.int32) * v_bits)  # [L]
    codes = (packed.to(torch.int32).unsqueeze(-1) >> shifts) & mask               # [N, Lv, L]
    if signed:
        codes = torch.where(codes >= (1 << (v_bits - 1)), codes - (1 << v_bits), codes)  # sign-extend
    return codes.reshape(N, D)


def _kv_rotate_quant_torch(k2d, v2d, v_bits):
    D = int(k2d.shape[-1])
    R = rotation(D, str(k2d.device))                        # [D,D] bf16 symmetric WHT
    k_rot = (k2d.to(torch.bfloat16) @ R).float()            # [N,D]
    v_rot = (v2d.to(torch.bfloat16) @ R).float()
    k_scale = (k_rot.abs().amax(-1, keepdim=True) / _FP8_E4M3_MAX).clamp_min(1e-12)   # [N,1]
    k_fp8 = (k_rot / k_scale).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    if v_bits >= 8:  # k8v8: store V as fp8 like K (near-lossless), no int packing
        v_scale = (v_rot.abs().amax(-1, keepdim=True) / _FP8_E4M3_MAX).clamp_min(1e-12)
        v_fp8 = (v_rot / v_scale).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn)  # [N,D]
        return k_fp8, k_scale.float(), v_fp8, v_scale.float()
    # V: Lloyd-Max CENTROID quant (matches vLLM upstream turboquant; uniform int4 = word-salad).
    # Scale = per-token L2 norm (the centroids are optimal for a coordinate of a unit-L2 vector).
    _cents, mids = _centroids_midpoints(v_bits, v_rot.device)
    v_scale = v_rot.norm(dim=-1, keepdim=True).clamp_min(1e-12)                       # [N,1]
    v_unit = (v_rot / v_scale).contiguous()                                          # ~centroid dist
    v_codes = torch.bucketize(v_unit, mids).to(torch.int32)                           # [N,D] unsigned 0..15
    return k_fp8, k_scale.float(), _pack_subbyte(v_codes, v_bits), v_scale.float()


def _kv_dequant_unrotate_torch(k_fp8, k_scale, v_packed, v_scale, v_bits, out_fp8):
    D = int(k_fp8.shape[-1])
    Rt = rotation(D, str(k_fp8.device)).t().contiguous()    # R.T un-rotates (R = Hn@diag(s), R@R.T=I)
    # Sanitize the fp8 K read FIRST: the KV cache planes are allocated UNINITIALIZED, so
    # padding/unwritten slots hold garbage fp8 bytes (0x7F/0xFF == NaN in e4m3) -> NaN K ->
    # NaN attention scores that survive masking -> null tokens. Padding is masked out anyway,
    # so forcing it to 0 is correct. (Root cause = allocation; this is the robust guard.)
    k_fp8 = torch.nan_to_num(k_fp8.to(torch.bfloat16).float(), nan=0.0, posinf=0.0, neginf=0.0)
    k_scale = torch.nan_to_num(k_scale.float(), nan=0.0, posinf=0.0, neginf=0.0)
    k_rot = k_fp8 * k_scale                                                           # [N,D]
    k = (k_rot.to(torch.bfloat16) @ Rt)                     # unrotate -> [N,D] bf16
    v_scale = torch.nan_to_num(v_scale.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if v_bits >= 8:  # k8v8: V stored as fp8 (like K)
        v_deq = torch.nan_to_num(v_packed.to(torch.bfloat16).float(), nan=0.0, posinf=0.0, neginf=0.0)
    else:  # Lloyd-Max centroid dequant: unsigned code -> centroid value * per-token L2 scale
        cents, _mids = _centroids_midpoints(v_bits, k_fp8.device)
        codes = _unpack_subbyte(v_packed, D, v_bits, signed=False).long().clamp_(0, cents.numel() - 1)
        v_deq = cents[codes]                                                          # [N,D] ~unit-norm
        # NORM CORRECTION (upstream '_nc'): renormalize to EXACT unit norm so the recovered vector
        # magnitude == the stored ||v_orig|| (v_scale). Removes the residual magnitude drift that
        # compounds across decode steps into repetition.
        v_deq = v_deq / v_deq.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    v = ((v_deq * v_scale).to(torch.bfloat16) @ Rt)                                   # unrotate
    k = torch.nan_to_num(k, nan=0.0, posinf=0.0, neginf=0.0)
    v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    if int(out_fp8):
        k = k.to(torch.float8_e4m3fn)
    return k, v


@torch.compiler.disable
def kv_rotate_quant_op(k2d: torch.Tensor, v2d: torch.Tensor, v_bits: int):
    """k2d,v2d [N,D] (any dtype -> bf16). Returns (k_fp8, k_scale, v_packed, v_scale)."""
    if _KVQ_BACKEND == "torch":
        return _kv_rotate_quant_torch(k2d, v2d, int(v_bits))
    R = rotation(int(k2d.shape[-1]), str(k2d.device))
    out = _ensure("kv_rotate_quant")(k2d.contiguous().to(torch.bfloat16),
                                     v2d.contiguous().to(torch.bfloat16), R, int(v_bits))
    return out[0], out[1], out[2], out[3]


@torch.compiler.disable
def kv_dequant_unrotate_op(k_fp8, k_scale, v_packed, v_scale, v_bits: int, out_fp8: int = 0):
    """Inverse of kv_rotate_quant_op. Returns (k, v) [N,D] (K bf16 if out_fp8==0)."""
    if _KVQ_BACKEND == "torch":
        return _kv_dequant_unrotate_torch(k_fp8, k_scale, v_packed, v_scale, int(v_bits), int(out_fp8))
    R = rotation(int(k_fp8.shape[-1]), str(k_fp8.device))  # R_t == R for symmetric WHT
    out = _ensure("kv_dequant_unrotate")(k_fp8.contiguous(), k_scale.contiguous(),
                                         v_packed.contiguous(), v_scale.contiguous(),
                                         R, int(v_bits), int(out_fp8))
    return out[0], out[1]


# ==========================================================================
# MLA LATENT int4 STORE (GLM/DeepSeek). The MLA cache is ONE combined latent
# vector [T, kv_lora_rank + rope]; we int4-store the leading kv_lora latent and
# keep the rope tail full-precision -- BYTE-IDENTICAL to the on-card-TEXT-VALIDATED
# fake-quant (vllm_gaudi oot_mla._mla_fake_quant_latent, int4rot g64 uniform), but
# STORING the codes for the real capacity win instead of dequant-in-place. fp32 WHT
# to match the fake-quant exactly. NO fp8 here (pure int4 + bf16 rope) -> the fp8
# top-bin bug does not apply to this path.
# ==========================================================================
@functools.lru_cache(maxsize=8)
def _rotation_fp32(n: int, device_str: str) -> torch.Tensor:
    """fp32 self-inverse WHT (/sqrt(n)), byte-identical to oot_mla._wht_matrix."""
    M = torch.ones(1, 1, dtype=torch.float32)
    while M.shape[0] < n:
        M = torch.cat([torch.cat([M, M], 1), torch.cat([M, -M], 1)], 0)
    return (M[:n, :n] / (float(n) ** 0.5)).to(device_str)


@torch.compiler.disable
def mla_latent_quant_op(latent, kv_lora_rank: int, v_bits: int = 4, group: int = 64):
    """latent [T, L+rope] -> (packed [T, L/(32//v_bits)] int32, scale [T, L/g] f32, rope [T, rope] bf16)."""
    D = int(kv_lora_rank)
    lat = latent[..., :D].float()
    rope = latent[..., D:]
    H = _rotation_fp32(D, str(latent.device))
    lat_rot = lat @ H                                                     # [T, D]
    g = group if D % group == 0 else D
    T = lat_rot.shape[0]
    qmax = float(2 ** (v_bits - 1) - 1)                                   # int4 -> 7 (signed)
    lat_g = lat_rot.view(T, D // g, g)
    scale = lat_g.abs().amax(-1, keepdim=True).clamp_min(1e-8) / qmax     # [T, D/g, 1]
    codes = torch.clamp(torch.round(lat_g / scale), -qmax, qmax).view(T, D).to(torch.int32)  # signed
    packed = _pack_subbyte(codes, v_bits)                                # [T, D/(32//v_bits)]
    return packed, scale.view(T, D // g).float(), rope.to(torch.bfloat16)


@torch.compiler.disable
def mla_latent_dequant_op(packed, scale, rope, kv_lora_rank: int, v_bits: int = 4, group: int = 64):
    """Inverse of mla_latent_quant_op. -> [N, L+rope] bf16 (the full latent, ready for MQA/MHA)."""
    D = int(kv_lora_rank)
    g = group if D % group == 0 else D
    N = packed.shape[0]
    codes = _unpack_subbyte(packed, D, v_bits, signed=True).float()      # [N, D] signed int4
    deq = (codes.view(N, D // g, g) * scale.view(N, D // g, 1)).view(N, D)
    H = _rotation_fp32(D, str(packed.device))                            # self-inverse -> unrotate
    lat = deq @ H                                                        # [N, D]
    return torch.concat([lat.to(torch.bfloat16), rope.to(torch.bfloat16)], dim=-1)
