"""HPUMLAImpl.forward_mha compilation fix — now a no-op.

HISTORY:
The original root cause of the aten.slice.Tensor crash was that
GPTQDequantToBF16LinearMethod.apply() had @torch.compiler.disable, which created
a graph break inside the compiled region. This caused the HPU backend to receive
opaque tensors for the kv_b_proj output, making the subsequent:

    kv_nope = self.kv_b_proj(k_c_normed)[0]
        .view(-1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
    k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

fail with:
  AttributeError: 'NoneType' object has no attribute 'file'
  While executing %slice_1 : [num_users=1] = call_function[
      target=torch.ops.aten.slice.Tensor](args = (%view, 2, 0, 192))

The workaround was to force forward_mha into eager mode via @torch.compiler.disable.
However, this caused forward_mha to run in eager while the surrounding code was
compiled, creating a compiler boundary that produced wrong outputs (token salad).

FIX APPLIED (moe_wna16_hpu.py): Removed @torch.compiler.disable from apply() so
it is now a plain traceable F.linear call. This eliminates the original graph break,
meaning the aten.slice crash no longer occurs, and forward_mha can safely compile.

This module is now a no-op — forward_mha compiles normally as part of the regional
HPU graph, which is required for numerically correct outputs.
"""


def register() -> None:
    """No-op: the apply() fix in moe_wna16_hpu.py eliminates the root cause.

    Previously wrapped HPUMLAImpl.forward_mha with @torch.compiler.disable to
    avoid an aten.slice crash. That crash was caused by a graph break in apply()
    (which had @torch.compiler.disable). Since apply() no longer has that decorator,
    forward_mha can compile normally inside the HPU regional graph.

    Forcing forward_mha into eager mode while the surrounding code is compiled
    creates a compiler boundary that corrupts the tensor metadata, producing
    wrong outputs (token salad: 'iecieciec' pattern). Allowing full compilation
    fixes this.
    """
    pass
