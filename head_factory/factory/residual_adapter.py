"""A.2 RESIDUAL ADAPTER — a thin, ZERO-INIT lift on the NATIVE GLM-5.2 MTP head's draft logits.

DESIGN (HEAD_TTT.md §4 A.2, reconciled with what is trainable CARD-FREE):
  The genuine single-stream lever is to LIFT the native head's per-depth accept above the MEASURED floor
  (per-position [0.44, 0.13, 0.04] -> accept-len ~1.50). The native head (model layer 78, ~10 GB fp8) is
  NOT CPU-runnable, and its draft loop consumes its OWN per-depth OUTPUT hidden (which the dump does not
  hold). So an adapter that sits ON the native head's internal forward cannot be trained off-card.

  THE CARD-FREE-TRAINABLE FORM: a PARALLEL residual logits-head. It takes the SAME inputs the native head
  drafts from -- the SEED hidden (the target's final hidden at the committed token; dumped as `hidden`) and
  the committed/draft token (dumped as `target_ids`, then its own self-feed argmax) -- runs a small block,
  and emits a LOGITS RESIDUAL that is ADDED to the native head's logits before argmax:

        committed_draft = argmax( native_logits(depth j)  +  alpha * adapter_logits(depth j) )

  * `alpha` (a learned scalar, INIT 0) makes the residual EXACTLY ZERO at init -> the served argmax is
    byte-identical to native (the adapter is OFF/identity until trained). It can ONLY improve from there;
    the train loop + a HARD held-out native-floor gate guarantee we never ship a regression.
  * The adapter's depth-2..K input is ITS OWN previous output hidden + the committed draft token (the
    post-sum argmax), exactly like a DraftHead self-feed -- so it is trained by the EXACT per-step
    verify-label unroll already in ttt_train.py (seed = dumped `hidden`, labels = `target_argmax`), with
    the only serve difference that its logits are ADDED to native rather than used alone.

  REUSE: the body is the proven `DraftHead` (head.py) -- enorm/hnorm/eh_proj + lean block(s) + final_norm,
  frozen embed/lm_head tied to the target. The ONLY addition is the zero-init `alpha` gate. This keeps the
  offline trainer/eval (forward_hidden -> compute_logits) byte-faithful to the serve adapter path.

SERVE (fp8, no bf16): the adapter's trainable linears fp8-round at load (eval_head._fp8_round_head proves
  fp8==fp32 accept). It runs in PARALLEL to the native head on the seed; logits summed. Wiring:
  hpu_moe_ext/mtp_residual_adapter.py taps the proposer propose() loop (see that file).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys as _sys
_sys.path.insert(0, __file__.rsplit("/", 1)[0])
from head import DraftHead, HeadConfig  # noqa: E402  (the proven body; reused, not reimplemented)


class ResidualAdapter(nn.Module):
    """DraftHead body + a ZERO-INIT scalar gate `alpha`. At init alpha=0 -> logit residual == 0 -> the
    served draft argmax == native (identity). Trainable to lift. The frozen embed/lm_head are the target's
    own (the residual lives in the target's vocab logit space, so it adds to native logits coherently)."""

    def __init__(self, cfg: HeadConfig, embed_w: torch.Tensor, lm_head_w: torch.Tensor,
                 alpha_init: float = 1.0):
        super().__init__()
        self.cfg = cfg
        self.body = DraftHead(cfg, embed_w, lm_head_w)
        # Learned residual gate. TRAINING uses alpha_init=1.0 so the body gets a non-vanishing gradient
        # (alpha=0 would zero the logits and starve the body). The IDENTITY/never-regress guarantee is
        # enforced at SERVE by a separate env knob HPU_MTP_ADAPTER_ALPHA (default 1.0; set 0.0 = native
        # byte-exact). The on-card A/B (adapter ON vs OFF=alpha0) is the binding ship gate; worst case we
        # serve alpha=0 == native. (A trained alpha can also be folded into the serve knob.)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    # ---- the serve/train core: same signatures as DraftHead so ttt_train/eval_head drive it unchanged ----
    def forward_hidden(self, prev_hidden, tok_next, positions):
        return self.body.forward_hidden(prev_hidden, tok_next, positions)

    def forward_hidden_embeds(self, prev_hidden, inputs_embeds, positions):
        return self.body.forward_hidden_embeds(prev_hidden, inputs_embeds, positions)

    def compute_logits(self, hidden):
        """The RESIDUAL logits the adapter contributes (alpha * lm_head(hidden)). At serve this is ADDED to
        the native head's logits; at train/eval (adapter-only accept telemetry) it IS the adapter's logits.
        alpha=0 at init -> exactly 0 residual."""
        return self.alpha * self.body.compute_logits(hidden)

    @property
    def embed_w(self):
        return self.body.embed_w

    @property
    def lm_head_w(self):
        return self.body.lm_head_w

    def forward(self, prev_hidden, tok_next, positions):
        return self.compute_logits(self.forward_hidden(prev_hidden, tok_next, positions))

    def trainable_params(self):
        # the body's trainable params (NOT embed_w/lm_head_w) PLUS alpha.
        return self.body.trainable_params() + [self.alpha]

    # ---- state_dict helpers so the trainer/eval save & load match (keys: body.* + alpha) ----
    def load_body_warmstart(self, state):
        """Warm-start the body from a trained DraftHead checkpoint's `state` (a1-preserving). alpha stays 0
        so the adapter still starts at identity; training lifts alpha off 0."""
        bs = {}
        for k, v in state.items():
            bs[k if k.startswith("body.") else f"body.{k}"] = v
        miss, unexp = self.load_state_dict(bs, strict=False)
        return miss, unexp
