"""
Model definitions for the Western Negev restoration prioritization project.

Used by Notebook 05 (main model), 06 (experiments), and 07 (inference).
All architectural choices follow Sainte Fare Garnot & Landrieu (2020),
adapted for pixel-level multi-task regression rather than parcel-level
classification.

v3 additions:
    * MaskedMultiTaskLoss — a variant of MultiTaskLoss in which the three
      regression heads only receive gradient from pixels that experienced
      at least one fire in the fire-reference inventory. This reflects the thesis's
      research goal (quantifying post-fire dynamics) and prevents the
      regression predictions on non-burned pixels from being pushed toward
      arbitrary values by the loss.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Positional encoding (Vaswani-style sin+cos)
# ============================================================
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding.

    Encodes the position of each time step into a vector that is added to
    the input embedding. Uses Vaswani-style alternating sin+cos, scaled by
    tau. For sequences spanning ~3000 days, tau=10000 is appropriate.
    """

    def __init__(self, d_model, max_len, tau):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(tau) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)].unsqueeze(0)


# ============================================================
# L-TAE encoder
# ============================================================
class LTAE(nn.Module):
    """Lightweight Temporal Attention Encoder.

    Implements the architecture of Sainte Fare Garnot & Landrieu (2020):
      eq. 1 - channel grouping via tensor reshape
      eq. 2 - positional encoding added before head splitting
      eq. 3 - shared linear projection to keys
      eq. 4 - learned master query per head, scaled dot-product softmax
      eq. 5 - weighted temporal sum, values are input embeddings
      eq. 6 - final MLP on concatenated head outputs
    """

    def __init__(self, d_model, n_head, d_k, max_len, tau,
                 mlp_dims=None, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.n_head  = n_head
        self.d_model = d_model
        self.d_k     = d_k
        self.d_head  = d_model // n_head

        self.pos_enc      = PositionalEncoding(d_model, max_len, tau)
        self.master_query = nn.Parameter(torch.randn(n_head, d_k) * 0.1)
        self.key_proj     = nn.Linear(self.d_head, d_k, bias=False)
        self.dropout      = nn.Dropout(dropout)

        if mlp_dims is None:
            mlp_dims = [d_model]

        mlp_layers = []
        in_dim = d_model
        for out_dim in mlp_dims:
            mlp_layers.append(nn.Linear(in_dim, out_dim))
            mlp_layers.append(nn.LayerNorm(out_dim))
            mlp_layers.append(nn.GELU())
            mlp_layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        self.mlp = nn.Sequential(*mlp_layers)

        self.attention_weights = None

    def forward(self, x, mask=None):
        B, T, D = x.shape
        H = self.n_head

        x = self.pos_enc(x)
        x_heads = x.view(B, T, H, self.d_head)

        keys = self.key_proj(x_heads)
        q    = self.master_query.unsqueeze(0).unsqueeze(0)

        scores = (keys * q).sum(dim=-1) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(
                (~mask).unsqueeze(-1).expand_as(scores),
                float("-inf"),
            )

        attn = F.softmax(scores, dim=1)
        attn = self.dropout(attn)
        self.attention_weights = attn.detach()

        context = (attn.unsqueeze(-1) * x_heads).sum(dim=1)
        context = context.view(B, D)
        return self.mlp(context)


# ============================================================
# Multi-task model
# ============================================================
class FireRecoveryModel(nn.Module):
    """Pixel-level multi-task model for fire dynamics characterisation.

    Heads:
      burned_logits  - per-month binary classification (sigmoid + threshold)
      severity       - regression on analytical severity target
      persistence    - regression on analytical persistence target
      recovery_gap   - regression on analytical recovery-gap target

    The composite RIS is computed post-hoc in Notebook 07 with user-chosen
    weights, supporting the multi-dimensional decision-support framework.
    """

    def __init__(self, n_features, d_model, n_head, d_k,
                 max_len, tau, dropout=0.1):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.ltae = LTAE(d_model=d_model, n_head=n_head, d_k=d_k,
                         max_len=max_len, tau=tau, dropout=dropout)

        # Per-month burned head sees the local embedding plus the global context
        self.head_burned = nn.Sequential(
            nn.Linear(d_model * 2, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        def regression_head():
            return nn.Sequential(
                nn.Linear(d_model, 64), nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1), nn.Sigmoid(),
            )

        self.head_severity     = regression_head()
        self.head_persistence  = regression_head()
        self.head_recovery_gap = regression_head()

    def forward(self, x, mask=None):
        B, T, C = x.shape
        h       = self.input_proj(x)
        context = self.ltae(h, mask)

        context_exp   = context.unsqueeze(1).expand(B, T, -1)
        burned_input  = torch.cat([h, context_exp], dim=-1)
        burned_logits = self.head_burned(burned_input).squeeze(-1)

        return {
            "burned_logits": burned_logits,
            "severity":      self.head_severity(context).squeeze(-1),
            "persistence":   self.head_persistence(context).squeeze(-1),
            "recovery_gap":  self.head_recovery_gap(context).squeeze(-1),
        }


# ============================================================
# Loss functions
# ============================================================
class FocalLoss(nn.Module):
    """Focal loss with mask support for padded positions.

    alpha and gamma follow Lin et al. (2017). The mask zeros out padded
    timesteps so they do not contribute to the loss.
    """

    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets, mask=None):
        bce  = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none")
        prob = torch.sigmoid(logits)
        p_t  = prob * targets + (1 - prob) * (1 - targets)
        loss = bce * ((1 - p_t) ** self.gamma)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * loss

        if mask is not None:
            loss = loss * mask.float()
            return loss.sum() / mask.float().sum().clamp(min=1)
        return loss.mean()


class MultiTaskLoss(nn.Module):
    """Combined loss for burned classification + four regression dimensions.

    Burned loss uses Focal to handle the extreme class imbalance (~0.8%
    positive). Regression losses use plain MSE on all pixels. lambda_aux
    balances the two task types.

    Kept in place for backward compatibility with older experiments and
    ablation studies. New runs should use MaskedMultiTaskLoss (below).
    """

    def __init__(self, alpha, gamma, lambda_aux):
        super().__init__()
        self.focal      = FocalLoss(alpha=alpha, gamma=gamma)
        self.mse        = nn.MSELoss()
        self.lambda_aux = lambda_aux

    def forward(self, outputs, targets, mask):
        loss_burned = self.focal(
            outputs["burned_logits"], targets["burned"], mask)
        loss_sev = self.mse(outputs["severity"],     targets["severity"])
        loss_per = self.mse(outputs["persistence"],  targets["persistence"])
        loss_rec = self.mse(outputs["recovery_gap"], targets["recovery_gap"])
        loss_aux = (loss_sev + loss_per + loss_rec) / 3

        total = loss_burned + self.lambda_aux * loss_aux

        return {
            "total":        total,
            "burned":       loss_burned,
            "auxiliary":    loss_aux,
            "severity":     loss_sev,
            "persistence":  loss_per,
            "recovery_gap": loss_rec,
        }


class MaskedMultiTaskLoss(nn.Module):
    """Combined loss with regression heads masked to burned pixels only.

    The v3 loss reflects the research goal: the thesis quantifies
    post-fire dynamics, so severity, persistence, and recovery_gap only
    have a meaningful interpretation only on pixels that actually
    burned. Non-burned pixels have zero targets for these dimensions
    (they never burned, so there is nothing to recover from), and
    including them in the MSE would bias the regression heads toward
    zero everywhere.

    A pixel is considered burned if it experienced at least one fire
    month in the fire-reference label sequence, i.e. targets["burned"]
    contains at least one 1 for that pixel.

    The burned classification loss remains unchanged: every month of
    every pixel contributes to burned_logits training, so the model
    still sees the full distribution when learning to detect fires.
    """

    def __init__(self, alpha, gamma, lambda_aux):
        super().__init__()
        self.focal      = FocalLoss(alpha=alpha, gamma=gamma)
        self.lambda_aux = lambda_aux

    def _masked_mse(self, preds, targets, sample_mask):
        """Mean squared error computed only over the flagged samples."""
        denom = sample_mask.sum().clamp(min=1)
        err   = ((preds - targets) ** 2) * sample_mask
        return err.sum() / denom

    def forward(self, outputs, targets, mask):
        # Focal loss on the per-month burned classification (unchanged)
        loss_burned = self.focal(
            outputs["burned_logits"], targets["burned"], mask)

        # Per-pixel burned indicator: 1 if the pixel had at least one
        # burned month in the fire-reference labels, 0 otherwise
        burned_pixel = (targets["burned"].sum(dim=1) > 0).float()

        # Regression losses computed only on burned pixels
        loss_sev = self._masked_mse(
            outputs["severity"],     targets["severity"],     burned_pixel)
        loss_per = self._masked_mse(
            outputs["persistence"],  targets["persistence"],  burned_pixel)
        loss_rec = self._masked_mse(
            outputs["recovery_gap"], targets["recovery_gap"], burned_pixel)
        loss_aux = (loss_sev + loss_per + loss_rec) / 3

        total = loss_burned + self.lambda_aux * loss_aux

        return {
            "total":            total,
            "burned":           loss_burned,
            "auxiliary":        loss_aux,
            "severity":         loss_sev,
            "persistence":      loss_per,
            "recovery_gap":     loss_rec,
            "n_burned_pixels":  burned_pixel.sum().detach(),
        }
