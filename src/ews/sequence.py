"""Small causal sequence encoders (attention-GRU and TCN) + training loop."""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from . import config as C


class AttnGRU(nn.Module):
    """GRU over the past weeks + additive attention pooled with the last state.

    The attention weights say *which past weeks* drove the prediction, which we
    surface in the UI as a per-week explanation.
    """

    def __init__(self, n_features, hidden=48, dropout=0.2):
        super().__init__()
        self.inp = nn.Linear(n_features + 1, hidden)  # +1: "is real week" flag
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.att = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, hidden), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x, mask, return_attn=False):
        h = F.gelu(self.inp(torch.cat([x, mask.unsqueeze(-1)], -1)))
        out, _ = self.gru(h)                      # [B, T, H]
        last = out[:, -1]                         # windows are left-padded -> last step is "now"
        q = last.unsqueeze(1).expand_as(out)
        scores = self.att(torch.cat([out, q], -1)).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, -1)
        ctx = (attn.unsqueeze(-1) * out).sum(1)
        logit = self.head(torch.cat([last, ctx], -1)).squeeze(-1)
        return (logit, attn) if return_attn else logit


class CausalConv(nn.Module):
    def __init__(self, cin, cout, k, d):
        super().__init__()
        self.pad = (k - 1) * d
        self.conv = nn.Conv1d(cin, cout, k, dilation=d)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


class TCN(nn.Module):
    """Dilated causal CNN (dilations 1,2,4 -> receptive field 15 >= 12 weeks)."""

    def __init__(self, n_features, hidden=48, dropout=0.2):
        super().__init__()
        self.inp = nn.Conv1d(n_features + 1, hidden, 1)
        self.blocks = nn.ModuleList([CausalConv(hidden, hidden, 3, d) for d in (1, 2, 4)])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x, mask, return_attn=False):
        z = torch.cat([x, mask.unsqueeze(-1)], -1).transpose(1, 2)
        h = self.inp(z)
        for b in self.blocks:
            h = h + self.drop(F.gelu(b(h)))
        logit = self.head(h[:, :, -1]).squeeze(-1)
        return (logit, None) if return_attn else logit


ARCHS = {"attn_gru": AttnGRU, "tcn": TCN}


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def _batches(n, bs, shuffle, rng):
    idx = rng.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


@torch.no_grad()
def predict_logits(model, X, M, bs=4096):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(model(torch.from_numpy(X[i:i + bs]), torch.from_numpy(M[i:i + bs])).numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def train_seq_model(arch, Xtr, Mtr, ytr, Xva, Mva, yva, params=None, seed=C.SEED, verbose=True):
    """Train with early stopping on validation average precision."""
    p = {**C.SEQ_PARAMS, **(params or {})}
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ARCHS[arch](Xtr.shape[-1], hidden=p["hidden"], dropout=p["dropout"])
    opt = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=1)
    Xtr_t, Mtr_t, ytr_t = map(torch.from_numpy, (Xtr, Mtr, ytr))
    best, best_state, bad = -1.0, None, 0
    for ep in range(p["max_epochs"]):
        model.train()
        tot = 0.0
        for b in _batches(len(Xtr), p["batch_size"], True, rng):
            b = torch.from_numpy(b)
            loss = F.binary_cross_entropy_with_logits(model(Xtr_t[b], Mtr_t[b]), ytr_t[b])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(b)
        ap = average_precision_score(yva, predict_logits(model, Xva, Mva))
        sched.step(ap)
        if verbose:
            print(f"    [{arch}] epoch {ep + 1:02d} loss={tot / len(Xtr):.4f} val_AP={ap:.4f}")
        if ap > best + 1e-4:
            best, best_state, bad = ap, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= p["patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best
