"""
Matched-Filter CNN — Model 6 (Sensors 2023, 23/3/1365)
======================================================
Faithful PyTorch reproduction of the paper's best model:

  derivative beat (SEG=64) ─► Conv1D(13 filters, k=NK=32, FROZEN=MF templates)
                              ─► BatchNorm ─► Tanh ─► GlobalMaxPool ─► [13]
  4 RR features ──────────────► Dense(32) ReLU ─► Dense(16) ReLU
                              ─► Dense(8) ReLU ─► [8]
  concat[13+8=21] ───────────► Linear ─► softmax(3)   (N / S / V)

The conv kernels are initialized to the matched-filter templates (per-sub-class
mean derivative beat over DS1) and FROZEN (requires_grad=False) — Model 6.
The 13 templates of length SEG=64 are centre-cropped to NK=32 (paper: "discarded
to reduce ... from 64 to 32").
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ECG_MFCNN(nn.Module):
    def __init__(self, n_templates=13, nk=32, seg=64, n_rr=4, num_classes=3,
                 templates=None, freeze_conv=True, normalize_templates=False):
        super().__init__()
        self.n_templates = n_templates
        self.nk = nk
        self.num_classes = num_classes
        self.normalize_templates = normalize_templates

        # 'same' padding so GlobalMaxPool sees the full correlation profile.
        self.conv = nn.Conv1d(1, n_templates, kernel_size=nk,
                              padding=nk // 2, bias=False)
        self.bn   = nn.BatchNorm1d(n_templates)

        if templates is not None:
            self._init_matched_filter(templates)
        if freeze_conv:
            self.conv.weight.requires_grad_(False)

        # RR branch: 4 -> 32 -> 16 -> 8 (paper's dense stack)
        self.rr_mlp = nn.Sequential(
            nn.Linear(n_rr, 32), nn.ReLU(),
            nn.Linear(32, 16),   nn.ReLU(),
            nn.Linear(16, 8),    nn.ReLU(),
        )
        self.head = nn.Linear(n_templates + 8, num_classes)

    def _init_matched_filter(self, templates):
        """templates: (n_templates, seg) float. Centre-crop to nk and load as
        conv kernels (one filter per sub-class template)."""
        t = torch.as_tensor(templates, dtype=torch.float32)
        seg = t.shape[1]
        s = (seg - self.nk) // 2
        t = t[:, s:s + self.nk]                       # (n_templates, nk)
        if self.normalize_templates:
            # optional L2-normalize per template (NOT in the paper; the paper
            # loads raw mean-derivative templates and lets BatchNorm rescale).
            t = t / (t.norm(dim=1, keepdim=True) + 1e-8)
        with torch.no_grad():
            self.conv.weight.copy_(t.unsqueeze(1))    # (n_templates, 1, nk)

    def forward(self, x, rr):
        if x.dim() == 2:
            x = x.unsqueeze(1)                        # (B, 1, SEG)
        h = torch.tanh(self.bn(self.conv(x)))         # (B, 13, L)
        h = h.max(dim=-1).values                      # GlobalMaxPool -> (B, 13)
        r = self.rr_mlp(rr)                           # (B, 8)
        return self.head(torch.cat([h, r], dim=1))    # (B, 3)

    def predict(self, x, rr):
        return self.forward(x, rr).argmax(dim=1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_all(self):
        return sum(p.numel() for p in self.parameters())

    def layer_summary(self):
        return (f"ECG_MFCNN trainable={self.count_parameters()} "
                f"total={self.count_all()}")


def build_mfcnn(templates=None, freeze_conv=True, nk=32, seg=64,
                n_templates=13, n_rr=4, num_classes=3,
                normalize_templates=False):
    return ECG_MFCNN(n_templates=n_templates, nk=nk, seg=seg, n_rr=n_rr,
                     num_classes=num_classes, templates=templates,
                     freeze_conv=freeze_conv,
                     normalize_templates=normalize_templates)


if __name__ == "__main__":
    import numpy as np
    tpl = np.random.randn(13, 64).astype(np.float32)
    m = build_mfcnn(templates=tpl)
    print(m.layer_summary())
    x = torch.randn(4, 64); rr = torch.randn(4, 4)
    print("out", tuple(m(x, rr).shape))
