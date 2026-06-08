"""
Probe: does FC bias change argmax on Chapman test set?

Compares 3 FC-bias handling variants in the INT8 hardware forward pass:
  (0) no bias                  — what the RTL currently does
  (1) round(b_float * 2^0)     — what int8_forward / export currently compute (nb_fc=0)
  (2) round(b_float * 2^w_shift_fc) — bias in the SAME scale as the FC logits

Logit domain reasoning:
  fc_acc = Σ(gap_int8[2^0] * w_fc_int8[2^w_shift_fc]) → logits live at scale 2^w_shift_fc.
  So a "correct" bias must be scaled by 2^w_shift_fc to be commensurate with the logits.

Reuses int8_forward up to GAP, then re-runs FC with each bias variant.
Read-only: loads existing checkpoint, writes nothing.
"""
import sys, os
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.dataset import get_dataloaders, CLASS_NAMES
from quantization.qat_int8 import (
    ECG_1DCNN_QAT, round_shift,
)

CKPT = 'results/qat_int8/model_qat_int8.pth'
DATA = '../../data/Chapman'


def gap_int8(qat_model, x, w_int8, b_int8, nb, input_shift):
    """Run INT8 forward up to and including GAP. Returns gap_int8 (B,8)."""
    device = next(qat_model.parameters()).device
    if x.dim() == 2:
        x = x.unsqueeze(1)
    x = torch.clamp(torch.round(x * (2.0 ** input_shift)), -127, 127)

    def conv(x, name):
        w = torch.tensor(w_int8[name].astype(np.float32)).to(device)
        n = nb[name]
        bsc = torch.tensor(np.round(b_int8[name] * (2.0 ** n)).astype(np.float32)).to(device)
        out = F.conv1d(x, w, bsc, padding=getattr(qat_model, name).padding)
        return torch.clamp(round_shift(out, n), -127, 127)

    x = qat_model.pool1(conv(x, 'conv1'))
    x = qat_model.pool2(conv(x, 'conv2'))
    x = qat_model.pool3(conv(x, 'conv3'))

    w4 = torch.tensor(w_int8['conv4'].astype(np.float32)).to(device)
    n4 = nb['conv4']
    b4 = torch.tensor(np.round(b_int8['conv4'] * (2.0 ** n4)).astype(np.float32)).to(device)
    x = torch.clamp(round_shift(F.conv1d(x, w4, b4, padding=qat_model.conv4.padding), n4), -127, 127)
    x = torch.clamp(x, min=0)
    x = qat_model.pool4(x)
    x = qat_model.gap(x).squeeze(-1)   # (B, 8) integer-valued floats
    return x


def main():
    device = torch.device('cpu')
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)

    qat_model = ECG_1DCNN_QAT(
        c1_out=ckpt['c1_out'], c2_out=ckpt['c2_out'],
        c3_out=ckpt['c3_out'], c4_out=ckpt['c4_out'],
    ).to(device)
    qat_model.load_state_dict(ckpt['model_state_dict'])
    qat_model.eval()

    w_int8 = {k: np.array(v, dtype=np.int8) for k, v in ckpt['w_int8'].items()}
    b_int8 = {k: np.array(v, dtype=np.float64) for k, v in ckpt['b_int8'].items()}
    nb = ckpt['nb']
    w_shift = ckpt['w_shift']
    input_shift = ckpt['input_shift_bits']

    b_fc = b_int8['fc']
    ws_fc = w_shift['fc']
    print(f"FC bias float      : {b_fc}")
    print(f"w_shift['fc']      : {ws_fc}")
    print(f"round(b * 2^0)     : {np.round(b_fc * 1.0).astype(int)}")
    print(f"round(b * 2^{ws_fc})    : {np.round(b_fc * (2.0**ws_fc)).astype(int)}")
    print()

    w_fc = torch.tensor(w_int8['fc'].astype(np.float32)).to(device)

    b0   = torch.zeros(4)                                                     # no bias
    b_s0 = torch.tensor(np.round(b_fc * 1.0).astype(np.float32))             # 2^0 (current)
    b_s8 = torch.tensor(np.round(b_fc * (2.0 ** ws_fc)).astype(np.float32))  # 2^w_shift_fc

    _, _, test_loader = get_dataloaders(DATA, batch_size=128, num_workers=0)

    P = {'no_bias': [], 'bias_x1': [], 'bias_x256': []}
    labels = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(device)
            y = batch[1]
            g = gap_int8(qat_model, x, w_int8, b_int8, nb, input_shift)
            P['no_bias'].append(F.linear(g, w_fc, b0).argmax(1).cpu().numpy())
            P['bias_x1'].append(F.linear(g, w_fc, b_s0).argmax(1).cpu().numpy())
            P['bias_x256'].append(F.linear(g, w_fc, b_s8).argmax(1).cpu().numpy())
            labels.append(y.numpy())

    labels = np.concatenate(labels)
    for k in P:
        P[k] = np.concatenate(P[k])

    print(f"Test set size: {len(labels)}")
    print()
    print(f"{'variant':<12} {'accuracy':>10} {'F1-macro':>10} {'!=no_bias':>10}")
    from sklearn.metrics import f1_score
    for k in ['no_bias', 'bias_x1', 'bias_x256']:
        acc = (P[k] == labels).mean()
        f1 = f1_score(labels, P[k], average='macro')
        diff = (P[k] != P['no_bias']).sum()
        print(f"{k:<12} {acc*100:>9.2f}% {f1:>10.4f} {diff:>10d}")

    print()
    print(f"argmax(bias_x1)   == argmax(no_bias) on all samples? "
          f"{(P['bias_x1'] == P['no_bias']).all()}")
    print(f"argmax(bias_x256) == argmax(no_bias) on all samples? "
          f"{(P['bias_x256'] == P['no_bias']).all()}")


if __name__ == '__main__':
    main()
