# Lưu đồ thuật toán CNN và quá trình học

> Hai lưu đồ ASCII cùng phong cách box-art với Hình 3.1 (CHUONG3.md). Copy vào chỗ
> muốn trong CHUONG3.md và đánh lại số Hình cho đúng thứ tự.

---

## Lưu đồ A — Thuật toán suy luận CNN (forward / inference)

Đây là đường đi của **một** mẫu ECG qua mạng khi phân loại (không cập nhật trọng số).
Số liệu tầng lấy từ Bảng 3.3; đường ống INT8 lấy từ Mục 3.1.3.

```
        Đầu vào: ECG lead II, 2500 mẫu INT8  (input_shift = 2)
                              │
                              ▼
   ┌─────────────────────── Conv1 ───────────────────────┐
   │  Tích chập 1D: Cin=1, Cout=4, K=5, pad=2, stride=1   │
   │  acc_int32 = Σ(x·w) + bias                            │
   │  Tái tỉ lệ: round_half_up(acc / 2^nb), nb=8          │
   │  clamp[-127,127]   ·   (KHÔNG ReLU)                   │
   │  MaxPool K=5, S=5                                     │
   └────────────────────────┬─────────────────────────────┘
                            ▼  4 × 500
   ┌─────────────────────── Conv2 ───────────────────────┐
   │  Cin=4, Cout=4, K=5, pad=2 · nb=6 · KHÔNG ReLU        │
   │  MaxPool /5                                           │
   └────────────────────────┬─────────────────────────────┘
                            ▼  4 × 100
   ┌─────────────────────── Conv3 ───────────────────────┐
   │  Cin=4, Cout=8, K=5, pad=2 · nb=6 · KHÔNG ReLU        │
   │  MaxPool /5                                           │
   └────────────────────────┬─────────────────────────────┘
                            ▼  8 × 20
   ┌─────────────────────── Conv4 ───────────────────────┐
   │  Cin=8, Cout=8, K=5, pad=2 · nb=7 · ✅ CÓ ReLU        │
   │  MaxPool /5                                           │
   └────────────────────────┬─────────────────────────────┘
                            ▼  8 × 4
   ┌───────────── GAP (Global Average Pool) ──────────────┐
   │  gap = floor( Σ_4(x) / 4 ) = (Σ) >> 2  (chia nguyên) │
   └────────────────────────┬─────────────────────────────┘
                            ▼  8 × 1
   ┌──────────────── FC (Fully-Connected) ────────────────┐
   │  logit[j] = Σ_8 (gap · w_fc[j]) ,  j = 0..3           │
   │  nb=0 → logit INT32 thô (không tái tỉ lệ)             │
   └────────────────────────┬─────────────────────────────┘
                            ▼  4 logit
   ┌──────────────────── Argmax ──────────────────────────┐
   │  class = arg max_j logit[j]                           │
   └────────────────────────┬─────────────────────────────┘
                            ▼
             Nhãn dự đoán ∈ {AFIB=0, GSVT=1, SB=2, SR=3}
```

**Hình 3.x — Lưu đồ suy luận (forward) của ECG-1DCNN.** ReLU chỉ ở Conv4; ba tầng đầu giữ
đặc trưng âm. Tái tỉ lệ mọi tầng dùng dịch bit + round-half-up (0 DSP).

---

## Lưu đồ B — Quá trình học và cập nhật trọng số

Toàn bộ giai đoạn phần mềm gồm ba pha huấn luyện nối tiếp (float32 → cắt tỉa+tinh chỉnh →
QAT). Số liệu lấy từ Bảng 3.4, 3.5 và Mục 3.1.3.

```
              ┌─────────────────────────────────────────────┐
              │  PHA 1 — Huấn luyện float32 (cơ sở)          │
              └─────────────────────────────────────────────┘
   Khởi tạo trọng số (Kaiming/uniform, PyTorch)  ·  1244 tham số
                              │
                              ▼
        ┌──────────── vòng lặp epoch ─────────────┐
        │                                         │
        │   forward: logit = CNN(x)  (float32)    │
        │              │                          │
        │              ▼                          │
        │   loss = CrossEntropy(logit, y)         │
        │              │                          │
        │              ▼                          │
        │   backward: ∇w = ∂loss/∂w  (autograd)   │
        │              │                          │
        │              ▼                          │
        │   cập nhật: w ← Adam(w, ∇w, lr=1e-3)    │
        │              │                          │
        │        val_acc tăng? ── giữ best ──┐    │
        └──────────────┬────────────────────┘    │
                       │ hội tụ / hết epoch       │
                       ▼                          │
              best_model.pth  (float32, 1244) ◄───┘
                       │
                       ▼
   ┌─────────────────────────────────────────────────────┐
   │  PHA 2 — Cắt tỉa kênh có cấu trúc + tinh chỉnh        │
   └─────────────────────────────────────────────────────┘
   Xếp hạng bộ lọc theo chuẩn L1 (+ Taylor bậc 1)
                       │
                       ▼
   Loại bộ lọc yếu:  Conv2 8→4 , Conv4 16→8 , FC vào 16→8
                       │  → 640 tham số (kênh 4,4,8,8)
                       ▼
   Fine-tune 2 pha (Adam, CrossEntropy):
     · pha 1: 30 epoch, lr=1e-3  (phục hồi sau cú sốc cắt tỉa)
     · pha 2: 20 epoch, lr=1e-4  (tinh chỉnh mịn)
                       │
                       ▼
            best_model_pruned.pth  (float32, 640)
                       │
                       ▼
   ┌─────────────────────────────────────────────────────┐
   │  PHA 3 — QAT power-of-2 (fake-quant, INT8)            │
   └─────────────────────────────────────────────────────┘
   Calibrate: shift_bits = floor(log2(127/abs_max)) mỗi tensor
              → nb {8,6,6,7,0} · w_shift {6,6,6,7,8} · input_shift 2
                       │
                       ▼
        ┌──────────── vòng lặp epoch QAT ─────────┐
        │                                         │
        │   forward (fake-quant):                 │
        │     ŵ = clamp(round(w·2^s),-127,127)/2^s│  ← giả lập INT8
        │     logit = CNN(x; ŵ)                   │     round-half-up
        │              │                          │
        │              ▼                          │
        │   loss = CrossEntropy(logit, y)         │
        │              │                          │
        │              ▼                          │
        │   backward qua STE:                     │
        │     ∂round/∂w ≈ 1  (bỏ qua bậc thang)   │  ← gradient chảy
        │              │                          │     qua điểm không
        │              ▼                          │     khả vi
        │   cập nhật: w ← Adam(w, ∇w)  (float32)  │  ← lưu w ĐỘ PHÂN
        └──────────────┬──────────────────────────┘     GIẢI CAO
                       │ hội tụ
                       ▼
   Convert: w_int8 = clamp(round(w·2^w_shift),-127,127)
            bias_int32 = round(b·2^nb) , little-endian
                       │
                       ▼
            model_qat_int8.pth  (INT8, bit-exact với RTL)
                       │
                       ▼
   Trích xuất → flat_weights.hex (580 INT8) → nạp phần cứng
```

**Hình 3.y — Lưu đồ quá trình học ba pha.** Chu trình huấn luyện chung là
forward → tính loss (cross-entropy) → backward (autograd) → cập nhật Adam. QAT khác ở chỗ
forward chèn bước lượng tử-giả (fake-quant) và backward dùng bộ ước lượng thẳng (STE) để
gradient vượt qua phép làm tròn không khả vi; trọng số float độ phân giải cao vẫn được giữ
để cập nhật, chỉ convert sang INT8 sau khi hội tụ.
