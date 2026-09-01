# Phương trình quy đổi Float → INT8 (Power-of-2 QAT)

> Dạng văn bản Unicode — dán thẳng vào Word hiển thị đúng (không cần LaTeX).
> Ký hiệu: ⌊·⌉ = làm tròn gần nhất; ⌊·⌋ = làm tròn xuống;
> clamp(x,a,b) = min(max(x,a),b); ≫ = dịch phải bit.
> Mỗi layer ℓ có: số bit dịch rescale nb⁽ℓ⁾, số bit dịch trọng số sw⁽ℓ⁾.

---

(1) Chọn số bit dịch (power-of-2, tính một lần khi calibrate, mỗi layer):

        s = ⌊ log₂( 127 / max|v| ) ⌋ ,        scale = 2ˢ

trong đó max|v| là biên độ tuyệt đối lớn nhất của đại lượng cần lượng tử hóa
(trọng số hoặc activation) trên tập calibrate.

(2) Lượng tử hóa input ECG (s_in = 2):

        x_q = clamp( ⌊ x · 2^(s_in) ⌉ , −127 , 127 )

(3) Lượng tử hóa trọng số:

        w_q = clamp( ⌊ w · 2^(sw⁽ℓ⁾) ⌉ , −127 , 127 )

(4) Lượng tử hóa bias (lưu INT32 little-endian):

        b_q = ⌊ b · 2^(nb⁽ℓ⁾) ⌉

(5) Tích chập + cộng bias (tích lũy INT32):

        a = Σ_c Σ_k  w_q[c,k] · x_q[c, ·]  +  b_q

(6) Rescale round-half-up về INT8:

        y = clamp( ⌊ ( a + 2^(nb−1) ) / 2^(nb) ⌋ , −127 , 127 )

        tương đương:   y = clamp( ( a + 2^(nb−1) ) ≫ nb , −127 , 127 )

→ chỉ gồm MỘT phép cộng + MỘT phép dịch phải, KHÔNG dùng nhân (0 DSP).

(7) Phi tuyến + pooling (ReLU chỉ ở Conv4):

        y ← max(0, y)            (chỉ Conv4)
        y_pool = max over window  y[j]

(8) Global Average Pooling (chia nguyên):

        g = ⌊ ( Σ_{j=1..4} y[j] ) / 4 ⌋ = ( Σ_j y[j] ) ≫ 2

(9) Fully-connected (không rescale, nb_FC = 0) + Argmax:

        logit_m = Σ_{c=1..8}  w_q^FC[m,c] · g[c]
        ŷ = argmax_m  logit_m

---

Tham số cụ thể (Conv1, Conv2, Conv3, Conv4, FC):

        nb = { 8, 6, 6, 7, 0 }
        sw = { 6, 6, 6, 7, 8 }
        s_in = 2
