# INTEGRITY_CHECK — Phase K (đạo văn / ý tưởng / AI)

> Mục tiêu: chuẩn bị bản thảo ICDV (`ICDV_draft.md`) qua 3 cổng liêm chính học thuật:
> **(1) đạo văn** (text-similarity), **(2) trùng ý tưởng** (novelty vs prior art),
> **(3) AI-generated detection**. File này gồm 2 phần:
> - **Phần A — Quy trình & công cụ**: dùng công cụ nào, ngưỡng nào, diễn giải kết quả ra sao.
>   Bạn tự chạy (cần tài khoản trường / dịch vụ trả phí + upload bản thảo).
> - **Phần B — Self-audit thủ công**: tôi rà `ICDV_draft.md` offline trong repo, đánh dấu đoạn
>   rủi ro, kiểm cite, soi "giọng AI" để bạn sửa tay.
>
> ⚠️ **Tôi KHÔNG upload bản thảo lên dịch vụ ngoài** (Turnitin/iThenticate/GPTZero…). Phần A là
> hướng dẫn để bạn tự làm; Phần B là kiểm tra offline tôi đã thực hiện trên bản hiện tại.

Ngày: 2026-06-18. Bản thảo soi: `paper/ICDV_draft.md` (8 sections, ~6 trang).

---

## PHẦN A — Quy trình & công cụ

### A.1 Check đạo văn (text similarity)

| Công cụ | Truy cập | Dùng cho | Ngưỡng tham khảo |
|---|---|---|---|
| **Turnitin** | Qua tài khoản trường (LMS/Moodle) | Similarity Index tổng + nguồn khớp | Tổng **< 15–20%**; **không** đoạn nào > 3% từ 1 nguồn |
| **iThenticate** | Trả phí / qua thư viện trường | Bản dành cho paper (loại self-cite) | Tương tự; IEEE/Elsevier hay yêu cầu < 15% |
| **Crossref Similarity Check** | Qua nhà xuất bản khi nộp | Tự động lúc submit | Tùy venue |

**Cách diễn giải đúng:**
- Similarity Index cao **không** đồng nghĩa đạo văn — phần lớn đến từ thuật ngữ kỹ thuật cố định
  ("quantization-aware training", "max-pooling with stride", "patient-independent split"),
  tên dataset, công thức. Đọc **breakdown theo nguồn**, không nhìn con số tổng.
- **Cờ đỏ thật**: một đoạn văn 25+ từ liên tục khớp 1 nguồn → viết lại bằng lời mình + cite.
- **Loại trừ hợp lệ**: references, quoted material, tên riêng, công thức toán.
- Bật **"exclude bibliography"** và **"exclude quotes"** trước khi đọc %.

### A.2 Check trùng ý tưởng (novelty / prior-art)

Đạo văn ý tưởng ≠ đạo văn chữ. Quy trình:
1. **Đối chiếu với direct competitor** (Liu 2023, [reference-paper-liu2023]): bảng phân định
   "cái gì Liu đã có" vs "cái gì là của ta" — xem Phần B.4 dưới. Đây là rào chắn reviewer mạnh nhất.
2. **Google Scholar / IEEE Xplore / Semantic Scholar**: search cụm novelty
   ("round-half-up power-of-two ECG FPGA", "bit-exact INT8 RTL verification ECG") xem có ai
   claim trùng. Lưu ngày + kết quả.
3. **Connected Papers / litmaps**: dựng đồ thị quanh Liu 2023 để chắc không bỏ sót paper gần.
4. Mọi claim "first / novel / unlike prior work" phải có **1 câu phòng thủ** + cite tương ứng.

### A.3 Check AI-generated

| Công cụ | Lưu ý |
|---|---|
| **GPTZero**, **Originality.ai**, **Copyleaks AI** | Trả phí; **độ tin cậy thấp** với văn kỹ thuật — hay false-positive ở câu cấu trúc chặt |
| Tự rà thủ công | **Đáng tin hơn** với paper kỹ thuật — xem Phần B.3 |

**Quan điểm:** AI-detector cho văn học thuật kỹ thuật **không đáng tin** (cả false-positive lẫn
false-negative cao). Cách phòng thủ thật là **làm cho mọi câu có nội dung kiểm chứng được**: số
đo, cite, công thức. Câu nào "kêu" mà rỗng nội dung → đó mới là câu rủi ro, sửa nó dù detector
nói gì. Nếu trường **bắt buộc** điểm AI-detector, chạy GPTZero trên bản cuối và viết lại các đoạn
bị flag theo Phần B.3.

### A.4 Checklist trước nộp
- [ ] Turnitin/iThenticate: tổng < 20%, không nguồn nào > 3%, đã exclude bib+quotes.
- [ ] Mọi đoạn flag > 25 từ liên tục đã viết lại + cite.
- [ ] Bảng phân định novelty vs Liu (Phần B.4) đã rà; mọi "first/novel" có câu phòng thủ.
- [ ] (Nếu trường yêu cầu) GPTZero bản cuối; đoạn flag đã sửa theo B.3.
- [ ] References đầy đủ DOI (xem `REFERENCES.md`); [CITE] placeholder đã thay key thật.

---

## PHẦN B — Self-audit thủ công `ICDV_draft.md` (offline, 2026-06-18)

> Bản thảo do tôi soạn từ số liệu nội bộ (PAPER_DATA.md, SOTA_TABLE.md, Article.xml) — không
> copy-paste từ nguồn ngoài. Dưới đây là rà soát theo từng cổng.

### B.1 Rủi ro đạo văn văn bản — THẤP

Toàn bộ draft là văn paraphrase từ dữ liệu/khái niệm, không có đoạn dán nguyên từ nguồn. Các cụm
**dùng chung trong lĩnh vực** (sẽ làm similarity-tool sáng đèn nhưng **không** là đạo văn):
- "quantization-aware training", "round-half-up", "max-pooling with stride 5",
  "patient-independent 70/15/15", "global average pooling", "macro-F1".
- Mô tả model 4-conv (4-4-8-8, K=5, pad=2) — **trùng kiến trúc với Liu 2023 là CÓ CHỦ Ý** (ta
  cố tình so cùng kiến trúc); cần đảm bảo §2 nói rõ "architecturally identical network … of Liu
  *et al.*" (đã có, dòng 86–90) → đây là **so sánh có cite**, không phải giấu nguồn.

**Hành động:** không có đoạn cần viết lại vì đạo văn. Khi chạy Turnitin, các cụm trên sẽ khớp —
giải trình bằng "common technical terminology", không sửa.

### B.2 Số liệu cần khóa trước khi nộp (tránh "đạo văn số" / số sai) — phải xử lý

Đây không phải đạo văn nhưng là **tính chính trực số liệu** — mọi số phải truy được nguồn:

| Vị trí draft | Số | Trạng thái | Việc |
|---|---|---|---|
| Abstract, §6.1, §8 | Acc 94.65% / F1 0.9396 | ✅ PAPER_DATA (FC-bias retrain) | Dùng nhất quán |
| Table 4 A2 | 94.37% | 🔴 **lệch** với 94.65 headline | Footnote "pre-FC-bias ablation" HOẶC re-gen |
| §6.3 Fmax | 104.85 / 108.94 MHz | ✅ | KHÔNG dùng 137.6 (internal path) |
| §6.3 Energy | 623mW / 10.3µJ / 32.5µJ | 🟠 từ run notes | Xác nhận từ `.pow.rpt` PowerPlay |
| §6.4 on-board | 94.27% (1004/1065) | 🟠 từ run notes | Regenerate log cite-được |
| §2, §6.3 Liu | 92.95% INT8 / 66µs / 66mW | ✅ Article.xml Table 1+3 | OK |
| Table B competitor | Wei/Carreras/… | 🔲 phần lớn từ Liu ref-list | Mở paper gốc trước cite (xem SOTA_TABLE) |

→ **5 mục 🔴/🟠 trùng với "Open items blocking camera-ready" cuối draft** — đã đồng bộ, không phát sinh mới.

### B.3 Soi "giọng AI" — câu cần bạn sửa tay

Các câu/đoạn **văn phong hơi marketing / sweeping**, dễ bị AI-detector flag VÀ dễ bị reviewer cho
là rỗng. Đề xuất tông neutral hơn (bạn quyết, đây không phải lỗi số liệu):

| Vị trí | Câu hiện tại (rủi ro) | Gợi ý |
|---|---|---|
| Abstract cuối | "The result is a fully-verified, deployable ECG IP core suited to single-lead wearable monitoring." | OK nhưng "fully-verified" hơi tuyệt đối → "bit-exact verified" (chính xác hơn) |
| §1 mở đầu | "can be life-saving" | Giữ nhưng nên kèm 1 cite dịch tễ, kẻo thành câu khẩu hiệu |
| §8 kết | "The combination — verified correctness, multiplier-free rescaling, and a compact deterministic core — makes this a practical ECG IP core" | Cấu trúc em-dash liệt kê 3 vế là pattern "giọng Ai" điển hình → tách thành câu thường |
| §4.3 | "is the key area lever" | "lever" hơi báo chí → "the principal area reduction" |
| Nhiều chỗ | em-dash `—` mật độ cao | AI-detector nhạy với em-dash; giảm bớt, thay bằng dấu phẩy/câu riêng ở vài chỗ |

**Nguyên tắc sửa:** giữ mọi câu **có số/cite**; chỉ làm phẳng câu **không có nội dung kiểm chứng**.
Sau khi sửa, các câu này vẫn nói đúng nội dung nhưng bớt "kêu".

### B.4 Phân định novelty vs Liu 2023 — RÀ XONG, defendable

> Cổng quan trọng nhất: tránh bị quy "trùng ý tưởng" với direct competitor.

| Yếu tố | Liu 2023 đã có | Của ta (novelty) | Draft xử lý đúng? |
|---|---|---|---|
| Power-of-2 shift rescale | ✅ có | KHÔNG claim mới | ✅ §1 C1, §2 nói rõ "we do not claim … novel" |
| Floor truncation | ✅ floor | **Round-half-up (+0.38%)** | ✅ §3.3 |
| Ablation p2 vs general vs floor | ❌ không | ✅ Table 4 + 5-fold | ✅ §6.1 |
| Bit-exact RTL verify | ❌ không | ✅ 21 checkpoint, max\|diff\|=0 | ✅ §5, C2 |
| Kiến trúc | fully-mapped (spatial) | **folded streaming** (time-mux 8 PE) | ✅ §2, §4.3 |
| Cross-dataset on-FPGA | ❌ | ✅ Chapman↔PTB-XL, weight reload | ✅ §6.4 (pitch là "short supporting result") |

**Kết luận novelty:** không có claim nào lấn sang phần Liu đã làm mà không thừa nhận. Câu phòng
thủ "we do not claim power-of-two rescaling as novel" (§1, §2) là lá chắn chính — **giữ nguyên**.
Khớp với memory [c1-quant-novelty-vs-liu] và [ptbxl-rtl-verify-scope].

### B.5 Tổng kết Phase K

| Cổng | Kết quả self-audit | Còn lại (bạn làm) |
|---|---|---|
| Đạo văn văn bản | THẤP — không đoạn dán nguyên | Chạy Turnitin xác nhận < 20% |
| Tính chính trực số | 5 số 🔴/🟠 cần khóa (B.2) | Xử lý theo "Open items" draft |
| Giọng AI | 5 chỗ nên làm phẳng (B.3) | Sửa tay; (nếu cần) chạy GPTZero bản cuối |
| Trùng ý tưởng | Defendable — phân định rõ vs Liu (B.4) | Search Scholar xác nhận không ai claim trùng |

**Không phát sinh blocker mới ngoài 6 "Open items" đã ghi cuối `ICDV_draft.md`.** Phase K chủ
yếu xác nhận: bản thảo sạch về đạo văn/ý tưởng; rủi ro thật nằm ở **khóa số liệu** (đã track) và
**làm phẳng vài câu giọng AI** (B.3).
