# REFERENCES — ICDV (Phase E01)

> Định dạng theo quy định trường: Tác giả, "Tiêu đề", *Nguồn (in nghiêng)*, Vol/No/pp, năm; kèm
> **ISBN (sách) / ISSN (tạp chí) / DOI**. **Hạn chế link Internet** trong references. Tối thiểu 15 mục.
>
> **Trạng thái verify:** ✅ = DOI/nguồn verify 2026-06-18 từ paper gốc (search). 🔲 = cần bổ sung
> trang/volume khi chốt camera-ready. Mỗi mục map tới entry Bảng A/B trong [SOTA_TABLE.md](../SOTA_TABLE.md).

Ngày dựng: 2026-06-18. **19 mục** (≥ 15 yêu cầu) — gồm 3 dataset, 3 methodology, 8 ECG-FPGA, 3 software-Chapman, 2 sách.

---

## A. Dataset & benchmark nền (3)

[1] J. Zheng, J. Zhang, S. Danioko, H. Yao, H. Guo, and C. Rakovski, "A 12-lead electrocardiogram
database for arrhythmia research covering more than 10,000 patients," *Scientific Data*, vol. 7,
art. 48, 2020. DOI: 10.1038/s41597-020-0386-x. ✅ *(Chapman dataset — bảng A5)*

[2] P. Wagner, N. Strodthoff, R.-D. Bousseljot, D. Kreiseler, F. I. Lunze, W. Samek, and T.
Schaeffter, "PTB-XL, a large publicly available electrocardiography dataset," *Scientific Data*,
vol. 7, art. 154, 2020. DOI: 10.1038/s41597-020-0495-6. ✅ *(cross-dataset PTB-XL)*

[3] G. B. Moody and R. G. Mark, "The impact of the MIT-BIH arrhythmia database," *IEEE Engineering
in Medicine and Biology Magazine*, vol. 20, no. 3, pp. 45–50, 2001. DOI: 10.1109/51.932724. ✅
*(MIT-BIH — dataset của nhiều competitor bảng B)*

## B. Quantization & model compression methodology (3)

[4] B. Jacob, S. Kligys, B. Chen, M. Zhu, M. Tang, A. Howard, H. Adam, and D. Kalenichenko,
"Quantization and training of neural networks for efficient integer-arithmetic-only inference,"
in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, pp. 2704–2713, 2018.
DOI: 10.1109/CVPR.2018.00286. ✅ *(general-scale INT8 — đối chứng C1)*

[5] S. Han, J. Pool, J. Tran, and W. J. Dally, "Learning both weights and connections for efficient
neural networks," in *Proc. Advances in Neural Information Processing Systems (NeurIPS)*, vol. 28,
pp. 1135–1143, 2015. DOI: 10.5555/2969239.2969366. ✅ *(pruning nền — bảng A pruned model)*

[6] D. D. Lin, S. S. Talathi, and V. S. Annapureddy, "Fixed point quantization of deep convolutional
networks," in *Proc. 33rd Int. Conf. Machine Learning (ICML)*, vol. 48, pp. 2849–2858, 2016.
DOI: 10.5555/3045390.3045690. ✅ *(fixed-point/shift quantization nền — đối chứng C1)*

## C. ECG-FPGA accelerators — direct & landscape (8)

[7] Y. Liu, *et al.*, "A fully-mapped and energy-efficient FPGA accelerator for dual-function
AI-based analysis of ECG," *Frontiers in Physiology*, vol. 14, art. 1079503, 2023.
DOI: 10.3389/fphys.2023.1079503. ✅ *(direct competitor — bảng B2; Chapman, Cyclone V)*

[8] M. Wess, S. M. P. Dinakarrao, and A. Jantsch, "Neural network based ECG anomaly detection on
FPGA and trade-off analysis," in *Proc. IEEE Int. Symp. Circuits and Systems (ISCAS)*, 2017.
DOI: 10.1109/ISCAS.2017.8050805. ✅ *(bảng B10 — MLP+PCA, MIT-BIH)*

[9] M. Carreras, G. Deriu, L. Raffo, L. Benini, and P. Meloni, "Optimizing temporal convolutional
network inference on FPGA-based accelerators," *IEEE Journal on Emerging and Selected Topics in
Circuits and Systems*, vol. 10, no. 3, pp. 348–361, 2020. DOI: 10.1109/JETCAS.2020.3014503. ✅
*(bảng B7 — TCN trên FPGA)*

[10] R. Srivastava, B. Kumar, F. Alenezi, A. Alhudhaif, S. A. Althubiti, and K. Polat, "Automatic
arrhythmia detection based on the probabilistic neural network with FPGA implementation,"
*Mathematical Problems in Engineering*, vol. 2022, art. 7564036, 2022. DOI: 10.1155/2022/7564036.
✅ *(bảng B8 — PNN, Artix-7, MIT-BIH 8-class)*

[11] T. M. Ingolfsson, X. Wang, M. Hersche, A. Burrello, L. Cavigelli, and L. Benini, "ECG-TCN:
Wearable cardiac arrhythmia detection with a temporal convolutional network," in *Proc. IEEE Int.
Conf. Artificial Intelligence Circuits and Systems (AICAS)*, 2021. DOI: 10.1109/AICAS51828.2021.9458520.
✅ *(TCN wearable, ECG5000 94.2% — phân biệt với [9])*

[12] X. Cheng, L. Wei, C. Zhang, *et al.*, "Efficient hardware architecture of convolutional neural
network for ECG classification in wearable healthcare device," *IEEE Transactions on Circuits and
Systems I: Regular Papers*, vol. 68, no. 7, pp. 2976–2985, 2021. DOI: 10.1109/TCSI.2021.3072622. ✅
*(bảng B6 — 1-D CNN Zynq-7045, Chapman; first author = Cheng, "Wei et al." theo cách Liu cite)*

[13] S. Ran, X. Yang, M. Liu, Y. Zhang, C. Cheng, H. Zhu, *et al.*, "Homecare-oriented ECG diagnosis
with large-scale deep neural network for continuous monitoring on embedded devices," *IEEE
Transactions on Instrumentation and Measurement*, vol. 71, pp. 1–13, 2022. DOI: 10.1109/TIM.2022.3147328.
✅ *(bảng B9 — CNN Zynq-7020 + ARM, MIT-BIH)*

[14] V. Rawal, P. Prajapati, and A. Darji, "Hardware implementation of 1D-CNN architecture for ECG
arrhythmia classification," *Biomedical Signal Processing and Control*, vol. 85, art. 104865, 2023.
DOI: 10.1016/j.bspc.2023.104865. ✅ *(bảng B5 — cùng paper báo nhiều dataset: CinC-2017 PRCA 90.80% SW
/ 86.37% HW / AF 97.34% / 628 mW ZYNQ UltraScale, và MIT-BIH 98.6% + PTB 99.67%. Nghi-vấn-2-paper đã
giải: chỉ 1 paper Rawal, đa-dataset.)*

## D. Software models trên Chapman — so accuracy/params (3)

[15] K.-H. Le, H.-H. Pham, T.-B. Nguyen, *et al.*, "LightX3ECG: A lightweight and explainable deep
learning system for 3-lead electrocardiogram classification," *Biomedical Signal Processing and
Control*, vol. 85, art. 104963, 2023. DOI: 10.1016/j.bspc.2023.104963. ✅ 🔲 *(bản published thay
arXiv:2207.12381; 5.31M params, 4-superclass — bảng A2; xác nhận art-no/DOI từ PII S1746809423003968)*

[16] T. Yoon and D. Kang, "Bimodal CNN for cardiovascular disease classification by co-training ECG
grayscale images and scalograms," *Scientific Reports*, vol. 13, art. 2937, 2023.
DOI: 10.1038/s41598-023-30208-8. ✅ *(bảng A3 — Chapman 4-superclass, 95.08% Lead-II / F1 0.944)*

[17] 🔲 CardioPatternFormer (Transformer ECG), *(arXiv:2505.20481, 2025; bảng A4)*. Ưu tiên bản
published nếu có; nếu không, cân nhắc thay bằng 1 paper Transformer-ECG đã xuất bản chính thức.

## E. Sách / textbook nền (2) — mục ISBN

[18] L. Đ. Hùng và C. T. Bảo Thương, *Thiết kế logic mạch số với Verilog HDL*. NXB ĐHQG TP. HCM,
2020 (Tái bản lần 1). ISBN: 🔲 *(điền từ trang bản quyền sách — nền RTL/Verilog của thiết kế; sách
của chính nhóm tác giả, đúng kiểu mẫu slide trường)*

[19] S. Palnitkar, *Verilog HDL: A Guide to Digital Design and Synthesis*, 2nd ed. Prentice Hall PTR,
2003. ISBN: 0-13-044911-3. ✅ *(textbook Verilog kinh điển — nền phương pháp RTL)*

---

## Việc còn lại (Phase E01)

**Tiến độ định danh: 16/19 có DOI/ISBN ✅** (15 DOI + 1 ISBN sách Palnitkar). Cập nhật 2026-06-18.
Còn lại:
- [ ] 🔴 [17] CardioPatternFormer — tìm bản published có DOI, hoặc thay bằng Transformer-ECG đã xuất bản.
- [x] ✅ [14] GIẢI nghi vấn: B5 và [14] là CÙNG 1 paper Rawal BSPC 2023 (đa-dataset: CinC + MIT-BIH + PTB).
      DOI 10.1016/j.bspc.2023.104865 đúng cho cả hai. Số CinC 86.37% HW / 628 mW khớp bảng B5.
- [ ] 🟠 [15] xác nhận art-no/DOI chính xác LightX3ECG BSPC (suy từ PII, chưa verify số art 104963).
- [ ] 🟠 [18] điền ISBN sách Lê Đức Hùng từ trang bản quyền (không tra được online — cần sách giấy).
- [ ] 🟢 Convert sang BibTeX (`.bib`); map citation key vào ICDV_draft.md ([CITE ...] hiện là placeholder).

**Đã verify từ Liu Article.xml (offline, paper gốc) — citation đầy đủ:**
[12] Cheng/Wei TCAS-I 2021 · [13] Ran TIM 2022 · [9] Carreras JETCAS 2020 · [10] Srivastava MPE 2022 ·
[8] Wess ISCAS 2017 — tất cả khớp DOI.

> **Quy định trường**: hạn chế link Internet → [11] (AICAS) đã có DOI; [15][16] đã chuyển sang bản
> published (BSPC/Sci.Rep) thay arXiv; chỉ còn [17] là arXiv. Sách ISBN / tạp chí ISSN / hội nghị
> ISBN-proceedings — DOI là định danh đủ cho IEEE-style; thêm ISBN sách nếu muốn đủ cả 3 loại.
