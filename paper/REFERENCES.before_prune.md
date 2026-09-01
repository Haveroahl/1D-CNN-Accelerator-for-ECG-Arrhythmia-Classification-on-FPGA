# REFERENCES — ICDV (Phase E01)

> Định dạng theo quy định trường: Tác giả, "Tiêu đề", *Nguồn (in nghiêng)*, Vol/No/pp, năm; kèm
> **ISBN (sách) / ISSN (tạp chí) / DOI**. **Hạn chế link Internet** trong references. Tối thiểu 15 mục.
>
> **Trạng thái verify:** ✅ = DOI/nguồn verify 2026-06-18 từ paper gốc (search). 🔲 = cần bổ sung
> trang/volume khi chốt camera-ready. Mỗi mục map tới entry Bảng A/B trong [SOTA_TABLE.md](../SOTA_TABLE.md).

Ngày dựng: 2026-06-18 (bổ sung nhóm F+G 2026-07-17, rà toàn bộ software 2026-07-17; bổ sung nhóm H+I
phần cứng 2026-07-19). **42 mục** (≥ 15 yêu cầu) — gồm 3 dataset, 3 methodology, 8 ECG-FPGA,
3 software-Chapman, 2 sách, **8 training-pipeline (F: [20]–[24],[30]–[33]), 6 CNN-model
(G: [25]–[29],[34]), 5 hardware-technique (H: [35]–[39]), 3 toolflow/engine (I: [40]–[42])**.
Nhóm F+G dẫn chứng đúng từng kỹ thuật CÓ TRONG CODE: Adam/CE-loss, STE-QAT, power-of-2 (nguồn Miyashita
[30]/Nagel [31]), Taylor+L1 pruning ([32][33]), GAP ([34]); và các đối chứng bỏ BN/Dropout ([21][22]).
Nhóm H dẫn chứng kỹ thuật RTL CÓ TRONG `hardware/RTL/`: dataflow taxonomy/DNN-accelerator ([35]),
line-buffer + double-buffering conv streaming trên FPGA ([36]), systolic/MAC-array (đối chứng, [37]),
tính toán số học có giới hạn/số cố định + số học bão hoà ([38]), Cyclone V ALM/DSP18/M10K ([39]).
Nhóm I dẫn chứng LÝ DO chọn kiến trúc (không phải cơ chế): họ streaming/single-engine FINN ([40]),
fpgaConvNet + DSE ([41]), survey phân loại streaming-vs-single-engine ([42]) — đối chứng với
fully-mapped Liu 2023 ([7]).

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

[43] E. A. Perez Alday, A. Gu, A. J. Shah, C. Robichaux, A.-K. I. Wong, C. Liu, F. Liu, A. B. Rad,
A. Elola, S. Seyedi, Q. Li, A. Sharma, G. D. Clifford, and M. A. Reyna, "Classification of 12-lead
ECGs: the PhysioNet/Computing in Cardiology Challenge 2020," *Physiological Measurement*, vol. 41,
no. 12, art. 124003, 2020. DOI: 10.1088/1361-6579/abc960. ✅ *(Georgia 12-Lead ECG Challenge
Database — cross-dataset far-transfer, Mục 2.1.3)*

## B. Quantization & model compression methodology (3)

[4] B. Jacob, S. Kligys, B. Chen, M. Zhu, M. Tang, A. Howard, H. Adam, and D. Kalenichenko,
"Quantization and training of neural networks for efficient integer-arithmetic-only inference,"
in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, pp. 2704–2713, 2018.
DOI: 10.1109/CVPR.2018.00096. ✅ *(general-scale INT8 + QAT gốc — đối chứng C1; sửa DOI .00286→.00096 2026-07-17)*

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

## F. Training pipeline — optimizer, regularization, quantization (5)

> Dẫn chứng cho lựa chọn pipeline huấn luyện & lượng tử (Section 3 "CNN Model and Power-of-2 QAT
> Methodology"). Bổ sung cho [4][5][6] đã có (general-scale INT8 / pruning / fixed-point).

[20] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in *Proc. 3rd Int. Conf.
Learning Representations (ICLR)*, San Diego, CA, USA, 2015. arXiv:1412.6980. ✅ *(optimizer dùng
train float32 + QAT fine-tune — không có DOI hội nghị, dùng arXiv ID theo thông lệ)*

[21] S. Ioffe and C. Szegedy, "Batch normalization: Accelerating deep network training by reducing
internal covariate shift," in *Proc. 32nd Int. Conf. Machine Learning (ICML)*, vol. 37, pp. 448–456,
2015. DOI: 10.5555/3045118.3045167. ✅ *(⚠️ĐỐI CHỨNG — model KHÔNG dùng BatchNorm: fold BN vào power-of-2
scale làm phức tạp pipeline bit-exact; cite để biện luận lý do BỎ, không phải kỹ thuật đã áp dụng)*

[22] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, "Dropout: A simple
way to prevent neural networks from overfitting," *Journal of Machine Learning Research*, vol. 15,
no. 1, pp. 1929–1958, 2014. ISSN: 1532-4435. ✅ *(⚠️ĐỐI CHỨNG — model KHÔNG dùng Dropout: đã regularize
đủ bằng structured pruning (640 params) + GAP; cite để biện luận lý do BỎ, không phải kỹ thuật đã áp dụng)*

[23] B. Jacob, S. Kligys, B. Chen, *et al.*, "Quantization and training of neural networks for efficient
integer-arithmetic-only inference," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition
(CVPR)*, pp. 2704–2713, 2018. DOI: 10.1109/CVPR.2018.00096. ✅ *(**cùng nguồn [4]** — đây là QAT gốc
+ general-scale; giữ 1 entry [4], KHÔNG nhân đôi — ghi chú tại đây để tránh trùng)*

[24] Y. Bengio, N. Léonard, and A. Courville, "Estimating or propagating gradients through stochastic
neurons for conditional computation," *arXiv:1308.3432*, 2013. ✅ *(straight-through estimator — cơ chế
back-prop qua fake-quant, dùng thực trong `qat_int8.py` FakeQuantize.forward; chưa có bản published, dùng arXiv)*

[30] D. Miyashita, E. H. Lee, and B. Murmann, "Convolutional neural networks using logarithmic data
representation," *arXiv:1603.01025*, 2016. ✅ *(⭐ NGUỒN GỐC power-of-2/log quantization — nhân → bit-shift;
cite để KHÔNG claim power-of-2 là mới của mình HAY của Liu. Xem [[c1-quant-novelty-vs-liu]])*

[31] M. Nagel, M. Fournarakis, R. A. Amjad, Y. Bondarenko, M. van Baalen, and T. Blankevoort, "A white
paper on neural network quantization," *arXiv:2106.08295*, 2021. ✅ *(chính paper Liu 2023 cite; nêu rõ
power-of-2 scale `s=2^-k` = bit-shift — dẫn chứng power-of-2 là kỹ thuật established, không phải novelty)*

[32] P. Molchanov, S. Tyree, T. Karras, T. Aila, and J. Kautz, "Pruning convolutional neural networks
for resource efficient inference," in *Proc. 5th Int. Conf. Learning Representations (ICLR)*, 2017.
arXiv:1611.06440. ✅ *(⭐ phương pháp pruning CHÍNH dùng thực — Taylor first-order importance,
`prune_finetune.py::_taylor_rank`; bổ sung [5] Han là magnitude-only)*

[33] H. Li, A. Kadav, I. Durdanovic, H. Samet, and H. P. Graf, "Pruning filters for efficient ConvNets,"
in *Proc. 5th Int. Conf. Learning Representations (ICLR)*, 2017. arXiv:1608.08710. ✅ *(L1-norm
structured filter pruning — fallback trong `prune_finetune.py::_l1_rank`; nền cho structured channel
pruning giữ dense connectivity, phù hợp phần cứng)*

## G. CNN model & so sánh kiến trúc (5)

> Dẫn chứng cho lựa chọn kiến trúc 1D-CNN (vs 2D-CNN / RNN / Transformer). Bổ sung cho [12][14] (1D-CNN
> FPGA) và [15][16][17] (model Chapman) đã có.

[25] S. Kiranyaz, T. Ince, and M. Gabbouj, "Real-time patient-specific ECG classification by 1-D
convolutional neural networks," *IEEE Transactions on Biomedical Engineering*, vol. 63, no. 3,
pp. 664–675, 2016. DOI: 10.1109/TBME.2015.2468589. ✅ *(⭐ nền tảng 1D-CNN trực tiếp trên raw ECG —
lý do chọn 1D thay 2D: bỏ hand-crafted feature, real-time)*

[26] A. Y. Hannun, P. Rajpurkar, M. Haghpanahi, *et al.*, "Cardiologist-level arrhythmia detection and
classification in ambulatory electrocardiograms using a deep neural network," *Nature Medicine*,
vol. 25, no. 1, pp. 65–69, 2019. DOI: 10.1038/s41591-018-0268-3. ✅ *(deep 1D-CNN đạt mức bác sĩ —
minh chứng CNN đủ mạnh cho ECG, đối chứng độ sâu)*

[27] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc.
IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, pp. 770–778, 2016.
DOI: 10.1109/CVPR.2016.90. ✅ *(ResNet — đối chứng "sâu hơn ≠ tốt hơn" cho model nhỏ; lý do KHÔNG
chọn residual depth cho wearable)*

[28] Z. Ebrahimi, M. Loni, M. Daneshtalab, and A. Gharehbaghi, "A review on deep learning methods for
ECG arrhythmia classification," *Expert Systems with Applications: X*, vol. 7, art. 100033, 2020.
DOI: 10.1016/j.eswax.2020.100033. ✅ *(⭐ survey so sánh CNN/DBN/RNN/LSTM/GRU trên ECG — CNN chiếm 52%,
là dẫn chứng chính cho lý do chọn CNN thay RNN)*

[29] O. S. Lih, V. Jahmunah, T. R. San, *et al.*, "Comprehensive electrocardiographic diagnosis based
on deep learning," *Artificial Intelligence in Medicine*, vol. 103, art. 101789, 2020.
DOI: 10.1016/j.artmed.2019.101789. ✅ *(so sánh CNN vs CNN-LSTM hybrid trên ECG đa bệnh — đối chứng
hybrid; CNN-LSTM đạt 98.51% 4-class. First author Oh S.L. cite theo "Lih" theo tên giữa)*

[34] M. Lin, Q. Chen, and S. Yan, "Network in network," in *Proc. 2nd Int. Conf. Learning
Representations (ICLR)*, 2014. arXiv:1312.4400. ✅ *(⭐ nguồn Global Average Pooling — model dùng GAP thay
Flatten (`model.py::gap`), KHÁC Liu 2023 (Flatten). GAP: ít param FC hơn, ít overfit, thay regularization
cho Dropout. Dẫn chứng cho quyết định kiến trúc + lý do bỏ Dropout [22])*

## H. Kiến trúc phần cứng & kỹ thuật FPGA (5)

> Dẫn chứng cho các kỹ thuật RTL trong `hardware/RTL/` (Chương thiết kế phần cứng): phân loại luồng dữ
> liệu, tích chập bằng line-buffer/shift-register-window, bộ đệm ping-pong, mảng MAC/systolic (đối
> chứng), số học bão hoà + số cố định, và đặc tính khối FPGA Cyclone V. Bổ sung cho [18][19] (sách
> Verilog) và [6][30][31] (power-of-2/bit-shift đã có ở nhóm B/F).

[35] V. Sze, Y.-H. Chen, T.-J. Yang, and J. S. Emer, "Efficient processing of deep neural networks:
A tutorial and survey," *Proceedings of the IEEE*, vol. 105, no. 12, pp. 2295–2329, 2017.
DOI: 10.1109/JPROC.2017.2761740. ✅ *(⭐ phân loại dataflow weight/output/input-stationary + phân tích
năng lượng DNN accelerator — nền cho lựa chọn time-multiplexed 8-PE và cấu trúc MAC; xem
[[paper-venue-icdv-dse]])*

[36] C. Zhang, P. Li, G. Sun, Y. Guan, B. Xiao, and J. Cong, "Optimizing FPGA-based accelerator design
for deep convolutional neural networks," in *Proc. ACM/SIGDA Int. Symp. Field-Programmable Gate Arrays
(FPGA)*, pp. 161–170, 2015. DOI: 10.1145/2684746.2689060. ✅ *(⭐ nền line-buffer/on-chip buffering +
ping-pong double-buffering + loop tiling cho conv streaming trên FPGA — dẫn chứng SRW window
`cp_engine.v` và ping-pong `ping_pong_sram.v`)*

[37] H. T. Kung, "Why systolic architectures?," *Computer*, vol. 15, no. 1, pp. 37–46, 1982.
DOI: 10.1109/MC.1982.1653825. ✅ *(mảng systolic/MAC-array kinh điển — đối chứng: đề tài dùng
adder-tree MAC 5-tap + 8-PE channel-parallel thay mảng systolic 2D, vì tiny-1D-CNN kênh ≤8 lãng phí PE)*

[38] J. L. Hennessy and D. A. Patterson, *Computer Architecture: A Quantitative Approach*, 6th ed.
Morgan Kaufmann, 2019. ISBN: 978-0-12-811905-1. ✅ *(số học số nguyên bù-hai, số cố định, bão hoà/clamp
+ pipelining — nền cho rescale dịch-bit `>>>nb`, clamp[−127,127], pipeline 5 tầng CP-block)*

[39] Intel Corporation, *Cyclone V Device Handbook, Volume 1: Device Interfaces and Integration*,
Intel/Altera, 2020. *(khối ALM, DSP18 variable-precision, M10K embedded memory — cơ sở suy ra bộ nhân
DSP18, BRAM ping-pong/input, và ước lượng tài nguyên; thay $signed → DSP18 inference. URL/version điền
khi camera-ready)* 🔲

## I. Toolflow / họ accelerator engine dùng chung (3) — dẫn chứng LÝ DO chọn kiến trúc

> Dẫn chứng cho *quyết định lựa chọn* kiến trúc RTL (không phải cơ chế hoạt động — đó là nhóm H). Chứng
> minh time-multiplexed single-computation-engine là họ kiến trúc đã được thiết lập trong tài liệu
> (FINN/fpgaConvNet), và phân loại streaming-vs-single-engine ([42] survey) là khung lập luận đối chứng
> với fully-mapped của Liu 2023 [7]. Xem [[paper-venue-icdv-dse]].

[40] Y. Umuroglu, N. J. Fraser, G. Gambardella, M. Blott, P. Leong, M. Jahre, and K. Vissers, "FINN:
A framework for fast, scalable binarized neural network inference," in *Proc. ACM/SIGDA Int. Symp.
Field-Programmable Gate Arrays (FPGA)*, pp. 65–74, 2017. DOI: 10.1145/3020078.3021744. ✅ *(⭐ họ
streaming dataflow engine per-layer tailored — nền cho lựa chọn engine dùng chung/folded thay
fully-mapped; SRW streaming + per-layer nb/cp_en của đề tài cùng triết lý)*

[41] S. I. Venieris and C.-S. Bouganis, "fpgaConvNet: A framework for mapping convolutional neural
networks on FPGAs," in *Proc. IEEE 24th Annual Int. Symp. Field-Programmable Custom Computing Machines
(FCCM)*, pp. 40–47, 2016. DOI: 10.1109/FCCM.2016.22. ✅ *(⭐ mô hình hoá CNN như ứng dụng streaming +
khám phá không gian performance-resource — dẫn chứng cho DSE 2 dataflow (8-PE vs SIMD-20) của đề tài)*

[42] S. I. Venieris, A. Kouris, and C.-S. Bouganis, "Toolflows for mapping convolutional neural
networks on FPGAs: A survey and future directions," *ACM Computing Surveys*, vol. 51, no. 3, art. 56,
pp. 1–39, 2018. DOI: 10.1145/3186332. ✅ *(⭐ survey phân loại streaming architecture vs
single-computation-engine — khung lập luận chính cho "vì sao time-multiplexed"; xếp Liu 2023
fully-mapped [7] và đề tài single-engine vào 2 nhóm đối lập)*

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
