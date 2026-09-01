# Hai bài SOTA bổ sung (Nhóm B) — link để kiểm tra

Hai bài dưới đây được chọn để bổ sung vào Bảng 4.15. Mọi số đều lấy từ phần
tóm tắt/kết quả do chính bài báo công bố; **cần mở bản gốc xác nhận lại trước
khi nộp** (một số bài chỉ đọc được abstract nếu không có quyền truy cập).

---

## [3] Liu, Z.; Ling, X.; Zhu, Y.; Wang, N. (2025)

**FPGA-based 1D-CNN accelerator for real-time arrhythmia classification**
*Journal of Real-Time Image Processing*, vol. 22, art. 66, 2025.

- DOI: https://doi.org/10.1007/s11554-025-01642-w
- Springer: https://link.springer.com/article/10.1007/s11554-025-01642-w
- ACM DL: https://dl.acm.org/doi/abs/10.1007/s11554-025-01642-w

| Chỉ tiêu | Giá trị công bố |
|---|---|
| Chip | Xilinx Zynq 7Z020 (thiết kế PS–PL) |
| Mô hình | LW-CNN: tích chập tách theo chiều sâu (DSC) + kết nối tắt |
| Nén mô hình | Cắt tỉa phi cấu trúc + lượng tử hóa tăng dần (INQ) |
| Công cụ | Tổng hợp mức cao (HLS) |
| Độ chính xác phần mềm | 99,59% |
| Độ chính xác trên FPGA | **96,55%** |
| Tần số | 50 MHz |
| Độ trễ | **63 ms** |
| Công suất | **1,78 W** |
| Tập dữ liệu | MIT-BIH (phân loại theo nhịp đập) |

**Vì sao chọn:** bài FPGA–ECG mới nhất tìm được (2025), và là ví dụ đối chứng
cho hướng thiết kế ngược với luận văn: dùng HLS + mô hình lớn hơn nhiều, đổi
lại độ trễ 63 ms và công suất 1,78 W (gấp ~1.200× độ trễ và ~2,2× công suất
của luận văn). Cho thấy chi phí của việc không tối ưu thủ công ở mức RTL.

⚠️ **Lưu ý khi so sánh:** bài này phân loại **theo nhịp đập (beat)** trên
MIT-BIH, không phải theo bản ghi 10 s như luận văn → **không so trực tiếp cột
độ chính xác**. Chỉ so cột độ trễ/công suất/tài nguyên.

---

## [4] Lu, J.; Liu, D.; Cheng, X.; Wei, L.; Hu, A.; Zou, X. (2022)

**An Efficient Unstructured Sparse Convolutional Neural Network Accelerator for
Wearable ECG Classification Device**
*IEEE Transactions on Circuits and Systems I: Regular Papers*, vol. 69, no. 11,
pp. 4572–4582, 2022.

- IEEE Xplore: https://ieeexplore.ieee.org/document/9857602/
- DOI: https://doi.org/10.1109/TCSI.2022.3194155

| Chỉ tiêu | Giá trị công bố |
|---|---|
| Chip | Xilinx Zynq-7000 ZC706 |
| Kiến trúc | Bộ tăng tốc thưa phi cấu trúc, luồng dữ liệu tile-first |
| Cắt tỉa | Thưa 70% |
| Độ chính xác | 98,99% (PTB-XL); mất ~0,1% khi lên phần cứng |
| Tần số | 200 MHz |
| **Năng lượng** | **3,93 µJ / lần phân loại** |
| Hiệu suất tính toán | 118,75% |

**Vì sao chọn:** đây là mốc **năng lượng tốt nhất** trong nhóm bộ tăng tốc ECG
đeo được mà tôi tìm thấy, nên là phép thử khắt khe nhất cho luận điểm tiết kiệm
năng lượng của luận văn (42,02 µJ). Đối chiếu trung thực: luận văn **không**
thắng ở chỉ tiêu này.

⚠️ **Lưu ý:** con số 3,93 µJ đo ở 200 MHz trên chip Xilinx 28 nm, còn 42,02 µJ
của luận văn ước lượng bằng PowerPlay trên Cyclone V với độ tin cậy thấp; hai
số **không cùng phương pháp đo**. Ngoài ra bài dùng PTB-XL 5 lớp theo nhịp đập.

---

## Nguồn tra cứu đối chiếu (không đưa vào bảng)

- Tổng quan hệ thống 2025 (đối chiếu chéo số của hai bài trên):
  https://arxiv.org/html/2503.07276v1
- Tổng quan FPGA–ECG, MDPI Electronics 15(2):301, 2026:
  https://doi.org/10.3390/electronics15020301
