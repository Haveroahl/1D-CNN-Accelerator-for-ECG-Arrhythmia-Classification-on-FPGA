# Demo trên board DE10-Standard — bản RTL ROM (Chương 4)

Bộ file này chạy toàn bộ tập kiểm tra Chapman + Georgia trên board thật và ghi ra file log
để chụp hình đưa vào khóa luận.

## Đã chuẩn bị sẵn

| File | Vai trò |
|---|---|
| `../output_files/jtag_top.sof` | Bitstream, trọng số Chapman bake sẵn trong ROM (compile 2026-07-31) |
| `ecg_jtag_rom.tcl` | Driver System Console cho **bản ROM** (không nạp trọng số/topology) |
| `demo_data/ningba_test_*` (nhãn: **Chapman**) | 4973 mẫu in-distribution — kỳ vọng **94,27 % (4688/4973)** |
| `demo_data/georgia_test_*` | 5459 mẫu kiểm tra chéo zero-shot — kỳ vọng **93,00 % (5077/5459)** |

Một lần chạy làm **cả hai tập, 10.432 mẫu**, trên **cùng một bitstream**: Georgia
là zero-shot (dùng đúng trọng số Chapman), không cần nạp lại trọng số — đó chính
là lý do bản ROM chạy được bài kiểm tra chéo.

> **Dùng đúng script.** `ecg_jtag_console.tcl` là driver của bản *weight-load*
> (`hardware/RTL_weight/`): nó ghi vào CONFIG window + weight window và đọc bit
> `isram_free`. Bản ROM **không có** những thứ đó → script kia sẽ lỗi. Bản ROM
> dùng `ecg_jtag_rom.tcl`.

## Các bước chạy

**1. Nạp bitstream** (USB-Blaster đã cắm, board đã bật):

```
cd d:\Thesis101\hardware\fpga
D:\altera_lite\25.1std\quartus\bin64\quartus_pgm.exe -m jtag -o "p;output_files/jtag_top.sof"
```

Kiểm tra board có nhận diện được không: `D:\altera_lite\25.1std\quartus\bin64\jtagconfig.exe`
(phải liệt kê ra `5CSEBA6/5CSEMA6/...`; nếu báo "No JTAG hardware available" thì
chưa cắm USB-Blaster hoặc chưa bật nguồn board).

**2. Chạy demo:**

```
cd d:\Thesis101\hardware\fpga\soc
D:\altera_lite\25.1std\quartus\sopc_builder\bin\system-console.exe --script=ecg_jtag_rom.tcl
```

**3. Kết quả** → file `ecg_rom_<ngày>_<giờ>.log` ngay trong thư mục `soc/`.

Chạy thử nhanh trước — gõ **trước** khi `source` (script tôn trọng giá trị đặt sẵn):

```tcl
set ::MAX_SAMPLES 20
source D:/Thesis101/hardware/fpga/soc/ecg_jtag_rom.tcl
```

Xong thì mở lại System Console (hoặc `set ::MAX_SAMPLES 0`) để chạy toàn tập.

## Kỳ vọng

Cuối log có khối tổng kết gộp (phần này để chụp hình):

```
==============================================
  SUMMARY — DE10-Standard (Cyclone V), 1 bitstream
==============================================
  dataset   on-board       expected(SW)   match
  Chapman   4688/4973 94.27%  4688/4973 94.27%  yes
  Georgia   5077/5459 93.00%  5077/5459 93.00%  yes
==============================================
```

Trước đó mỗi tập có khối riêng với recall từng lớp (AFIB/GSVT/SB/SR).

Con số phải **khớp đúng**, không phải xấp xỉ: RTL đã chứng minh khớp-bit với
Python, nên mỗi mẫu phải cho cùng một lớp. Cột `match` in `yes` khi đúng. Nếu ra
`NO` → kênh JTAG rớt hoặc nạp nhầm bitstream, **không phải** sai số mô hình.

> **94,73 % là số nào?** Đó là mô phỏng Python dùng GAP số thực. Phần cứng dùng
> GAP số nguyên `floor(sum/4)`, cho **94,27 %**. File meta ghi số 94,27 % vì đó
> mới là số board phải cho ra.

## Thời gian chạy

Kênh JTAG là nút cổ chai, không phải lõi (lõi chỉ mất 52,16 µs/mẫu). Nhưng cả
mẫu 2500 byte được gửi bằng **một** lệnh `master_write_32` dạng khối, không phải
2500 giao dịch rời — đây là tối ưu đã có sẵn trong DATA WINDOW của
`avalon_slave.v`. Nhờ vậy 10.432 mẫu là khả thi trong một lần chạy.

Vẫn nên chạy lúc không cần dùng máy: **đừng đụng cáp USB, tắt chế độ ngủ**. Kênh
JTAG rớt giữa chừng là mất cả lần chạy. Log ghi từng mẫu và `flush` ngay, nên
nếu rớt vẫn còn phần đã chạy.

## Đã kiểm chứng trước khi ra board

`hardware/testbench/tb_demo_bus.v` phát lại **đúng chuỗi giao dịch** mà driver
này gửi (block-write DATA WINDOW → START → poll bit done → đọc RESULT) trên 40
mẫu Chapman trải đều 4 lớp:

```
RTL vs Python (bit-exact) : 40/40 agree
RTL vs ground truth       : 37/40 correct (92.5%)
Avg cycles/inference      : 5219
RESULT: PASS
```

Chạy lại: `cd hardware/fpga/simulation/questa && vsim -c -do run_tb_demo_bus.do`.
Nghĩa là giao thức bus của driver đã đúng — nếu trên board sai thì vấn đề nằm ở
kết nối/bitstream, không phải ở logic driver.
