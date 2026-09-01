// ============================================================================
// tb_power_vcd.v — sinh VCD activity file cho Quartus PowerPlay (bản ROM).
//
// TẠI SAO PHẢI CÓ TESTBENCH RIÊNG (không dùng lại tb_top.v):
// PowerPlay tính dynamic power từ TOGGLE RATE của từng node. Nếu VCD chứa cả
// giai đoạn nạp 2500 byte qua Avalon — lúc đó lõi tính toán ĐỨNG YÊN — thì
// toggle rate trung bình bị pha loãng và power bị BÁO THẤP HƠN thực tế. Đây
// không phải sai số ngẫu nhiên mà là sai lệch có hệ thống, theo hướng có lợi
// cho ta → không dùng được để bảo vệ.
//
// Testbench này nạp dữ liệu với dump TẮT, chỉ bật $dumpon đúng cửa sổ suy luận
// (START → done, 5216 chu kỳ), rồi tắt lại. VCD thu được mô tả đúng trạng thái
// "accelerator đang làm việc" — là chế độ cần cho số energy/inference.
//
// Dump toàn hierarchy DUT ($dumpvars(0, ...)) để PowerPlay có dữ liệu cho mọi
// node, thay vì phải suy đoán → nâng Power Estimation Confidence.
//
// CHẠY:  vsim -c -do run_tb_power_vcd.do
// XUẤT:  ecg_power.vcd  (nạp vào PowerPlay qua .qsf)
// ============================================================================
`timescale 1ns / 1ps

module tb_power_vcd;

    localparam SAMPLE_LEN = 2500;

    reg         clk = 0;
    reg         rst = 1;
    reg         rst_n = 0;
    reg  [13:0] avs_address = 0;
    reg         avs_write = 0;
    reg         avs_read = 0;
    reg  [31:0] avs_writedata = 0;
    wire [31:0] avs_readdata;

    always #5 clk = ~clk;   // 100 MHz — khớp SDC

    ecg_accelerator_top dut (
        .clk(clk), .rst(rst), .rst_n(rst_n),
        .avs_address(avs_address), .avs_write(avs_write), .avs_read(avs_read),
        .avs_writedata(avs_writedata), .avs_readdata(avs_readdata)
    );

    reg [7:0] ecg [0:SAMPLE_LEN-1];
    integer i, cyc0, cyc_infer;
    reg [31:0] status;

    task bus_write(input [13:0] addr, input [31:0] data);
        begin
            @(negedge clk);
            avs_address = addr; avs_writedata = data; avs_write = 1;
            @(negedge clk);
            avs_write = 0;
        end
    endtask

    task bus_read(input [13:0] addr, output [31:0] data);
        begin
            @(negedge clk);
            avs_address = addr; avs_read = 1;
            @(negedge clk);
            avs_read = 0;
            @(negedge clk);
            data = avs_readdata;
        end
    endtask

    initial begin
        $readmemh("ecg_sample0.hex", ecg);

        $display("========================================================");
        $display(" tb_power_vcd — sinh VCD cho PowerPlay (ban ROM)");
        $display("========================================================");

        // Khai bao dump TOAN BO hierarchy DUT, nhung TAT ngay: giai doan nap
        // du lieu khong duoc lot vao VCD (xem header).
        $dumpfile("ecg_power.vcd");
        $dumpvars(0, tb_power_vcd.dut);
        $dumpoff;

        rst = 1; rst_n = 0;
        repeat (5) @(posedge clk);
        rst = 0; rst_n = 1;
        repeat (5) @(posedge clk);

        // ---- Nap 1 mau qua DATA WINDOW (dump dang TAT) ----
        for (i = 0; i < SAMPLE_LEN; i = i + 1)
            bus_write(14'h1000 + i[13:0], {24'b0, ecg[i]});
        @(posedge clk); #1;
        $display(" Da nap 2500 mau ECG (dump TAT)");

        // ---- BAT dump: tu day la cua so suy luan ----
        $dumpon;
        cyc0 = $time / 10;

        bus_write(14'h0003, 32'h1);            // START

        status = 0;
        while ((status & 32'h2) == 0)          // doi done_latched (bit1)
            bus_read(14'h0004, status);

        cyc_infer = ($time / 10) - cyc0;
        bus_read(14'h0005, status);

        $dumpoff;
        // ---- TAT dump: het cua so suy luan ----

        $display(" Suy luan xong: lop = %0d", status[1:0]);
        $display(" So chu ky ghi vao VCD : %0d", cyc_infer);
        $display(" Tuong duong           : %0.2f us @ 100 MHz", cyc_infer / 100.0);
        $display("--------------------------------------------------------");
        $display(" VCD: ecg_power.vcd  -> nap vao PowerPlay");
        $display("========================================================");
        $finish;
    end

endmodule
