// tb_layer_cycles.v — do SO CHU KY thuc te cua tung trang thai FSM.
//
// Muc dich: cung cap so lieu DO DUOC cho bang "Phan tich so chu ky theo tung
// lop" trong bao cao, thay vi cong tay. Dem so chu ky clock ma layer_state
// giu tung gia tri trong DUNG mot luot suy luan.
//
// Giao thuc Avalon lay dung theo tb_top.v (DATA_IN/ADDR_IN/WR_EN + START).
// Chay: vsim -c -do run_tb_layer_cycles.do
`timescale 1ns / 1ps

module tb_layer_cycles;

    localparam CLK_P = 10;

    reg         clk = 0;
    reg         rst = 1;
    reg         rst_n = 0;
    reg  [13:0] avs_address = 0;
    reg         avs_write = 0;
    reg  [31:0] avs_writedata = 0;
    reg         avs_read = 0;
    wire [31:0] avs_readdata;

    always #(CLK_P/2) clk = ~clk;

    ecg_accelerator_top dut (
        .clk(clk), .rst(rst), .rst_n(rst_n),
        .avs_address(avs_address), .avs_write(avs_write),
        .avs_read(avs_read), .avs_writedata(avs_writedata),
        .avs_readdata(avs_readdata)
    );

    // ---- trang thai / co doc qua tham chieu phan cap (giong tb_top.v) ----
    wire [2:0] st     = dut.u_core.ctrl_layer_state;
    wire       done   = dut.u_core.done;
    wire [1:0] result = dut.u_core.result;

    // ---- ma trang thai (khop cnn_controller.v) ----
    localparam IDLE = 0, LOAD_INPUT = 1, CONV1 = 2, CONV2 = 3,
               CONV3 = 4, CONV4 = 5, GAP_FC_S = 6, DONE_S = 7;

    integer cnt [0:7];
    integer i, total;
    time    t_start, t_done;
    reg     counting = 0;

    always @(posedge clk) begin
        if (counting) begin
            cnt[st] <= cnt[st] + 1;
            total   <= total + 1;
        end
    end

    task avs_wr;
        input [4:0]  addr;
        input [31:0] data;
        begin
            @(negedge clk);
            avs_address   = addr;
            avs_writedata = data;
            avs_write     = 1;
            @(posedge clk); #1;
            avs_write = 0;
        end
    endtask

    task load_ecg_hex;
        input [255:0] filename;
        reg [7:0] ecg [0:2499];
        integer k;
        begin
            $readmemh(filename, ecg);
            for (k = 0; k < 2500; k = k + 1) begin
                avs_wr(5'h00, {24'h0, ecg[k]});   // DATA_IN
                avs_wr(5'h01, k[31:0]);           // ADDR_IN
                avs_wr(5'h02, 32'd1);             // WR_EN
            end
            @(posedge clk); #1;
        end
    endtask

    initial begin
        for (i = 0; i < 8; i = i + 1) cnt[i] = 0;
        total = 0;

        repeat (5) @(posedge clk);
        rst   = 0;
        rst_n = 1;
        repeat (2) @(posedge clk);

        load_ecg_hex("ecg_sample0.hex");

        avs_wr(5'h03, 32'd1);            // START
        @(posedge clk); #1;
        t_start = $time;

        // dem tu khi FSM roi IDLE cho toi khi done
        wait (st != IDLE);
        counting = 1;

        wait (done == 1'b1);
        t_done = $time;
        @(posedge clk);
        counting = 0;
        @(posedge clk);

        $display("");
        $display("=== SO CHU KY THEO TUNG TRANG THAI (do thuc te) ===");
        $display("  LOAD_INPUT : %0d", cnt[LOAD_INPUT]);
        $display("  CONV1      : %0d", cnt[CONV1]);
        $display("  CONV2      : %0d", cnt[CONV2]);
        $display("  CONV3      : %0d", cnt[CONV3]);
        $display("  CONV4      : %0d", cnt[CONV4]);
        $display("  GAP_FC_S   : %0d", cnt[GAP_FC_S]);
        $display("  DONE_S     : %0d", cnt[DONE_S]);
        $display("  IDLE       : %0d", cnt[IDLE]);
        $display("  ---------------------------");
        $display("  TONG       : %0d", total);
        $display("  result     : %0d", result);
        $display("  --- cua so do khac nhau ---");
        $display("  FSM state-cycles      : %0d", total);
        $display("  START -> done ($time) : %0d", (t_done - t_start)/10);
        $display("");
        $finish;
    end

    initial begin
        #50_000_000;
        $display("TIMEOUT");
        $finish;
    end

endmodule
