// tb_cp_block_simple.v — TB toi gian cho cp_block
// ---------------------------------------------------------------------------
// Muc dich: nap 5 tap x + 5 weight, tinh 1 pixel conv -> rescale, roi lam day
// mot pool window (5 pixel) de ra pool_out. KHONG task library, chi chuoi tuan
// tu + gia tri ky vong ghi thang, de doc trong Chuong 4 / xem tren GUI.
//
// cp_block KHONG chua SRW. x_in la 5 tap DA dong goi (do cp_engine cap trong
// thiet ke that). O day ta dong goi tay de co lap datapath conv->pool.
//
// IN_CH=1: moi cycle co compute_en_in=1 & a_in=0 la MOT pixel hop le.
// nb=0    : pool_out = tong conv truc tiep (de doi chieu tay).
// relu_en=0.
//
// Pipeline (theo cp_engine.v:22-28): x_in/w -> prod -> tree(3) -> acc -> bias
// -> rescale -> relu_out, roi cp_pool gom 5 pixel -> pool_write + pool_out.
//
// Run:  vsim -c -do "do run_tb_cp_block_simple.do; quit -f"
// GUI:  vsim -gui -do wave_tb_cp_block_simple.do   (roi: run -all)

`timescale 1ns/1ps

module tb_cp_block_simple;

    // ── DUT ports ───────────────────────────────────────────────────────
    reg        clk, rst;
    reg [39:0] x_in;      // packed 5×8b: x_in[tap*8 +: 8]
    reg [39:0] w;         // packed 5×8b
    reg signed [31:0] bias_in;
    reg [3:0]  a_in, in_ch;
    reg        compute_en_in;
    reg [3:0]  nb;
    reg        relu_en;
    reg        pool_rst;
    wire       pool_write;
    wire signed [7:0] pool_out;

    cp_block dut (
        .clk          (clk),
        .rst          (rst),
        .x_in         (x_in),
        .w            (w),
        .bias_in      (bias_in),
        .a_in         (a_in),
        .in_ch        (in_ch),
        .compute_en_in(compute_en_in),
        .nb           (nb),
        .relu_en      (relu_en),
        .pool_rst     (pool_rst),
        .pool_write   (pool_write),
        .pool_out     (pool_out)
    );

    // ── Clock 100 MHz ───────────────────────────────────────────────────
    initial clk = 0;
    always #5 clk = ~clk;

    // ── Drive one pool sample (1 pixel) ──────────────────────────────────
    // cp_block nhan taps DA on dinh (trong thiet ke that cp_engine cap qua
    // SRW+MUX, va delay a_in/compute_en_in 5 cycle = a_d5/ce_d5). O TB co lap:
    // giu taps CO DINH suot pixel -> do latency 5 cycle khong lam sai gia tri.
    // Moi lan goi tao dung 1 out_valid (a_in=0 mot cycle) -> 1 mau pool.
    // compute_en_in giu cao suot de pool dem duoc relu_v khi no toi (~9 cy sau).
    task pixel;
        input [7:0] t0, t1, t2, t3, t4;   // 5 tap values (INT8)
        begin
            x_in = {t4, t3, t2, t1, t0};   // x_in[0]=t0 ... x_in[4]=t4
            // Buoc 1: dat taps, cho 5 cycle cho MAC pipeline dua taps toi tree_out.
            // a_in=1 -> khong out_valid; ce=1 giu pool dem duoc relu_v cu dang bay.
            compute_en_in = 1'b1; a_in = 4'd1;
            repeat (5) begin @(posedge clk); #1; end
            // Buoc 2: 1 cycle validate -> out_valid=1 chot acc (= conv cua taps nay).
            a_in = 4'd0;
            @(posedge clk); #1;
            a_in = 4'd1;
        end
    endtask

    // Drain: pipeline flush cho relu_v cuoi toi pool. compute_en_in van cao de
    // pool con dem duoc; a_in=1 nen khong sinh out_valid moi.
    task idle_cycles;
        input integer n;
        integer k;
        begin
            for (k = 0; k < n; k = k + 1) begin
                compute_en_in = 1'b1; a_in = 4'd1;
                @(posedge clk); #1;
            end
        end
    endtask

    integer pass_cnt, fail_cnt;
    reg signed [7:0] captured;
    reg              got_write;

    // Latch pool_write result so the check can read it after drain.
    always @(posedge clk) begin
        if (pool_write) begin
            captured  <= pool_out;
            got_write <= 1'b1;
        end
    end

    // Trace moi pixel khi relu_v=1 (de doc tren console / doi chieu Chuong 4).
    always @(posedge clk) begin
        if (dut.relu_v)
            $display("  pixel: conv=%0d  pool_cnt=%0d  pool_out(running max)=%0d",
                $signed(dut.u_accres.relu_out), dut.u_pool.pool_cnt, $signed(pool_out));
    end

    task check;
        input signed [7:0] expect_val;
        input [80*8-1:0] name;
        begin
            if (!got_write) begin
                $display("FAIL [%0s] pool_write NEVER fired", name);
                fail_cnt = fail_cnt + 1;
            end else if (captured === expect_val) begin
                $display("PASS [%0s] pool_out=%0d", name, captured);
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("FAIL [%0s] expected=%0d got=%0d", name, expect_val, captured);
                fail_cnt = fail_cnt + 1;
            end
        end
    endtask

    initial begin
        pass_cnt = 0; fail_cnt = 0; got_write = 0;
        // Static config for the whole run
        in_ch = 4'd1; nb = 4'd0; relu_en = 1'b0; bias_in = 32'sd0;
        w = 40'h0; x_in = 40'h0;
        a_in = 4'd0; compute_en_in = 1'b0; pool_rst = 1'b0;

        // Reset
        rst = 1'b1;
        repeat (5) @(posedge clk); #1;
        rst = 1'b0;
        @(posedge clk); #1;

        $display("=== tb_cp_block_simple: 5 pixel -> 1 pool_out ===");

        // Weight = [1,1,1,1,1] -> conv(pixel) = sum of 5 taps.
        w = {8'd1, 8'd1, 8'd1, 8'd1, 8'd1};

        // pool_rst dau window (giong controller pulse pool_rst dau moi layer).
        pool_rst = 1'b1; @(posedge clk); #1; pool_rst = 1'b0;
        @(posedge clk); #1;

        // Nap 5 pixel, moi pixel co conv sum khac nhau (de MaxPool chon max):
        //   pixel0 taps [1,2,3,4,5]   -> sum = 15
        //   pixel1 taps [10,0,0,0,0]  -> sum = 10
        //   pixel2 taps [20,20,0,0,0] -> sum = 40   <-- max
        //   pixel3 taps [5,5,5,0,0]   -> sum = 15
        //   pixel4 taps [1,1,1,1,0]   -> sum = 4
        // nb=0, bias=0 -> pool_out = max(15,10,40,15,4) = 40
        pixel(8'd1,  8'd2,  8'd3, 8'd4, 8'd5);   // 15
        pixel(8'd10, 8'd0,  8'd0, 8'd0, 8'd0);   // 10
        pixel(8'd20, 8'd20, 8'd0, 8'd0, 8'd0);   // 40
        pixel(8'd5,  8'd5,  8'd5, 8'd0, 8'd0);   // 15
        pixel(8'd1,  8'd1,  8'd1, 8'd1, 8'd0);   //  4

        // Drain pipeline (depth ~11) so all 5 relu_v reach cp_pool.
        idle_cycles(16);
        compute_en_in = 1'b0;

        check(8'sd40, "pool_out_max_of_5_pixels");

        $display("=== SUMMARY: %0d PASS, %0d FAIL ===", pass_cnt, fail_cnt);
        if (fail_cnt == 0) $display("ALL PASS"); else $display("SOME FAIL");
        $finish;
    end

    initial begin
        #50000;
        $display("TIMEOUT");
        $finish;
    end

endmodule
