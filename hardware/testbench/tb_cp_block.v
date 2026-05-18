// tb_cp_block.v — Unit test for cp_block.v
// 18 test cases covering S1-S9 pipeline stages
//
// Key strategy: taps_in=0 → tree_out=0 → acc_final=0 → biased=bias_in
// Use nb=0 for S7/S8/S9 tests (shifted=biased directly, no rounding needed)
// Use nb=8 only for TC02 (round-half-up verification)
// Drive a_in/compute_en_in directly — skip cp_engine delay chain

`timescale 1ns/1ps

module tb_cp_block;

    // ── DUT signals ──────────────────────────────────────────────────
    reg        clk, rst;
    reg [39:0] taps_in;   // packed 5×8b: taps_in[tap*8+:8]
    reg [39:0] w;         // packed 5×8b: w[tap*8+:8]
    reg signed [31:0] bias_in;
    reg [3:0]  a_in, in_ch;
    reg        compute_en_in;
    reg [4:0]  nb;
    reg        relu_en;
    reg        pool_rst;
    wire       pool_write;
    wire signed [7:0] pool_out;

    cp_block dut (
        .clk          (clk),
        .rst          (rst),
        .taps_in      (taps_in),
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

    // ── Clock ─────────────────────────────────────────────────────────
    initial clk = 0;
    always #5 clk = ~clk;

    // ── Counters ──────────────────────────────────────────────────────
    integer pass_cnt, fail_cnt;
    time    tc_t0;

    // ── Tasks ─────────────────────────────────────────────────────────

    task apply_reset;
        integer ri;
        begin
            rst = 1; pool_rst = 0;
            compute_en_in = 0; a_in = 0; in_ch = 4'd1;
            bias_in = 0; nb = 0; relu_en = 0;
            taps_in = 40'h0;
            w       = 40'h0;
            for (ri = 0; ri < 10; ri = ri + 1) begin
                @(posedge clk); #1;
            end
            rst = 0;
            @(posedge clk); #1;
            tc_t0 = $time;
        end
    endtask

    // Drive N out_valid pulses for IN_CH=1 (a_in=0, compute_en_in=1)
    task drive_pixels_inch1;
        input integer n;
        integer i;
        begin
            for (i = 0; i < n; i = i + 1) begin
                compute_en_in = 1; a_in = 0;
                @(posedge clk); #1;
            end
            compute_en_in = 0;
        end
    endtask

    // Drive N out_valid pulses for IN_CH=4 (cycle a_in 0..3, out_valid at a_in=3)
    task drive_pixels_inch4;
        input integer n;
        integer i, j;
        begin
            for (i = 0; i < n; i = i + 1) begin
                for (j = 0; j < 4; j = j + 1) begin
                    compute_en_in = 1; a_in = j;
                    @(posedge clk); #1;
                end
            end
            compute_en_in = 0;
        end
    endtask

    // Drive N out_valid pulses for IN_CH=8 (cycle a_in 0..7)
    task drive_pixels_inch8;
        input integer n;
        integer i, j;
        begin
            for (i = 0; i < n; i = i + 1) begin
                for (j = 0; j < 8; j = j + 1) begin
                    compute_en_in = 1; a_in = j;
                    @(posedge clk); #1;
                end
            end
            compute_en_in = 0;
        end
    endtask

    // Wait for pool_write with timeout; returns 1 on success, 0 on timeout
    task wait_pool_write;
        output reg success;
        integer timeout;
        begin
            success = 0;
            for (timeout = 0; timeout < 40 && !success; timeout = timeout + 1) begin
                @(posedge clk); #1;
                if (pool_write) success = 1;
            end
        end
    endtask

    // Check pool_out value and print PASS/FAIL
    task check_val;
        input signed [7:0] expected;
        input [127:0] tc_name;
        begin
            if ($signed(pool_out) === $signed(expected)) begin
                $display("PASS [%0s] pool_out=%0d  (%0t ns)", tc_name, $signed(pool_out), $time - tc_t0);
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("FAIL [%0s] expected=%0d got=%0d  (%0t ns)", tc_name, $signed(expected), $signed(pool_out), $time - tc_t0);
                fail_cnt = fail_cnt + 1;
            end
        end
    endtask

    task check_pool_write_count;
        input integer expected_count;
        input [127:0] tc_name;
        input integer cycles;
        integer i, cnt;
        begin
            cnt = 0;
            for (i = 0; i < cycles; i = i + 1) begin
                @(posedge clk); #1;
                if (pool_write) cnt = cnt + 1;
            end
            if (cnt === expected_count) begin
                $display("PASS [%0s] pool_write count=%0d  (%0t ns)", tc_name, cnt, $time - tc_t0);
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("FAIL [%0s] expected pw_count=%0d got=%0d  (%0t ns)", tc_name, expected_count, cnt, $time - tc_t0);
                fail_cnt = fail_cnt + 1;
            end
        end
    endtask

    // ── Main test sequence ─────────────────────────────────────────────
    reg success;

    initial begin
        pass_cnt = 0; fail_cnt = 0;
        $display("=== tb_cp_block: starting 19 test cases ===");

        // ── TC01: Basic pipeline S1-S5 ─────────────────────────────────
        // taps=[1,2,3,4,5], w=[1,1,1,1,1] → tree_out=15
        // nb=8, bias=0 → shifted=0 → pool_out=0
        // TC01a: simplest case first (taps=0,bias=0,nb=0 → pool_out=0)
        apply_reset;
        in_ch = 4'd1; nb = 5'd0; relu_en = 0;
        taps_in = 40'h0; w = 40'h0; bias_in = 0;
        drive_pixels_inch1(5);
        wait_pool_write(success);
        check_val(8'sd0, "TC01a_zero_pipeline");

        // TC01b: taps=[1..5], w=[1..1], nb=8 → pool_out=0
        apply_reset;
        in_ch = 4'd1; nb = 5'd8; relu_en = 0;
        taps_in = {8'd5, 8'd4, 8'd3, 8'd2, 8'd1};
        w       = {8'd1, 8'd1, 8'd1, 8'd1, 8'd1};
        bias_in = 0;
        drive_pixels_inch1(5);
        wait_pool_write(success);
        check_val(8'sd0, "TC01b_basic_pipeline");

        // ── TC02: Round-half-up (nb=8) ─────────────────────────────────
        // taps=0: biased=bias_in
        // bias=128: (128+128)>>8=1 → pool_out=1
        apply_reset;
        in_ch = 4'd1; nb = 5'd8; relu_en = 0;
        taps_in = 40'h0;
        w       = {8'd0, 8'd0, 8'd0, 8'd0, 8'd1};  // w[0]=1
        bias_in = 32'sd128;
        drive_pixels_inch1(5);
        wait_pool_write(success);
        check_val(8'sd1, "TC02a_round_half_up_128");

        // bias=127: (127+128)>>8=0 → pool_out=0
        apply_reset;
        in_ch = 4'd1; nb = 5'd8; relu_en = 0;
        taps_in = 40'h0;
        w       = 40'h0;
        bias_in = 32'sd127;
        drive_pixels_inch1(5);
        wait_pool_write(success);
        check_val(8'sd0, "TC02b_round_half_up_127");

        // ── TC03: Clamp upper (nb=0, bias=200 → shifted=200 → clamped=127) ──
        apply_reset;
        in_ch = 4'd1; nb = 5'd0; relu_en = 0;
        taps_in = 40'h0;
        w       = 40'h0;
        bias_in = 32'sd200;
        drive_pixels_inch1(5);
        wait_pool_write(success);
        check_val(8'sd127, "TC03_clamp_upper");

        // ── TC04: Clamp lower (nb=0, bias=-200 → clamped=-127) ─────────
        apply_reset;
        in_ch = 4'd1; nb = 5'd0; relu_en = 0;
        taps_in = 40'h0;
        w       = 40'h0;
        bias_in = -32'sd200;
        drive_pixels_inch1(5);
        wait_pool_write(success);
        check_val(-8'sd127, "TC04_clamp_lower");

        // ── TC05: Clamp exact upper (bias=127, nb=0 → clamped=127) ─────
        apply_reset;
        in_ch = 4'd1; nb = 5'd0; relu_en = 0;
        taps_in = 40'h0;
        w       = 40'h0;
        bias_in = 32'sd127;
        drive_pixels_inch1(5);
        wait_pool_write(success);
        check_val(8'sd127, "TC05_clamp_exact_upper");

        // ── TC06: Clamp exact lower (bias=-127, nb=0 → clamped=-127) ───
        apply_reset;
        in_ch = 4'd1; nb = 5'd0; relu_en = 0;
        taps_in = 40'h0;
        w       = 40'h0;
        bias_in = -32'sd127;
        drive_pixels_inch1(5);
        wait_pool_write(success);
        check_val(-8'sd127, "TC06_clamp_exact_lower");

        // ── TC07: ReLU on + negative (relu_en=1, bias=-50 → relu_out=0) ─
        apply_reset;
        in_ch = 4'd1; nb = 5'd0; relu_en = 1;
        taps_in = 40'h0;
        w       = 40'h0;
        bias_in = -32'sd50;
        drive_pixels_inch1(5);
        wait_pool_write(success);
        check_val(8'sd0, "TC07_relu_on_negative");

        // ── TC08: ReLU on + positive (relu_en=1, bias=50 → relu_out=50) ─
        apply_reset;
        in_ch = 4'd1; nb = 5'd0; relu_en = 1;
        taps_in = 40'h0;
        w       = 40'h0;
        bias_in = 32'sd50;
        drive_pixels_inch1(5);
        wait_pool_write(success);
        check_val(8'sd50, "TC08_relu_on_positive");

        // ── TC09: ReLU off + negative pass-through (relu_en=0, bias=-50) ─
        apply_reset;
        in_ch = 4'd1; nb = 5'd0; relu_en = 0;
        taps_in = 40'h0;
        w       = 40'h0;
        bias_in = -32'sd50;
        drive_pixels_inch1(5);
        wait_pool_write(success);
        check_val(-8'sd50, "TC09_relu_off_negative");

        // ── TC10: MaxPool — max at first [100,50,30,20,10] ─────────────
        // Feed 5 pixels with decreasing bias_in; pool_out should be 100
        // Bias drive shifted by 1 iter: RTL S_bias samples bias_in 1 cy after out_valid,
        // so bias for pixel#N must be set during the cycle of pixel#N+1's out_valid.
        apply_reset;
        in_ch = 4'd1; nb = 5'd0; relu_en = 0;
        taps_in = 40'h0;
        w       = 40'h0;
        bias_in = 32'sd0;   drive_pixels_inch1(1);  // pixel #0 in (bias for #0 driven next iter)
        bias_in = 32'sd100; drive_pixels_inch1(1);  // pixel #1 in, samples bias=100 for #0
        bias_in = 32'sd50;  drive_pixels_inch1(1);  // pixel #2 in, samples bias=50  for #1
        bias_in = 32'sd30;  drive_pixels_inch1(1);  // pixel #3 in, samples bias=30  for #2
        bias_in = 32'sd20;  drive_pixels_inch1(1);  // pixel #4 in, samples bias=20  for #3
        bias_in = 32'sd10;                          // hold bias=10 for #4 drain
        wait_pool_write(success);
        check_val(8'sd100, "TC10_pool_max_at_first");

        // ── TC11: MaxPool — max at last [10,20,30,50,100] ─────────────
        apply_reset;
        in_ch = 4'd1; nb = 5'd0; relu_en = 0;
        taps_in = 40'h0;
        w       = 40'h0;
        bias_in = 32'sd0;   drive_pixels_inch1(1);  // pixel #0 in
        bias_in = 32'sd10;  drive_pixels_inch1(1);  // bias=10  for #0
        bias_in = 32'sd20;  drive_pixels_inch1(1);  // bias=20  for #1
        bias_in = 32'sd30;  drive_pixels_inch1(1);  // bias=30  for #2
        bias_in = 32'sd50;  drive_pixels_inch1(1);  // bias=50  for #3
        bias_in = 32'sd100;                         // hold bias=100 for #4 drain
        wait_pool_write(success);
        check_val(8'sd100, "TC11_pool_max_at_last");

        // ── TC12: pool_write fires exactly once for 5 pixels ───────────
        apply_reset;
        in_ch = 4'd1; nb = 5'd0; relu_en = 0;
        taps_in = 40'h0;
        w       = 40'h0;
        bias_in = 32'sd42;
        drive_pixels_inch1(5);
        // Count pool_write pulses in next 20 cycles: expect exactly 1
        begin : tc12_check
            integer cnt12, t12;
            cnt12 = 0;
            for (t12 = 0; t12 < 20; t12 = t12 + 1) begin
                @(posedge clk); #1;
                if (pool_write) cnt12 = cnt12 + 1;
            end
            if (cnt12 === 1) begin
                $display("PASS [TC12_pool_write_once] count=%0d  (%0t ns)", cnt12, $time - tc_t0);
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("FAIL [TC12_pool_write_once] expected=1 got=%0d  (%0t ns)", cnt12, $time - tc_t0);
                fail_cnt = fail_cnt + 1;
            end
        end

        // ── TC13: 2 windows — verify both pool_write and pool_out ──────
        // Window 1: 5 pixels = 20; Window 2: 5 pixels = 5
        apply_reset;
        in_ch = 4'd1; nb = 5'd0; relu_en = 0;
        taps_in = 40'h0;
        w       = 40'h0;
        bias_in = 32'sd20; drive_pixels_inch1(5);
        // Wait for first pool_write before driving window 2
        wait_pool_write(success);
        check_val(8'sd20, "TC13a_window1");
        bias_in = 32'sd5;  drive_pixels_inch1(5);
        // Wait for second pool_write → check pool_out=5
        wait_pool_write(success);
        check_val(8'sd5, "TC13b_window2");

        // ── TC14: IN_CH=4 accumulation ─────────────────────────────────
        // taps=[10,0,0,0,0], w=[1,0,0,0,0]: tree_out=10 per cycle
        // ACC 4 cycles: a_in=0 RST, a_in=1..3 ACC; acc_final=4*10=40
        // nb=0, bias=0: biased=40, shifted=40, clamped=40, pool_out=40
        apply_reset;
        in_ch = 4'd4; nb = 5'd0; relu_en = 0;
        taps_in = {8'd0, 8'd0, 8'd0, 8'd0, 8'd10};  // taps_in[0]=10
        w       = {8'd0, 8'd0, 8'd0, 8'd0, 8'd1};   // w[0]=1
        bias_in = 0;
        drive_pixels_inch4(5);
        wait_pool_write(success);
        check_val(8'sd40, "TC14_inch4_accumulate");

        // ── TC15: IN_CH=8 accumulation ─────────────────────────────────
        // acc_final = 8*10=80; nb=0, bias=0 → pool_out=80
        apply_reset;
        in_ch = 4'd8; nb = 5'd0; relu_en = 0;
        taps_in = {8'd0, 8'd0, 8'd0, 8'd0, 8'd10};  // taps_in[0]=10
        w       = {8'd0, 8'd0, 8'd0, 8'd0, 8'd1};   // w[0]=1
        bias_in = 0;
        drive_pixels_inch8(5);
        wait_pool_write(success);
        check_val(8'sd80, "TC15_inch8_accumulate");

        // ── TC16: compute_en_in=0 → no out_valid, no pool_write (NOP) ──
        apply_reset;
        in_ch = 4'd1; nb = 5'd0; relu_en = 0;
        taps_in = 40'h0;
        w       = 40'h0;
        bias_in = 32'sd99;
        compute_en_in = 0; a_in = 0;  // NOP: no valid
        begin : tc16_check
            integer cnt16, t16;
            cnt16 = 0;
            for (t16 = 0; t16 < 50; t16 = t16 + 1) begin
                @(posedge clk); #1;
                if (pool_write) cnt16 = cnt16 + 1;
            end
            if (cnt16 === 0) begin
                $display("PASS [TC16_nop_no_pool_write]  (%0t ns)", $time - tc_t0);
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("FAIL [TC16_nop_no_pool_write] got pool_write count=%0d  (%0t ns)", cnt16, $time - tc_t0);
                fail_cnt = fail_cnt + 1;
            end
        end

        // ── TC17: pool_rst resets pool_cnt mid-window ──────────────────
        // Feed 3 pixels, then pool_rst, then feed 5 more → pool_write only after 5 new
        apply_reset;
        in_ch = 4'd1; nb = 5'd0; relu_en = 0;
        taps_in = 40'h0;
        w       = 40'h0;
        bias_in = 32'sd60;
        drive_pixels_inch1(3);           // 3 pixels into window (pool_cnt→3)
        // Wait for pipeline to drain those 3 through to relu_v (~8 cycles)
        repeat(12) @(posedge clk); #1;
        pool_rst = 1; @(posedge clk); #1; pool_rst = 0;  // reset pool_cnt
        // Now feed 5 fresh pixels → pool_write should fire after 5
        bias_in = 32'sd33;
        drive_pixels_inch1(5);
        wait_pool_write(success);
        if (success) begin
            $display("PASS [TC17_pool_rst_mid_window] pool_write fired after 5 fresh pixels  (%0t ns)", $time - tc_t0);
            pass_cnt = pass_cnt + 1;
        end else begin
            $display("FAIL [TC17_pool_rst_mid_window] pool_write did not fire  (%0t ns)", $time - tc_t0);
            fail_cnt = fail_cnt + 1;
        end
        check_val(8'sd33, "TC17_pool_rst_value");

        // ── TC18: rst clears all regs, no stale pool_write ─────────────
        // Start feeding pixels, then rst mid-pipeline → no spurious pool_write after rst
        apply_reset;
        in_ch = 4'd1; nb = 5'd0; relu_en = 0;
        taps_in = 40'h0;
        w       = 40'h0;
        bias_in = 32'sd77;
        drive_pixels_inch1(4);  // partially fill window
        repeat(3) @(posedge clk); #1;
        // Apply rst
        rst = 1; @(posedge clk); #1; rst = 0;
        compute_en_in = 0;
        // Observe next 30 cycles: no pool_write should occur
        begin : tc18_check
            integer cnt18, t18;
            cnt18 = 0;
            for (t18 = 0; t18 < 30; t18 = t18 + 1) begin
                @(posedge clk); #1;
                if (pool_write) cnt18 = cnt18 + 1;
            end
            if (cnt18 === 0) begin
                $display("PASS [TC18_rst_clears_pipeline]  (%0t ns)", $time - tc_t0);
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("FAIL [TC18_rst_clears_pipeline] spurious pool_write count=%0d  (%0t ns)", cnt18, $time - tc_t0);
                fail_cnt = fail_cnt + 1;
            end
        end

        // ── Summary ────────────────────────────────────────────────────
        $display("=== SUMMARY: %0d PASS, %0d FAIL ===", pass_cnt, fail_cnt);
        if (fail_cnt === 0)
            $display("ALL TESTS PASSED");
        else
            $display("SOME TESTS FAILED");
        $finish;
    end

    // Timeout watchdog
    initial begin
        #100000;
        $display("TIMEOUT — simulation exceeded 100us");
        $finish;
    end

endmodule
