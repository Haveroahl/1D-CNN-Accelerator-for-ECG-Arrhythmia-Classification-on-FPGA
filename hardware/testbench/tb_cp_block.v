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
    reg [39:0] taps_in;   // packed 5×8b: taps_in[tap*8+:8] → drives x_in port
    reg [39:0] w;         // packed 5×8b: w[tap*8+:8]
    reg signed [31:0] bias_in;
    reg [3:0]  a_in, in_ch;
    reg        compute_en_in;
    reg [3:0]  nb;        // rescale shift (max used = 8; port narrowed to [3:0])
    reg        relu_en;
    reg        pool_rst;
    wire       pool_write;
    wire signed [7:0] pool_out;

    cp_block dut (
        .clk          (clk),
        .rst          (rst),
        .x_in         (taps_in),
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

    // ── pool_write capture ──────────────────────────────────────────────
    // pool_write is a 1-cycle pulse that fires during the drive/drain phase,
    // before a subsequent wait task runs. Latch it (and the pooled value) so
    // checks can read it regardless of when they execute. Cleared by apply_reset.
    reg               pw_seen;
    reg signed [7:0]  pw_value;
    integer           pw_count;
    always @(posedge clk) begin
        if (pool_write) begin
            pw_seen  <= 1'b1;
            pw_value <= pool_out;
            pw_count <= pw_count + 1;
        end
    end

    // ── Tasks ─────────────────────────────────────────────────────────

    task apply_reset;
        integer ri;
        begin
            rst = 1; pool_rst = 0;
            compute_en_in = 0; a_in = 0; in_ch = 4'd1;
            bias_in = 0; nb = 0; relu_en = 0;
            taps_in = 40'h0;
            w       = 40'h0;
            pw_seen = 1'b0; pw_value = 8'sd0; pw_count = 0;
            for (ri = 0; ri < 10; ri = ri + 1) begin
                @(posedge clk); #1;
            end
            rst = 0;
            @(posedge clk); #1;
            tc_t0 = $time;
        end
    endtask

    // ── Pipeline-depth model (how the pool stage is driven) ─────────────────
    // The S9 MaxPool stage counts a pixel while (relu_v && compute_en_in). relu_v
    // lags out_valid by 9 pipeline stages (prod→sum01/23→sum0123→tree_out→
    // acc_final→biased→shifted→clamped→relu_out), so a pixel is pooled only if
    // compute_en_in is still high when its relu_v arrives ~9 cycles later.
    //
    // For IN_CH=1, out_valid = compute_en_in && (a_in == 0). To feed exactly K
    // pooled pixels we therefore: (1) drive K out_valid pulses (a_in=0), then
    // (2) DRAIN — hold compute_en_in=1 but park a_in=1 (≠ in_ch-1) so no NEW
    // out_valid is generated while the in-flight pixels still drain through the
    // gate. This counts exactly K pixels (one pool_write per 5), with no spurious
    // extra pixels, matching how the controller streams a window in the full design.
    localparam integer DRAIN_N = 5;    // one pool window

    // Hold compute_en_in high with a_in parked off the valid slot so in-flight
    // pixels reach the pool stage without creating new out_valids (IN_CH=1).
    task drain_pipeline;
        input integer cycles;
        integer d;
        begin
            for (d = 0; d < cycles; d = d + 1) begin
                compute_en_in = 1; a_in = 1;   // a_in != in_ch-1 => no out_valid
                @(posedge clk); #1;
            end
        end
    endtask

    // Drive N out_valid pulses for IN_CH=1 (a_in=0), then drain so they all pool.
    task drive_pixels_inch1;
        input integer n;
        integer i;
        begin
            for (i = 0; i < n; i = i + 1) begin
                compute_en_in = 1; a_in = 0;
                @(posedge clk); #1;
            end
            drain_pipeline(14);            // flush 9-stage pipeline + pool window
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
            // Drain: keep compute_en_in high but park a_in=0 (!=in_ch-1=3) -> no valid
            for (i = 0; i < 14; i = i + 1) begin
                compute_en_in = 1; a_in = 0;
                @(posedge clk); #1;
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
            // Drain: keep compute_en_in high but park a_in=0 (!=in_ch-1=7) -> no valid
            for (i = 0; i < 14; i = i + 1) begin
                compute_en_in = 1; a_in = 0;
                @(posedge clk); #1;
            end
            compute_en_in = 0;
        end
    endtask

    // Wait until a pool_write has been latched (may already have fired during
    // the drive/drain phase). Returns 1 on success, 0 on timeout.
    task wait_pool_write;
        output reg success;
        integer timeout;
        begin
            for (timeout = 0; timeout < 40 && !pw_seen; timeout = timeout + 1) begin
                @(posedge clk); #1;
            end
            success = pw_seen;
        end
    endtask

    // Check the latched pooled value (pw_value) and print PASS/FAIL
    task check_val;
        input signed [7:0] expected;
        input [127:0] tc_name;
        begin
            if ($signed(pw_value) === $signed(expected)) begin
                $display("PASS [%0s] pool_out=%0d  (%0t ns)", tc_name, $signed(pw_value), $time - tc_t0);
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("FAIL [%0s] expected=%0d got=%0d  (%0t ns)", tc_name, $signed(expected), $signed(pw_value), $time - tc_t0);
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
        drive_pixels_inch1(DRAIN_N);
        wait_pool_write(success);
        check_val(8'sd0, "TC01a_zero_pipeline");

        // TC01b: taps=[1..5], w=[1..1], nb=8 → pool_out=0
        apply_reset;
        in_ch = 4'd1; nb = 5'd8; relu_en = 0;
        taps_in = {8'd5, 8'd4, 8'd3, 8'd2, 8'd1};
        w       = {8'd1, 8'd1, 8'd1, 8'd1, 8'd1};
        bias_in = 0;
        drive_pixels_inch1(DRAIN_N);
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
        drive_pixels_inch1(DRAIN_N);
        wait_pool_write(success);
        check_val(8'sd1, "TC02a_round_half_up_128");

        // bias=127: (127+128)>>8=0 → pool_out=0
        apply_reset;
        in_ch = 4'd1; nb = 5'd8; relu_en = 0;
        taps_in = 40'h0;
        w       = 40'h0;
        bias_in = 32'sd127;
        drive_pixels_inch1(DRAIN_N);
        wait_pool_write(success);
        check_val(8'sd0, "TC02b_round_half_up_127");

        // ── TC03: Clamp upper (nb=0, bias=200 → shifted=200 → clamped=127) ──
        apply_reset;
        in_ch = 4'd1; nb = 5'd0; relu_en = 0;
        taps_in = 40'h0;
        w       = 40'h0;
        bias_in = 32'sd200;
        drive_pixels_inch1(DRAIN_N);
        wait_pool_write(success);
        check_val(8'sd127, "TC03_clamp_upper");

        // ── TC04: Clamp lower (nb=0, bias=-200 → clamped=-127) ─────────
        apply_reset;
        in_ch = 4'd1; nb = 5'd0; relu_en = 0;
        taps_in = 40'h0;
        w       = 40'h0;
        bias_in = -32'sd200;
        drive_pixels_inch1(DRAIN_N);
        wait_pool_write(success);
        check_val(-8'sd127, "TC04_clamp_lower");

        // ── TC05: Clamp exact upper (bias=127, nb=0 → clamped=127) ─────
        apply_reset;
        in_ch = 4'd1; nb = 5'd0; relu_en = 0;
        taps_in = 40'h0;
        w       = 40'h0;
        bias_in = 32'sd127;
        drive_pixels_inch1(DRAIN_N);
        wait_pool_write(success);
        check_val(8'sd127, "TC05_clamp_exact_upper");

        // ── TC06: Clamp exact lower (bias=-127, nb=0 → clamped=-127) ───
        apply_reset;
        in_ch = 4'd1; nb = 5'd0; relu_en = 0;
        taps_in = 40'h0;
        w       = 40'h0;
        bias_in = -32'sd127;
        drive_pixels_inch1(DRAIN_N);
        wait_pool_write(success);
        check_val(-8'sd127, "TC06_clamp_exact_lower");

        // ── TC07: ReLU on + negative (relu_en=1, bias=-50 → relu_out=0) ─
        apply_reset;
        in_ch = 4'd1; nb = 5'd0; relu_en = 1;
        taps_in = 40'h0;
        w       = 40'h0;
        bias_in = -32'sd50;
        drive_pixels_inch1(DRAIN_N);
        wait_pool_write(success);
        check_val(8'sd0, "TC07_relu_on_negative");

        // ── TC08: ReLU on + positive (relu_en=1, bias=50 → relu_out=50) ─
        apply_reset;
        in_ch = 4'd1; nb = 5'd0; relu_en = 1;
        taps_in = 40'h0;
        w       = 40'h0;
        bias_in = 32'sd50;
        drive_pixels_inch1(DRAIN_N);
        wait_pool_write(success);
        check_val(8'sd50, "TC08_relu_on_positive");

        // ── TC09: ReLU off + negative pass-through (relu_en=0, bias=-50) ─
        apply_reset;
        in_ch = 4'd1; nb = 5'd0; relu_en = 0;
        taps_in = 40'h0;
        w       = 40'h0;
        bias_in = -32'sd50;
        drive_pixels_inch1(DRAIN_N);
        wait_pool_write(success);
        check_val(-8'sd50, "TC09_relu_off_negative");

        // ── TC10: MaxPool — max at first [100,50,30,20,10] ─────────────
        // Feed 5 pixels with decreasing bias_in; pool_out should be 100.
        // Bias is now folded into the accumulator init (a_in==0 cycle), so bias
        // for pixel#N is sampled DURING pixel#N's own out_valid cycle — set bias
        // before that cycle's edge (1 cy earlier than the old S_bias contract).
        apply_reset;
        in_ch = 4'd1; nb = 5'd0; relu_en = 0;
        taps_in = 40'h0;
        w       = 40'h0;
        compute_en_in = 1; a_in = 0;
        bias_in = 32'sd100; @(posedge clk); #1;  // pixel #0; bias=100 for #0
        bias_in = 32'sd50;  @(posedge clk); #1;  // pixel #1; bias=50  for #1
        bias_in = 32'sd30;  @(posedge clk); #1;  // pixel #2; bias=30  for #2
        bias_in = 32'sd20;  @(posedge clk); #1;  // pixel #3; bias=20  for #3
        bias_in = 32'sd10;  @(posedge clk); #1;  // pixel #4; bias=10  for #4
        drain_pipeline(14);                       // flush; no new out_valid (a_in=1)
        compute_en_in = 0;
        wait_pool_write(success);
        check_val(8'sd100, "TC10_pool_max_at_first");

        // ── TC11: MaxPool — max at last [10,20,30,50,100] ─────────────
        apply_reset;
        in_ch = 4'd1; nb = 5'd0; relu_en = 0;
        taps_in = 40'h0;
        w       = 40'h0;
        compute_en_in = 1; a_in = 0;
        bias_in = 32'sd10;  @(posedge clk); #1;  // pixel #0; bias=10  for #0
        bias_in = 32'sd20;  @(posedge clk); #1;  // pixel #1; bias=20  for #1
        bias_in = 32'sd30;  @(posedge clk); #1;  // pixel #2; bias=30  for #2
        bias_in = 32'sd50;  @(posedge clk); #1;  // pixel #3; bias=50  for #3
        bias_in = 32'sd100; @(posedge clk); #1;  // pixel #4; bias=100 for #4
        drain_pipeline(14);
        compute_en_in = 0;
        wait_pool_write(success);
        check_val(8'sd100, "TC11_pool_max_at_last");

        // ── TC12: pool_write fires exactly once for 5 pixels ───────────
        apply_reset;
        in_ch = 4'd1; nb = 5'd0; relu_en = 0;
        taps_in = 40'h0;
        w       = 40'h0;
        bias_in = 32'sd42;
        drive_pixels_inch1(DRAIN_N);   // exactly 5 out_valids = one window + drain
        // pw_count latches every pool_write since reset: expect exactly 1
        begin : tc12_check
            if (pw_count === 1) begin
                $display("PASS [TC12_pool_write_once] count=%0d  (%0t ns)", pw_count, $time - tc_t0);
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("FAIL [TC12_pool_write_once] expected=1 got=%0d  (%0t ns)", pw_count, $time - tc_t0);
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

        // ── TC19: FEC — relu_v=1 while compute_en_in=0 must NOT pool ───
        // Closes the focused-condition gap on `if (relu_v && compute_en_in)`
        // (cp_block.v:181 Row 3): proves compute_en_in independently gates the
        // pool. Feed pixels with compute_en_in=1 until relu_v rises, then drop
        // compute_en_in on that exact cycle and check pool_cnt does NOT advance.
        apply_reset;
        in_ch = 4'd1; nb = 5'd0; relu_en = 0;
        taps_in = 40'h0;
        w       = 40'h0;
        bias_in = 32'sd55;
        begin : tc19_check
            integer t19, pc_before;
            reg armed;
            armed = 0;
            // Stream pixels (compute_en_in=1) and watch internal relu_v.
            for (t19 = 0; t19 < 30 && !armed; t19 = t19 + 1) begin
                compute_en_in = 1; a_in = 0;
                @(posedge clk); #1;
                // When relu_v is asserted, the NEXT pool eval would count it.
                if (dut.relu_v === 1'b1) begin
                    pc_before = dut.u_pool.pool_cnt;   // snapshot before the gated cycle
                    compute_en_in = 0;          // <-- relu_v=1 AND compute_en_in=0
                    @(posedge clk); #1;          // pool eval happens here, gated off
                    armed = 1;
                end
            end
            if (!armed) begin
                $display("FAIL [TC19_fec_relu_v_no_compute_en] relu_v never observed  (%0t ns)", $time - tc_t0);
                fail_cnt = fail_cnt + 1;
            end else if (dut.u_pool.pool_cnt === pc_before) begin
                $display("PASS [TC19_fec_relu_v_no_compute_en] pool_cnt held=%0d (relu_v=1,ce=0 not counted)  (%0t ns)", dut.u_pool.pool_cnt, $time - tc_t0);
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("FAIL [TC19_fec_relu_v_no_compute_en] pool_cnt advanced %0d->%0d despite ce=0  (%0t ns)", pc_before, dut.u_pool.pool_cnt, $time - tc_t0);
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
