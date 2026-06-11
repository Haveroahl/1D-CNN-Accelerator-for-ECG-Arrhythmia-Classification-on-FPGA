// tb_overlap.v — Overlapped input reload throughput test for ecg_accelerator_top
//
// Goal: prove the isram_free overlap mechanism. After Conv1 of window N has
// released input_sram (state enters CONV2), the controller raises isram_free.
// From that point the NEXT window's 2500 samples can be written into input_sram
// while Conv2/3/4/GAP of window N still run — hiding load latency behind compute.
//
// Drives the DUT through the Avalon-MM slave (same path as tb_top / on-board),
// reading isram_free from STATUS reg 0x04 bit[2].
//
// Two passes over 3 windows (samples 0,1,2), measured in clock cycles:
//   PASS A  — SEQUENTIAL : load → start → wait done → read, repeat (no overlap)
//   PASS B  — OVERLAPPED : load(0) → start(0); while busy, once isram_free load(1);
//             on done(0) start(1) immediately; load(2) during window 1; etc.
//
// Correctness gate: all 3 results in BOTH passes must match expected_results.
// Throughput gate : PASS B total cycles < PASS A total cycles (overlap saved time).
//
// Requires (same as tb_top): RTL/conv*.hex, fc_weights.hex, conv_bias.hex,
//   testbench/ecg_sample0..2.hex, testbench/expected_results.hex
//
// NOTE on Avalon load cost: loading via the 3-writes-per-byte register map is
// slow (~7500 Avalon writes per window). Overlap hides at most the Conv2..GAP
// window (~2582 cycles) of that load; any load time beyond that is still serial.
// This TB reports the measured saving honestly — it does not assume load fits.

`timescale 1ns/1ps

module tb_overlap;

    // ── DUT signals ───────────────────────────────────────────────────
    reg        clk, rst, rst_n;
    reg [4:0]  avs_address;
    reg        avs_write, avs_read;
    reg [31:0] avs_writedata;
    wire [31:0] avs_readdata;

    ecg_accelerator_top u_top (
        .clk          (clk),
        .rst          (rst),
        .rst_n        (rst_n),
        .avs_address  (avs_address),
        .avs_write    (avs_write),
        .avs_read     (avs_read),
        .avs_writedata(avs_writedata),
        .avs_readdata (avs_readdata)
    );

    // ── Clock: 10 ns period ───────────────────────────────────────────
    initial clk = 0;
    always #5 clk = ~clk;

    // ── Sample data + expected results ────────────────────────────────
    reg [7:0]  ecg_win   [0:2][0:2499];   // 3 windows
    reg [7:0]  expected_results [0:2];

    // Temp linear buffers for $readmemh (can't readmemh into a 2D slice directly)
    reg [7:0] win0_tmp [0:2499];
    reg [7:0] win1_tmp [0:2499];
    reg [7:0] win2_tmp [0:2499];

    integer    pass_cnt, fail_cnt;

    // ── Avalon-MM helpers (verbatim from tb_top) ──────────────────────
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

    task avs_rd;
        input  [4:0]  addr;
        output [31:0] data;
        begin
            @(negedge clk);
            avs_address = addr;
            avs_read    = 1;
            @(posedge clk); #1;
            data     = avs_readdata;
            avs_read = 0;
        end
    endtask

    // Load one window (win index w) into input_sram via Avalon (3 wr/byte).
    task load_window;
        input integer w;
        integer i;
        begin
            for (i = 0; i < 2500; i = i + 1) begin
                avs_wr(5'h00, {24'h0, ecg_win[w][i]});  // DATA_IN
                avs_wr(5'h01, i[31:0]);                  // ADDR_IN
                avs_wr(5'h02, 32'd1);                    // WR_EN
            end
            @(posedge clk); #1;   // commit last we pulse (see tb_top note)
        end
    endtask

    task pulse_start;
        begin
            avs_wr(5'h03, 32'd1);   // START (also clears done_latched)
            // busy rises a couple cycles after start (IDLE→LOAD_INPUT→CONV1).
            // Settle so the subsequent busy-poll doesn't exit before busy asserts.
            @(posedge clk); @(posedge clk); #1;
        end
    endtask

    // Poll STATUS until busy clears; return latched result. Counts cycles
    // from the START pulse already issued by the caller.
    task wait_done_read;
        output [1:0] cls;
        reg [31:0] status;
        integer poll_iter;
        begin
            status = 1; poll_iter = 0;
            while (status[0] && poll_iter < 20000) begin
                @(posedge clk); #1;
                avs_rd(5'h04, status);   // [0]=busy
                poll_iter = poll_iter + 1;
            end
            avs_rd(5'h05, status);
            cls = status[1:0];
        end
    endtask

    task apply_reset;
        begin
            rst = 1; rst_n = 0;
            avs_write = 0; avs_read = 0;
            @(posedge clk); @(posedge clk); #1;
            rst = 0; rst_n = 1;
            @(posedge clk); #1;
        end
    endtask

    // Poll STATUS once: returns busy (bit0) and isram_free (bit2).
    task read_status;
        output busy_bit;
        output free_bit;
        reg [31:0] s;
        begin
            avs_rd(5'h04, s);
            busy_bit = s[0];
            free_bit = s[2];
        end
    endtask

    // ── Result check ──────────────────────────────────────────────────
    task check_result;
        input [255:0] tag;
        input integer w;
        input [1:0]   got;
        begin
            if (got === expected_results[w][1:0]) begin
                $display("PASS [%0s] window %0d class=%0d", tag, w, got);
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("FAIL [%0s] window %0d class=%0d expected=%0d",
                         tag, w, got, expected_results[w][1:0]);
                fail_cnt = fail_cnt + 1;
            end
        end
    endtask

    // ── Measurement ───────────────────────────────────────────────────
    time   seq_t0, seq_t1, ovl_t0, ovl_t1;
    integer seq_cycles, ovl_cycles;
    reg [1:0] cls0, cls1, cls2;
    integer w;
    reg [31:0] status;
    integer guard;
    reg bsy, fre;

    initial begin
        pass_cnt = 0; fail_cnt = 0;

        // Load 3 windows from the 3 golden samples.
        $readmemh("ecg_sample0.hex", win0_tmp);
        $readmemh("ecg_sample1.hex", win1_tmp);
        $readmemh("ecg_sample2.hex", win2_tmp);
        for (w = 0; w < 2500; w = w + 1) begin
            ecg_win[0][w] = win0_tmp[w];
            ecg_win[1][w] = win1_tmp[w];
            ecg_win[2][w] = win2_tmp[w];
        end
        $readmemh("expected_results.hex", expected_results);

        // ════════════════════════════════════════════════════════════════
        // PASS A — SEQUENTIAL (baseline, no overlap)
        //   load → start → wait → read, fully serial, 3 windows.
        // ════════════════════════════════════════════════════════════════
        apply_reset;
        seq_t0 = $time;
        load_window(0); pulse_start; wait_done_read(cls0);
        load_window(1); pulse_start; wait_done_read(cls1);
        load_window(2); pulse_start; wait_done_read(cls2);
        seq_t1 = $time;
        seq_cycles = (seq_t1 - seq_t0) / 10;
        check_result("SEQ", 0, cls0);
        check_result("SEQ", 1, cls1);
        check_result("SEQ", 2, cls2);
        $display("--- PASS A SEQUENTIAL : %0d cycles for 3 windows ---", seq_cycles);

        // ════════════════════════════════════════════════════════════════
        // PASS B — OVERLAPPED
        //   load(0) → start(0)
        //   while window 0 computes, once isram_free → load(1)
        //   on done(0) → start(1) immediately; meanwhile load(2)
        //   on done(1) → start(2); wait done(2)
        //   Window N+1 is loaded inside window N's compute (Conv2..GAP).
        // ════════════════════════════════════════════════════════════════
        apply_reset;
        ovl_t0 = $time;

        // Window 0: must be fully present before start (nothing to overlap with).
        load_window(0);
        pulse_start;                       // start(0)

        // Wait until input_sram is free (Conv1(0) done), then load window 1.
        guard = 0; read_status(bsy, fre);
        while (!fre && guard < 20000) begin @(posedge clk); #1; read_status(bsy, fre); guard = guard + 1; end
        load_window(1);                    // overlaps Conv2..GAP of window 0

        // Wait for window 0 to finish, capture result, start window 1 at once.
        guard = 0; read_status(bsy, fre);
        while (bsy && guard < 20000) begin @(posedge clk); #1; read_status(bsy, fre); guard = guard + 1; end
        avs_rd(5'h05, status); cls0 = status[1:0];
        pulse_start;                       // start(1) — window 1 already loaded

        // While window 1 computes, wait free then load window 2.
        guard = 0; read_status(bsy, fre);
        while (!fre && guard < 20000) begin @(posedge clk); #1; read_status(bsy, fre); guard = guard + 1; end
        load_window(2);                    // overlaps Conv2..GAP of window 1

        guard = 0; read_status(bsy, fre);
        while (bsy && guard < 20000) begin @(posedge clk); #1; read_status(bsy, fre); guard = guard + 1; end
        avs_rd(5'h05, status); cls1 = status[1:0];
        pulse_start;                       // start(2) — window 2 already loaded

        guard = 0; read_status(bsy, fre);
        while (bsy && guard < 20000) begin @(posedge clk); #1; read_status(bsy, fre); guard = guard + 1; end
        avs_rd(5'h05, status); cls2 = status[1:0];

        ovl_t1 = $time;
        ovl_cycles = (ovl_t1 - ovl_t0) / 10;
        check_result("OVL", 0, cls0);
        check_result("OVL", 1, cls1);
        check_result("OVL", 2, cls2);
        $display("--- PASS B OVERLAPPED : %0d cycles for 3 windows ---", ovl_cycles);

        // ── Summary ────────────────────────────────────────────────────
        $display("");
        $display("=== OVERLAP THROUGHPUT ===");
        $display("  sequential : %0d cycles", seq_cycles);
        $display("  overlapped : %0d cycles", ovl_cycles);
        if (ovl_cycles < seq_cycles)
            $display("  SAVED      : %0d cycles (%0d%%) by overlapping reload with compute",
                     seq_cycles - ovl_cycles,
                     (100 * (seq_cycles - ovl_cycles)) / seq_cycles);
        else
            $display("  NO SAVING  : overlap did not reduce cycles (load > free window?)");

        $display("");
        if (fail_cnt == 0 && ovl_cycles < seq_cycles)
            $display("=== tb_overlap: PASS (%0d checks, overlap saved time, results bit-correct) ===", pass_cnt);
        else if (fail_cnt == 0)
            $display("=== tb_overlap: RESULTS OK but NO throughput gain (%0d checks) ===", pass_cnt);
        else
            $display("=== tb_overlap: FAIL (%0d pass, %0d fail) ===", pass_cnt, fail_cnt);

        $finish;
    end

    // Safety timeout
    initial begin
        #200_000_000;
        $display("=== tb_overlap: TIMEOUT ===");
        $finish;
    end

endmodule
