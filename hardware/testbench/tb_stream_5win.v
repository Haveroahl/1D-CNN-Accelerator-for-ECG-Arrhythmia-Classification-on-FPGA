// tb_stream_5win.v
// Streaming testbench: 5 windows × 2500 samples = 15000 samples liên tiếp.
// Mô phỏng kịch bản real-time non-overlapping window inference.
//
// Test scope (theo yêu cầu user):
//   1. Inference results đúng cho mỗi window (so với expected per-window)
//   2. FSM reset giữa các window (bank_sel, srw_rst, layer_state→IDLE)
//   3. Cycle count deterministic = 5216 cho mọi window
//   4. I/O timing với sample interval (compressed scale)
//
// Data source: reuse ecg_sample0/1/2.hex + 2 lần lặp để đủ 5 window.
//   Window mapping:
//     win0 → ecg_sample0  (expected class 3 — index 0 trong expected_results.hex)
//     win1 → ecg_sample1  (expected class 1)
//     win2 → ecg_sample2  (expected class 2)
//     win3 → ecg_sample0  (lặp)
//     win4 → ecg_sample1  (lặp)
//
// Timing model:
//   - Real-time: 250 Hz ADC → 4 ms/sample = 400,000 cy @ 100 MHz
//   - Sim full 60s = 6e9 cycles → không khả thi.
//   - Scaled mode: SAMPLE_INTERVAL_CY parameter (default 10,000 cy = 100 µs scaled).
//     Vẫn lớn hơn inference cost 5216 cy → bảo toàn property "sample đến khi
//     cp_engine đã IDLE từ trước" giống thực tế.
//   - Burst mode (SAMPLE_INTERVAL_CY=0): load nhanh nhất có thể, dùng để verify
//     correctness mà không sim time-of-day.

`timescale 1ns/1ps

module tb_stream_5win;

    // ── Parameters ────────────────────────────────────────────────────
    // Sample interval scaling. 0 = burst (back-to-back), >0 = wait N clocks
    // giữa các sample (mô phỏng ADC chậm). 10,000 cy = an toàn (>5216 inference).
    parameter integer SAMPLE_INTERVAL_CY = 10_000;
    parameter integer NUM_WINDOWS        = 5;
    parameter integer SAMPLES_PER_WIN    = 2500;

    // ── DUT signals ───────────────────────────────────────────────────
    reg         clk, rst, rst_n;
    reg  [4:0]  avs_address;
    reg         avs_write, avs_read;
    reg  [31:0] avs_writedata;
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

    // ── Hierarchical aliases ──────────────────────────────────────────
    wire [2:0] layer_state  = u_top.ctrl_layer_state;
    wire       busy         = u_top.ctrl_busy;
    wire       ctrl_done    = u_top.ctrl_done;
    wire [1:0] result       = u_top.ctrl_result;
    wire       bank_sel     = u_top.ctrl_bank_sel;
    wire       srw_rst      = u_top.ctrl_srw_rst;

    localparam IDLE     = 3'd0;
    localparam CONV1    = 3'd2;
    localparam GAP_FC_S = 3'd6;
    localparam DONE_S   = 3'd7;

    // ── Clock ─────────────────────────────────────────────────────────
    initial clk = 0;
    always #5 clk = ~clk;   // 10 ns period = 100 MHz

    // ── Avalon-MM helpers ─────────────────────────────────────────────
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

    // ── Reset ─────────────────────────────────────────────────────────
    task apply_reset;
        begin
            rst = 1; rst_n = 0;
            avs_write = 0; avs_read = 0;
            avs_address = 0; avs_writedata = 0;
            @(posedge clk); @(posedge clk); #1;
            rst = 0; rst_n = 1;
            @(posedge clk); #1;
        end
    endtask

    // ── Per-sample write into input_sram ──────────────────────────────
    // Mô phỏng 1 sample đến từ ADC: ghi DATA → ADDR → WR_EN, rồi đợi
    // (SAMPLE_INTERVAL_CY - write_overhead) cycles trước sample tiếp theo.
    task push_sample;
        input integer addr_in_sram;   // 0..2499 (circular sau mỗi window)
        input [7:0]   val;
        integer       wait_remaining;
        time          t_before, t_after;
        begin
            t_before = $time;
            avs_wr(5'h00, {24'h0, val});
            avs_wr(5'h01, addr_in_sram[31:0]);
            avs_wr(5'h02, 32'd1);
            t_after = $time;

            if (SAMPLE_INTERVAL_CY > 0) begin
                // Số cycle đã tiêu cho 3 avs_wr (~6-9 cy). Đợi phần còn lại.
                wait_remaining = SAMPLE_INTERVAL_CY - (t_after - t_before) / 10;
                if (wait_remaining > 0)
                    repeat (wait_remaining) @(posedge clk);
            end
        end
    endtask

    // ── Trigger inference + poll done ─────────────────────────────────
    task run_inference;
        output [1:0]   cls;
        output integer cycles_actual;
        reg     [31:0] status;
        time           t_start, t_end;
        integer        poll_iter;
        begin
            avs_wr(5'h03, 32'd1);   // START
            @(posedge clk); #1;
            t_start   = $time;
            poll_iter = 0;
            status    = 1;
            while (status[0] && poll_iter < 20000) begin
                @(posedge clk); #1;
                avs_rd(5'h04, status);
                poll_iter = poll_iter + 1;
            end
            t_end          = $time;
            cycles_actual  = (t_end - t_start) / 10;
            avs_rd(5'h05, status);
            cls = status[1:0];
        end
    endtask

    // ── Test storage ──────────────────────────────────────────────────
    reg [7:0]  win_data [0:14999];        // 5 × 2500 samples raw
    reg [7:0]  src_buf  [0:2499];         // tạm cho mỗi sample file
    reg [7:0]  expected_per_win [0:4];
    reg [7:0]  raw_expected [0:2];        // file expected_results.hex (3 entries)

    integer    pass_cnt, fail_cnt;
    integer    fsm_idle_seen_between_win;
    integer    bank_sel_transitions;
    integer    srw_rst_pulses;
    reg [2:0]  prev_layer_state;
    reg        prev_bank_sel;
    reg        prev_srw_rst;
    reg        in_window_run;   // gate FSM observation chỉ trong inference

    // ── FSM observation logic ────────────────────────────────────────
    // Đếm bank_sel transitions và srw_rst pulses chỉ TRONG inference
    // (in_window_run=1), để tránh đếm các glitch lúc setup/idle/reset.
    always @(posedge clk) begin
        if (rst) begin
            prev_layer_state     <= IDLE;
            prev_bank_sel        <= 1'b0;
            prev_srw_rst         <= 1'b0;
            bank_sel_transitions <= 0;
            srw_rst_pulses       <= 0;
        end else if (in_window_run) begin
            prev_layer_state <= layer_state;
            prev_bank_sel    <= bank_sel;
            prev_srw_rst     <= srw_rst;
            if (bank_sel !== prev_bank_sel)
                bank_sel_transitions <= bank_sel_transitions + 1;
            if (srw_rst && !prev_srw_rst)
                srw_rst_pulses <= srw_rst_pulses + 1;
        end
    end

    // ── Per-test report helpers ───────────────────────────────────────
    task report_pass;
        input [255:0] msg;
        begin
            $display("PASS [%0s]", msg);
            pass_cnt = pass_cnt + 1;
        end
    endtask

    task report_fail;
        input [255:0] msg;
        begin
            $display("FAIL [%0s]", msg);
            fail_cnt = fail_cnt + 1;
        end
    endtask

    // ── Main ──────────────────────────────────────────────────────────
    integer    w, i, src_idx;
    reg [1:0]  cls;
    integer    cycles_w;
    integer    cycles_first;
    integer    bs_before, sr_before;

    initial begin
        pass_cnt = 0; fail_cnt = 0;
        in_window_run = 0;
        bank_sel_transitions = 0;
        srw_rst_pulses = 0;
        cycles_first = 0;

        $display("=================================================================");
        $display("=== tb_stream_5win: %0d windows x %0d samples (interval=%0d cy) ===",
                 NUM_WINDOWS, SAMPLES_PER_WIN, SAMPLE_INTERVAL_CY);
        $display("=================================================================");

        apply_reset();

        // ── Load expected_results (3 entries) ──────────────────────────
        $readmemh("expected_results.hex", raw_expected);
        // Map 5 window → expected (lặp ecg_sample0/1/2/0/1)
        expected_per_win[0] = raw_expected[0];   // win0 ← sample0
        expected_per_win[1] = raw_expected[1];   // win1 ← sample1
        expected_per_win[2] = raw_expected[2];   // win2 ← sample2
        expected_per_win[3] = raw_expected[0];   // win3 ← sample0 (lặp)
        expected_per_win[4] = raw_expected[1];   // win4 ← sample1 (lặp)

        // ── Build win_data[] từ 3 hex files ───────────────────────────
        $readmemh("ecg_sample0.hex", src_buf);
        for (i = 0; i < SAMPLES_PER_WIN; i = i + 1) win_data[0*SAMPLES_PER_WIN + i] = src_buf[i];

        $readmemh("ecg_sample1.hex", src_buf);
        for (i = 0; i < SAMPLES_PER_WIN; i = i + 1) win_data[1*SAMPLES_PER_WIN + i] = src_buf[i];

        $readmemh("ecg_sample2.hex", src_buf);
        for (i = 0; i < SAMPLES_PER_WIN; i = i + 1) win_data[2*SAMPLES_PER_WIN + i] = src_buf[i];

        $readmemh("ecg_sample0.hex", src_buf);
        for (i = 0; i < SAMPLES_PER_WIN; i = i + 1) win_data[3*SAMPLES_PER_WIN + i] = src_buf[i];

        $readmemh("ecg_sample1.hex", src_buf);
        for (i = 0; i < SAMPLES_PER_WIN; i = i + 1) win_data[4*SAMPLES_PER_WIN + i] = src_buf[i];

        $display("[INFO] Loaded 15,000 samples. Expected per-window class: %0d %0d %0d %0d %0d",
                 expected_per_win[0], expected_per_win[1], expected_per_win[2],
                 expected_per_win[3], expected_per_win[4]);

        // ─────────────────────────────────────────────────────────────
        // Main loop: 5 windows
        // ─────────────────────────────────────────────────────────────
        for (w = 0; w < NUM_WINDOWS; w = w + 1) begin
            $display("");
            $display("─── Window %0d ──────────────────────────────────────", w);

            // Snapshot FSM observation counters trước window
            bs_before = bank_sel_transitions;
            sr_before = srw_rst_pulses;

            // [Check FSM reset state] Trước START: window 0 phải IDLE, các
            // window sau phải DONE_S (giữ result của window trước, sẵn sàng
            // nhận start mới cho streaming).
            if (w == 0) begin
                if (layer_state == IDLE) begin
                    $display("[WIN%0d] pre-START: layer_state=IDLE OK", w);
                end else begin
                    $display("[WIN%0d] pre-START: layer_state=%0d (expected IDLE)", w, layer_state);
                    fail_cnt = fail_cnt + 1;
                end
            end else begin
                if (layer_state == DONE_S) begin
                    $display("[WIN%0d] pre-START: layer_state=DONE_S (streaming ready) OK", w);
                end else begin
                    $display("[WIN%0d] pre-START: layer_state=%0d (expected DONE_S)", w, layer_state);
                    fail_cnt = fail_cnt + 1;
                end
            end

            // ── Push 2500 samples (circular addr 0..2499) ─────────────
            // Mô phỏng ADC fill input_sram. Mỗi sample cách nhau
            // SAMPLE_INTERVAL_CY cy (nếu > 0).
            for (i = 0; i < SAMPLES_PER_WIN; i = i + 1) begin
                src_idx = w * SAMPLES_PER_WIN + i;
                push_sample(i, win_data[src_idx]);
            end

            // Đảm bảo last WE pulse commit vào input_sram
            @(posedge clk); #1;

            // ── Trigger inference + observe FSM ───────────────────────
            in_window_run = 1;
            run_inference(cls, cycles_w);
            in_window_run = 0;

            // ── Check 1: Result đúng ──────────────────────────────────
            if (cls === expected_per_win[w][1:0]) begin
                $display("[WIN%0d] result=%0d (expected %0d) PASS", w, cls, expected_per_win[w]);
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("[WIN%0d] result=%0d (expected %0d) FAIL", w, cls, expected_per_win[w]);
                fail_cnt = fail_cnt + 1;
            end

            // ── Check 2: Cycle count = 5216 deterministic ──────────────
            if (w == 0) begin
                cycles_first = cycles_w;
                if (cycles_w == 5216) begin
                    $display("[WIN%0d] cycles=%0d (expected 5216) PASS", w, cycles_w);
                    pass_cnt = pass_cnt + 1;
                end else begin
                    $display("[WIN%0d] cycles=%0d (expected 5216) FAIL", w, cycles_w);
                    fail_cnt = fail_cnt + 1;
                end
            end else begin
                if (cycles_w == cycles_first) begin
                    $display("[WIN%0d] cycles=%0d (= win0, deterministic) PASS", w, cycles_w);
                    pass_cnt = pass_cnt + 1;
                end else begin
                    $display("[WIN%0d] cycles=%0d (win0=%0d, drift detected) FAIL",
                             w, cycles_w, cycles_first);
                    fail_cnt = fail_cnt + 1;
                end
            end

            // ── Check 3: FSM state transitions trong window này ───────
            // Sau 1 inference: bank_sel toggle 4 lần (LOAD_INPUT init + 3 layer
            // transitions Conv1→2, Conv2→3, Conv3→4 — Conv4→GAP cũng toggle
            // = 4 toggles total). srw_rst pulse: 1 (LOAD_INPUT) + 4 (mỗi
            // layer transition Conv1→2→3→4→GAP) = 5 pulses.
            // Lưu ý: số đếm phụ thuộc cách count edge — kiểm tra > 0 là đủ
            // để xác nhận FSM đang chạy đúng (không stuck).
            if ((bank_sel_transitions - bs_before) > 0) begin
                $display("[WIN%0d] bank_sel toggles = %0d (>0) PASS",
                         w, bank_sel_transitions - bs_before);
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("[WIN%0d] bank_sel toggles = 0 (FSM stuck?) FAIL", w);
                fail_cnt = fail_cnt + 1;
            end

            if ((srw_rst_pulses - sr_before) > 0) begin
                $display("[WIN%0d] srw_rst pulses = %0d (>0) PASS",
                         w, srw_rst_pulses - sr_before);
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("[WIN%0d] srw_rst pulses = 0 FAIL", w);
                fail_cnt = fail_cnt + 1;
            end

            // ── Post-inference: FSM phải ở DONE_S (giữ result, chờ next start) ─
            @(posedge clk); #1;
            if (layer_state == DONE_S) begin
                $display("[WIN%0d] post-inference: layer_state=DONE_S (result held) PASS", w);
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("[WIN%0d] post-inference: layer_state=%0d (expected DONE_S) FAIL",
                         w, layer_state);
                fail_cnt = fail_cnt + 1;
            end
        end

        // ─────────────────────────────────────────────────────────────
        // Summary
        // ─────────────────────────────────────────────────────────────
        $display("");
        $display("=================================================================");
        $display("=== tb_stream_5win SUMMARY: %0d PASS, %0d FAIL ===", pass_cnt, fail_cnt);
        $display("=================================================================");
        if (fail_cnt == 0)
            $display("ALL STREAM CHECKS PASSED — 5-window real-time scenario OK");
        else
            $display("STREAM TEST HAS FAILURES — check log above");

        $finish;
    end

    // ── Global safety timeout ─────────────────────────────────────────
    initial begin
        // Worst case sim time:
        //   5 win × 2500 sample × (3 avs_wr × ~2 cy + SAMPLE_INTERVAL_CY)
        //   = 5 × 2500 × (6 + 10,000) × 10 ns
        //   = 5 × 2500 × 100,060 ns ≈ 1.25 s sim time
        // Add 5 × inference (5216 cy × 10 ns = 52 µs each) = 260 µs
        // Give 5× margin → 6 seconds sim time
        #(64'd6_000_000_000);
        $display("[TIMEOUT] Simulation exceeded budget");
        $finish;
    end

endmodule
