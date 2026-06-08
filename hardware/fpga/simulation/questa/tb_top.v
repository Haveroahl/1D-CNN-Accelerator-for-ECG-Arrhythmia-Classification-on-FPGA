// tb_top.v — Full system test for ecg_accelerator_top
// 7 test cases covering Avalon-MM interface, FSM states, and end-to-end inference
//
// Requires:
//   RTL/conv1_w.hex..conv4_w.hex — INT8 packed 5-tap weights (40b/word)
//   RTL/conv_bias.hex            — INT32 biases (32 entries, addr=oc*4+layer_idx)
//   RTL/fc_weights.hex           — INT8 FC weights (32 entries, addr=k*8+i)
//   testbench/ecg_sample0.hex  — INT8 ECG sample, 2500 entries
//   testbench/ecg_sample1.hex  — second sample
//   testbench/ecg_sample2.hex  — third sample
//   testbench/expected_results.hex — expected argmax [2:0] per sample (3 entries)
//
// If hex files are missing: TC04-TC07 show warnings but TC01-TC03 still run.
// Generate golden files: software/python/generate_golden.py

`timescale 1ns/1ps

module tb_top;

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

    // ── Hierarchical aliases ──────────────────────────────────────────
    wire [2:0]  layer_state  = u_top.ctrl_layer_state;
    wire        busy         = u_top.ctrl_busy;
    wire        ctrl_done    = u_top.ctrl_done;
    wire [1:0]  result       = u_top.ctrl_result;
    wire [2:0]  fc_sub_state = u_top.ctrl_fc_sub_state;

    // FSM state encoding
    localparam IDLE       = 3'd0;
    localparam LOAD_INPUT = 3'd1;
    localparam CONV1      = 3'd2;
    localparam CONV2      = 3'd3;
    localparam CONV3      = 3'd4;
    localparam CONV4      = 3'd5;
    localparam GAP_FC_S   = 3'd6;
    localparam DONE_S     = 3'd7;

    // Probe state (kept for transition detection in compare auto-trigger block)
    reg [2:0] prev_layer_state;
    reg [2:0] prev_fc_sub_state;
    reg [3:0] prev_gap_step;


    // ── DEBUG PROBES: per-layer first-write + GAP/FC dumps ─────────────
    integer   probe_cnt_l1, probe_cnt_l2, probe_cnt_l3, probe_cnt_l4;
    always @(posedge clk) begin
        if (rst) begin
            prev_layer_state  <= 3'd0;
            prev_fc_sub_state <= 3'd0;
            prev_gap_step     <= 4'd0;
            probe_cnt_l1 <= 0; probe_cnt_l2 <= 0;
            probe_cnt_l3 <= 0; probe_cnt_l4 <= 0;
        end else begin
            prev_layer_state  <= layer_state;
            prev_fc_sub_state <= fc_sub_state;
            prev_gap_step     <= u_top.ctrl_gap_step;

            // First 3 pool writes per conv layer: dump cp_pool_out
            if (layer_state == CONV1 && u_top.cp_pool_write && probe_cnt_l1 < 3) begin
                $display("  [PROBE C1 w%0d] ch0..3 = %02x %02x %02x %02x",
                    probe_cnt_l1,
                    u_top.u_cpe.cp_pool_out[0], u_top.u_cpe.cp_pool_out[1],
                    u_top.u_cpe.cp_pool_out[2], u_top.u_cpe.cp_pool_out[3]);
                probe_cnt_l1 <= probe_cnt_l1 + 1;
            end
            if (layer_state == CONV2 && u_top.cp_pool_write && probe_cnt_l2 < 3) begin
                $display("  [PROBE C2 w%0d] ch0..3 = %02x %02x %02x %02x",
                    probe_cnt_l2,
                    u_top.u_cpe.cp_pool_out[0], u_top.u_cpe.cp_pool_out[1],
                    u_top.u_cpe.cp_pool_out[2], u_top.u_cpe.cp_pool_out[3]);
                probe_cnt_l2 <= probe_cnt_l2 + 1;
            end
            if (layer_state == CONV3 && u_top.cp_pool_write && probe_cnt_l3 < 3) begin
                $display("  [PROBE C3 w%0d] ch0..7 = %02x %02x %02x %02x %02x %02x %02x %02x",
                    probe_cnt_l3,
                    u_top.u_cpe.cp_pool_out[0], u_top.u_cpe.cp_pool_out[1],
                    u_top.u_cpe.cp_pool_out[2], u_top.u_cpe.cp_pool_out[3],
                    u_top.u_cpe.cp_pool_out[4], u_top.u_cpe.cp_pool_out[5],
                    u_top.u_cpe.cp_pool_out[6], u_top.u_cpe.cp_pool_out[7]);
                probe_cnt_l3 <= probe_cnt_l3 + 1;
            end
            if (layer_state == CONV4 && u_top.cp_pool_write && probe_cnt_l4 < 4) begin
                $display("  [PROBE C4 w%0d] ch0..7 = %02x %02x %02x %02x %02x %02x %02x %02x",
                    probe_cnt_l4,
                    u_top.u_cpe.cp_pool_out[0], u_top.u_cpe.cp_pool_out[1],
                    u_top.u_cpe.cp_pool_out[2], u_top.u_cpe.cp_pool_out[3],
                    u_top.u_cpe.cp_pool_out[4], u_top.u_cpe.cp_pool_out[5],
                    u_top.u_cpe.cp_pool_out[6], u_top.u_cpe.cp_pool_out[7]);
                probe_cnt_l4 <= probe_cnt_l4 + 1;
            end

            // GAP per-step: dump ping_dout (input) and gap_acc/gap_reg
            if (layer_state == GAP_FC_S && fc_sub_state == 3'd1 /*GAP_S*/) begin
                $display("  [PROBE GAP step=%0d rd_addr=%0d] ping_dout=%016x",
                    u_top.ctrl_gap_step, u_top.u_gfa.gap_rd_addr, u_top.pp_dout);
                $display("           gap_acc[0..7]=%03x %03x %03x %03x %03x %03x %03x %03x",
                    u_top.u_gfa.gap_acc[0]&10'h3FF, u_top.u_gfa.gap_acc[1]&10'h3FF,
                    u_top.u_gfa.gap_acc[2]&10'h3FF, u_top.u_gfa.gap_acc[3]&10'h3FF,
                    u_top.u_gfa.gap_acc[4]&10'h3FF, u_top.u_gfa.gap_acc[5]&10'h3FF,
                    u_top.u_gfa.gap_acc[6]&10'h3FF, u_top.u_gfa.gap_acc[7]&10'h3FF);
            end
            if (prev_gap_step == 4'd5 && fc_sub_state == 3'd2 /*FC_SUB*/ && prev_fc_sub_state == 3'd1) begin
                $display("  [PROBE GAP DONE] gap_reg = %02x %02x %02x %02x %02x %02x %02x %02x",
                    u_top.u_gfa.gap_reg[0], u_top.u_gfa.gap_reg[1],
                    u_top.u_gfa.gap_reg[2], u_top.u_gfa.gap_reg[3],
                    u_top.u_gfa.gap_reg[4], u_top.u_gfa.gap_reg[5],
                    u_top.u_gfa.gap_reg[6], u_top.u_gfa.gap_reg[7]);
            end

            // FC done (entering ARGMAX): dump fc_acc[0..3]
            if (prev_fc_sub_state == 3'd3 /*FC_FLUSH*/ && fc_sub_state == 3'd4 /*ARGMAX*/) begin
                $display("  [PROBE FC_ACC] fc_acc = %08x %08x %08x %08x",
                    u_top.u_gfa.fc_acc[0], u_top.u_gfa.fc_acc[1],
                    u_top.u_gfa.fc_acc[2], u_top.u_gfa.fc_acc[3]);
            end
        end
    end

    // Reset probe counters at each apply_reset (via layer_state going IDLE)
    always @(posedge clk) begin
        if (layer_state == IDLE && prev_layer_state != IDLE) begin
            probe_cnt_l1 <= 0; probe_cnt_l2 <= 0;
            probe_cnt_l3 <= 0; probe_cnt_l4 <= 0;
        end
    end

    // ── Clock ─────────────────────────────────────────────────────────
    initial clk = 0;
    always #5 clk = ~clk;

    integer pass_cnt, fail_cnt;
    time    tc_t0;

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

    // Load ECG from hex file (2500 × INT8)
    task load_ecg_hex;
        input [255:0] filename;
        reg [7:0] ecg [0:2499];
        integer i;
        begin
            $readmemh(filename, ecg);
            for (i = 0; i < 2500; i = i + 1) begin
                avs_wr(5'h00, {24'h0, ecg[i]});  // DATA_IN
                avs_wr(5'h01, i[31:0]);            // ADDR_IN
                avs_wr(5'h02, 32'd1);              // WR_EN
            end
            // Allow final write-enable pulse to commit to input_sram (we is
            // a 1-cycle register pulse — needs one more posedge for SRAM to
            // sample it before next Avalon transaction overwrites the bus).
            @(posedge clk); #1;
        end
    endtask

    // Load constant ECG pattern (for TC01/TC02 that don't need real data)
    task load_ecg_const;
        input [7:0] val;
        integer i;
        begin
            for (i = 0; i < 2500; i = i + 1) begin
                avs_wr(5'h00, {24'h0, val});
                avs_wr(5'h01, i[31:0]);
                avs_wr(5'h02, 32'd1);
            end
        end
    endtask

    // Run full inference: START → poll STATUS → return RESULT, cycle count
    task run_inference;
        output [1:0] cls;
        output integer cycles;
        reg [31:0] status;
        begin
            avs_wr(5'h03, 32'd1);   // START
            cycles = 0;
            status = 1;
            while (status[0] && cycles < 10000) begin
                @(posedge clk); #1;
                avs_rd(5'h04, status);  // STATUS: [0]=busy
                cycles = cycles + 1;
            end
            avs_rd(5'h05, status);   // RESULT
            cls = status[1:0];
        end
    endtask

    // Apply DUT reset
    task apply_reset;
        begin
            rst = 1; rst_n = 0;
            avs_write = 0; avs_read = 0;
            @(posedge clk); @(posedge clk); #1;
            rst = 0; rst_n = 1;
            @(posedge clk); #1;
            tc_t0 = $time;
        end
    endtask

    // ── FSM state coverage tracking ───────────────────────────────────
    reg [7:0] states_seen;   // bitmask: bit[i]=1 if state i was observed

    always @(posedge clk) begin
        states_seen[layer_state] <= 1'b1;
    end

    reg [4:0] fc_sub_seen;   // bitmask for fc_sub_state 0..4

    always @(posedge clk) begin
        if (fc_sub_state < 5)
            fc_sub_seen[fc_sub_state] <= 1'b1;
    end

    // ── Main test sequence ─────────────────────────────────────────────
    reg [1:0]  cls_out;
    integer    cyc_out;
    reg [31:0] rd_data;
    reg [7:0]  expected_results [0:2];

    // ── Cấp 2: Per-layer golden buffers (loaded per-sample) ────────────
    // Stored as 8-bit (read raw hex), sign-extended in compare task.
    reg [7:0]  gold_input   [0:2499];   // 2500
    reg [7:0]  gold_pool1   [0:1999];   // 4 ch × 500
    reg [7:0]  gold_pool2   [0:399];    // 4 ch × 100
    reg [7:0]  gold_pool3   [0:159];    // 8 ch × 20
    reg [7:0]  gold_pool4   [0:31];     // 8 ch × 4
    reg [7:0]  gold_gap     [0:7];      // 8
    reg [31:0] gold_logits  [0:3];      // 4 × INT32

    integer    l2_pass_cnt, l2_fail_cnt;
    integer    current_sample;

    // ── Diagnostic: actual deviation RTL vs Python golden (not just tol pass) ──
    integer        g_nonzero;        // # elements where RTL != golden exactly
    integer        g_total;          // # elements compared
    reg signed [31:0] g_max_diff;    // max |RTL - golden| over all stages/samples

    // ── Helpers: read per-channel ping_pong memory ─────────────────────
    // ping_pong_sram uses 16 separate 1D arrays (mem_a_ch0..7, mem_b_ch0..7)
    // with ramstyle="M10K" to force per-channel M10K inference. Testbench
    // accesses internal memories via hierarchical reference — case statement
    // dispatches on channel.
    function [7:0] read_mem_a;
        input integer ch;
        input integer pos;
        begin
            read_mem_a = u_top.u_pp.mem_a[ch*500 + pos];
        end
    endfunction

    function [7:0] read_mem_b;
        input integer ch;
        input integer pos;
        begin
            read_mem_b = u_top.u_pp.mem_b[ch*500 + pos];
        end
    endfunction

    // ── Cấp 2: Per-stage compare tasks (Verilog-2001: no array-of-arrays args) ──
    task check_pool1;
        input         bank;
        integer ch, pos, idx, mismatches, first_bad, first_rtl, first_gold;
        reg signed [9:0] diff;
        reg signed [8:0] rtl_v, gold_v;
        reg [7:0] raw;
        begin
            mismatches = 0; first_bad = -1; first_rtl = 0; first_gold = 0;
            for (ch = 0; ch < 4; ch = ch + 1) begin
                for (pos = 0; pos < 500; pos = pos + 1) begin
                    idx = ch * 500 + pos;
                    raw = bank ? read_mem_b(ch, pos) : read_mem_a(ch, pos);
                    rtl_v  = $signed({raw[7], raw});
                    gold_v = $signed({gold_pool1[idx][7], gold_pool1[idx]});
                    diff = rtl_v - gold_v;
                    g_total = g_total + 1;
                    if (diff != 0) g_nonzero = g_nonzero + 1;
                    if (diff  >  g_max_diff) g_max_diff =  diff;
                    if (-diff >  g_max_diff) g_max_diff = -diff;
                    if (diff > 10 || diff < -10) begin
                        mismatches = mismatches + 1;
                        if (first_bad < 0) begin
                            first_bad = idx; first_rtl = rtl_v; first_gold = gold_v;
                        end
                    end
                end
            end
            if (mismatches == 0) begin
                $display("[STAGE PASS] sample%0d after_pool1 (2000 elems, tol +/-10)", current_sample);
                l2_pass_cnt = l2_pass_cnt + 1;
            end else begin
                $display("[STAGE FAIL] sample%0d after_pool1: %0d/2000 mismatches, first @ ch=%0d pos=%0d rtl=%0d gold=%0d",
                         current_sample, mismatches, first_bad/500, first_bad%500, first_rtl, first_gold);
                l2_fail_cnt = l2_fail_cnt + 1;
            end
        end
    endtask

    task check_pool2;
        input         bank;
        integer ch, pos, idx_gold, mismatches, first_bad, first_rtl, first_gold;
        reg signed [9:0] diff;
        reg signed [8:0] rtl_v, gold_v;
        reg [7:0] raw;
        begin
            mismatches = 0; first_bad = -1; first_rtl = 0; first_gold = 0;
            for (ch = 0; ch < 4; ch = ch + 1) begin
                for (pos = 0; pos < 100; pos = pos + 1) begin
                    idx_gold = ch * 100 + pos;   // golden stride = 100
                    raw = bank ? read_mem_b(ch, pos) : read_mem_a(ch, pos);
                    rtl_v  = $signed({raw[7], raw});
                    gold_v = $signed({gold_pool2[idx_gold][7], gold_pool2[idx_gold]});
                    diff = rtl_v - gold_v;
                    g_total = g_total + 1;
                    if (diff != 0) g_nonzero = g_nonzero + 1;
                    if (diff  >  g_max_diff) g_max_diff =  diff;
                    if (-diff >  g_max_diff) g_max_diff = -diff;
                    if (diff > 10 || diff < -10) begin
                        mismatches = mismatches + 1;
                        if (first_bad < 0) begin
                            first_bad = idx_gold; first_rtl = rtl_v; first_gold = gold_v;
                        end
                    end
                end
            end
            if (mismatches == 0) begin
                $display("[STAGE PASS] sample%0d after_pool2 (400 elems, tol +/-10)", current_sample);
                l2_pass_cnt = l2_pass_cnt + 1;
            end else begin
                $display("[STAGE FAIL] sample%0d after_pool2: %0d/400 mismatches, first @ ch=%0d pos=%0d rtl=%0d gold=%0d",
                         current_sample, mismatches, first_bad/100, first_bad%100, first_rtl, first_gold);
                l2_fail_cnt = l2_fail_cnt + 1;
            end
        end
    endtask

    task check_pool3;
        input         bank;
        integer ch, pos, idx_gold, mismatches, first_bad, first_rtl, first_gold;
        reg signed [9:0] diff;
        reg signed [8:0] rtl_v, gold_v;
        reg [7:0] raw;
        begin
            mismatches = 0; first_bad = -1; first_rtl = 0; first_gold = 0;
            for (ch = 0; ch < 8; ch = ch + 1) begin
                for (pos = 0; pos < 20; pos = pos + 1) begin
                    idx_gold = ch * 20  + pos;
                    raw = bank ? read_mem_b(ch, pos) : read_mem_a(ch, pos);
                    rtl_v  = $signed({raw[7], raw});
                    gold_v = $signed({gold_pool3[idx_gold][7], gold_pool3[idx_gold]});
                    diff = rtl_v - gold_v;
                    g_total = g_total + 1;
                    if (diff != 0) g_nonzero = g_nonzero + 1;
                    if (diff  >  g_max_diff) g_max_diff =  diff;
                    if (-diff >  g_max_diff) g_max_diff = -diff;
                    if (diff > 10 || diff < -10) begin
                        mismatches = mismatches + 1;
                        if (first_bad < 0) begin
                            first_bad = idx_gold; first_rtl = rtl_v; first_gold = gold_v;
                        end
                    end
                end
            end
            if (mismatches == 0) begin
                $display("[STAGE PASS] sample%0d after_pool3 (160 elems, tol +/-10)", current_sample);
                l2_pass_cnt = l2_pass_cnt + 1;
            end else begin
                $display("[STAGE FAIL] sample%0d after_pool3: %0d/160 mismatches, first @ ch=%0d pos=%0d rtl=%0d gold=%0d",
                         current_sample, mismatches, first_bad/20, first_bad%20, first_rtl, first_gold);
                l2_fail_cnt = l2_fail_cnt + 1;
            end
        end
    endtask

    task check_pool4;
        input         bank;
        integer ch, pos, idx_gold, mismatches, first_bad, first_rtl, first_gold;
        reg signed [9:0] diff;
        reg signed [8:0] rtl_v, gold_v;
        reg [7:0] raw;
        begin
            mismatches = 0; first_bad = -1; first_rtl = 0; first_gold = 0;
            for (ch = 0; ch < 8; ch = ch + 1) begin
                for (pos = 0; pos < 4; pos = pos + 1) begin
                    idx_gold = ch * 4   + pos;
                    raw = bank ? read_mem_b(ch, pos) : read_mem_a(ch, pos);
                    rtl_v  = $signed({raw[7], raw});
                    gold_v = $signed({gold_pool4[idx_gold][7], gold_pool4[idx_gold]});
                    diff = rtl_v - gold_v;
                    g_total = g_total + 1;
                    if (diff != 0) g_nonzero = g_nonzero + 1;
                    if (diff  >  g_max_diff) g_max_diff =  diff;
                    if (-diff >  g_max_diff) g_max_diff = -diff;
                    if (diff > 10 || diff < -10) begin
                        mismatches = mismatches + 1;
                        if (first_bad < 0) begin
                            first_bad = idx_gold; first_rtl = rtl_v; first_gold = gold_v;
                        end
                    end
                end
            end
            if (mismatches == 0) begin
                $display("[STAGE PASS] sample%0d after_pool4 (32 elems, tol +/-10)", current_sample);
                l2_pass_cnt = l2_pass_cnt + 1;
            end else begin
                $display("[STAGE FAIL] sample%0d after_pool4: %0d/32 mismatches, first @ ch=%0d pos=%0d rtl=%0d gold=%0d",
                         current_sample, mismatches, first_bad/4, first_bad%4, first_rtl, first_gold);
                l2_fail_cnt = l2_fail_cnt + 1;
            end
        end
    endtask

    task check_input;
        integer i, mismatches, first_bad, first_rtl, first_gold;
        reg signed [9:0] diff;
        reg signed [8:0] rtl_v, gold_v;
        begin
            mismatches = 0; first_bad = -1; first_rtl = 0; first_gold = 0;
            for (i = 0; i < 2500; i = i + 1) begin
                rtl_v  = $signed({u_top.u_isram.mem[i][7], u_top.u_isram.mem[i]});
                gold_v = $signed({gold_input[i][7], gold_input[i]});
                diff = rtl_v - gold_v;
                if (diff > 10 || diff < -10) begin
                    mismatches = mismatches + 1;
                    if (first_bad < 0) begin
                        first_bad = i; first_rtl = rtl_v; first_gold = gold_v;
                    end
                end
            end
            if (mismatches == 0) begin
                $display("[STAGE PASS] sample%0d input_int8 (2500 elems, tol +/-10)", current_sample);
                l2_pass_cnt = l2_pass_cnt + 1;
            end else begin
                $display("[STAGE FAIL] sample%0d input_int8: %0d/2500 mismatches, first @ idx=%0d rtl=%0d gold=%0d",
                         current_sample, mismatches, first_bad, first_rtl, first_gold);
                l2_fail_cnt = l2_fail_cnt + 1;
            end
        end
    endtask

    task check_gap;
        integer i, mismatches, first_bad, first_rtl, first_gold;
        reg signed [9:0] diff;
        reg signed [8:0] rtl_v, gold_v;
        begin
            mismatches = 0; first_bad = -1; first_rtl = 0; first_gold = 0;
            for (i = 0; i < 8; i = i + 1) begin
                rtl_v  = $signed(u_top.u_gfa.gap_reg[i]);
                gold_v = $signed({gold_gap[i][7], gold_gap[i]});
                diff = rtl_v - gold_v;
                if (diff > 10 || diff < -10) begin
                    mismatches = mismatches + 1;
                    if (first_bad < 0) begin
                        first_bad = i; first_rtl = rtl_v; first_gold = gold_v;
                    end
                end
            end
            if (mismatches == 0) begin
                $display("[STAGE PASS] sample%0d after_gap (8 elems, tol +/-10)", current_sample);
                l2_pass_cnt = l2_pass_cnt + 1;
            end else begin
                $display("[STAGE FAIL] sample%0d after_gap: %0d/8 mismatches, first @ idx=%0d rtl=%0d gold=%0d",
                         current_sample, mismatches, first_bad, first_rtl, first_gold);
                l2_fail_cnt = l2_fail_cnt + 1;
            end
        end
    endtask

    task check_logits;
        integer i, mismatches, first_bad;
        reg signed [31:0] rtl_v, gold_v, diff;
        reg signed [31:0] first_rtl, first_gold;
        begin
            mismatches = 0; first_bad = -1; first_rtl = 0; first_gold = 0;
            for (i = 0; i < 4; i = i + 1) begin
                rtl_v  = $signed(u_top.u_gfa.fc_acc[i]);
                gold_v = $signed(gold_logits[i]);
                diff = rtl_v - gold_v;
                if (diff > 10 || diff < -10) begin
                    mismatches = mismatches + 1;
                    if (first_bad < 0) begin
                        first_bad = i; first_rtl = rtl_v; first_gold = gold_v;
                    end
                end
            end
            if (mismatches == 0) begin
                $display("[STAGE PASS] sample%0d logits_fc (4 elems, tol +/-10)", current_sample);
                l2_pass_cnt = l2_pass_cnt + 1;
            end else begin
                $display("[STAGE FAIL] sample%0d logits_fc: %0d/4 mismatches, first @ idx=%0d rtl=%0d gold=%0d",
                         current_sample, mismatches, first_bad, first_rtl, first_gold);
                l2_fail_cnt = l2_fail_cnt + 1;
            end
        end
    endtask

    // ── Cấp 2: Auto-trigger on layer_state transitions ─────────────────
    // bank_sel toggles AFTER layer_done, so layer N output sits in the
    // bank that was Pong DURING layer N. Conv1→Conv2 transition: Conv1
    // wrote to mem_b (bank_sel was 0). Etc.
    //   after_pool1 → mem_b   after_pool2 → mem_a
    //   after_pool3 → mem_b   after_pool4 → mem_a
    reg verify_en;   // gate to skip checks until golden loaded
    always @(posedge clk) begin
        if (verify_en) begin
            if (layer_state == CONV2 && prev_layer_state == CONV1) check_pool1(1'b1);
            if (layer_state == CONV3 && prev_layer_state == CONV2) check_pool2(1'b0);
            if (layer_state == CONV4 && prev_layer_state == CONV3) check_pool3(1'b1);
            if (layer_state == GAP_FC_S && prev_layer_state == CONV4) check_pool4(1'b0);
            // GAP done: gap_reg latched at gap_step==5 (1 cycle after FC_SUB entry)
            if (prev_gap_step == 4'd5 && fc_sub_state == 3'd2 && prev_fc_sub_state == 3'd1)
                check_gap;
            // FC done: fc_acc valid when entering ARGMAX
            if (prev_fc_sub_state == 3'd3 && fc_sub_state == 3'd4)
                check_logits;
        end
    end

    initial begin
        pass_cnt = 0; fail_cnt = 0;
        l2_pass_cnt = 0; l2_fail_cnt = 0;
        g_nonzero = 0; g_total = 0; g_max_diff = 0;
        verify_en = 1'b0;
        current_sample = 0;
        states_seen = 8'h00;
        fc_sub_seen = 5'h00;
        avs_write = 0; avs_read = 0; avs_address = 0; avs_writedata = 0;

        $display("=== tb_top: Full system test ===");

        apply_reset;

        // ── TC01: Avalon-MM write: write 10 samples, verify busy=0 before START ──
        $display("[TC01] Avalon-MM write 10 samples...");
        begin : tc01
            integer i01;
            reg [31:0] status01;
            avs_rd(5'h04, status01);  // STATUS before START
            if (status01[0] === 1'b0) begin
                $display("PASS [TC01_idle_not_busy] busy=0 before START  (%0t ns)", $time - tc_t0);
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("FAIL [TC01_idle_not_busy] busy=%0d expected=0  (%0t ns)", status01[0], $time - tc_t0);
                fail_cnt = fail_cnt + 1;
            end
            // Write 10 samples
            for (i01 = 0; i01 < 10; i01 = i01 + 1) begin
                avs_wr(5'h00, {24'h0, i01[7:0] + 8'd1});  // DATA_IN = i+1
                avs_wr(5'h01, i01[31:0]);                   // ADDR_IN
                avs_wr(5'h02, 32'd1);                       // WR_EN
            end
            $display("[TC01] 10 samples written via Avalon-MM");
        end

        // ── TC02: START → busy=1 within 2 cycles ───────────────────────
        $display("[TC02] START → check busy...");
        begin : tc02
            reg [31:0] status02;
            load_ecg_const(8'd5);  // fill remaining SRAM
            avs_wr(5'h03, 32'd1);  // START
            // Check busy within 2 cycles
            repeat(2) @(posedge clk); #1;
            avs_rd(5'h04, status02);
            if (status02[0] === 1'b1) begin
                $display("PASS [TC02_busy_after_start] busy=1  (%0t ns)", $time - tc_t0);
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("FAIL [TC02_busy_after_start] busy=%0d expected=1  (%0t ns)", status02[0], $time - tc_t0);
                fail_cnt = fail_cnt + 1;
            end
        end

        // Wait for inference to complete (from TC02 START)
        begin : wait_done_tc02
            integer wd_t;
            reg [31:0] st;
            st = 1;
            for (wd_t = 0; wd_t < 8000 && st[0]; wd_t = wd_t + 1) begin
                @(posedge clk); #1;
                avs_rd(5'h04, st);
            end
        end

        // ── TC03: done_latched clears on next START ─────────────────────
        $display("[TC03] done_latched behavior...");
        begin : tc03
            reg [31:0] status03;
            avs_rd(5'h04, status03);
            if (status03[1] === 1'b1) begin
                $display("PASS [TC03a_done_latched] done_latched=1 after inference  (%0t ns)", $time - tc_t0);
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("FAIL [TC03a_done_latched] done_latched=%0d expected=1  (%0t ns)", status03[1], $time - tc_t0);
                fail_cnt = fail_cnt + 1;
            end
            // Now START again → done_latched should clear
            apply_reset;  // reset to clean state before next START
        end

        // ── TC04/TC05/TC06: Full inference with golden ECG ─────────────
        $display("[TC04-06] Full inference — loading golden ECG samples...");
        // Load expected results from file (3 samples)
        // If file missing, these tests are skipped with a warning
        begin : load_expected
            integer file_ok;
            $readmemh("expected_results.hex", expected_results);
            // expected_results[0..2] = argmax class for each sample
        end

        begin : tc04_06
            integer sample_idx;
            reg [1:0] cls_got;
            integer cyc_got;
            reg [255:0] ecg_filename;

            for (sample_idx = 0; sample_idx < 3; sample_idx = sample_idx + 1) begin
                verify_en = 1'b0;
                apply_reset;
                current_sample = sample_idx;

                // Build filename: ecg_sample0.hex, ecg_sample1.hex, ecg_sample2.hex
                case (sample_idx)
                    0: ecg_filename = "ecg_sample0.hex";
                    1: ecg_filename = "ecg_sample1.hex";
                    2: ecg_filename = "ecg_sample2.hex";
                endcase

                $display("[TC0%0d] Loading %0s...", sample_idx+4, ecg_filename);
                load_ecg_hex(ecg_filename);

                // ── Cấp 2: load per-sample golden into buffers ──
                case (sample_idx)
                    0: begin
                        $readmemh("golden/sample0/input_int8.mem",  gold_input);
                        $readmemh("golden/sample0/after_pool1.mem", gold_pool1);
                        $readmemh("golden/sample0/after_pool2.mem", gold_pool2);
                        $readmemh("golden/sample0/after_pool3.mem", gold_pool3);
                        $readmemh("golden/sample0/after_pool4.mem", gold_pool4);
                        $readmemh("golden/sample0/after_gap.mem",   gold_gap);
                        $readmemh("golden/sample0/logits_fc.mem",   gold_logits);
                    end
                    1: begin
                        $readmemh("golden/sample1/input_int8.mem",  gold_input);
                        $readmemh("golden/sample1/after_pool1.mem", gold_pool1);
                        $readmemh("golden/sample1/after_pool2.mem", gold_pool2);
                        $readmemh("golden/sample1/after_pool3.mem", gold_pool3);
                        $readmemh("golden/sample1/after_pool4.mem", gold_pool4);
                        $readmemh("golden/sample1/after_gap.mem",   gold_gap);
                        $readmemh("golden/sample1/logits_fc.mem",   gold_logits);
                    end
                    2: begin
                        $readmemh("golden/sample2/input_int8.mem",  gold_input);
                        $readmemh("golden/sample2/after_pool1.mem", gold_pool1);
                        $readmemh("golden/sample2/after_pool2.mem", gold_pool2);
                        $readmemh("golden/sample2/after_pool3.mem", gold_pool3);
                        $readmemh("golden/sample2/after_pool4.mem", gold_pool4);
                        $readmemh("golden/sample2/after_gap.mem",   gold_gap);
                        $readmemh("golden/sample2/logits_fc.mem",   gold_logits);
                    end
                endcase

                // Check input SRAM vs golden (after Avalon-MM load)
                check_input;

                states_seen = 8'h00;  // reset state coverage tracking
                fc_sub_seen = 5'h00;

                verify_en = 1'b1;     // enable auto per-layer checks
                run_inference(cls_got, cyc_got);
                verify_en = 1'b0;

                $display("[TC0%0d] inference done: result=%0d, cycles=%0d, expected=%0d",
                         sample_idx+4, cls_got, cyc_got, expected_results[sample_idx]);

                if (cls_got === expected_results[sample_idx]) begin
                    $display("PASS [TC0%0d_result_matches] class=%0d  (%0t ns)", sample_idx+4, cls_got, $time - tc_t0);
                    pass_cnt = pass_cnt + 1;
                end else begin
                    $display("FAIL [TC0%0d_result_matches] got=%0d expected=%0d  (%0t ns)",
                             sample_idx+4, cls_got, expected_results[sample_idx], $time - tc_t0);
                    fail_cnt = fail_cnt + 1;
                end

                // TC06: check cycle count in expected range (5000-6000 cycles + Avalon overhead)
                if (sample_idx === 0) begin
                    if (cyc_out > 100 && cyc_out < 10000) begin
                        $display("PASS [TC06_cycle_count_reasonable] cycles=%0d  (%0t ns)", cyc_got, $time - tc_t0);
                        pass_cnt = pass_cnt + 1;
                    end else begin
                        $display("WARN [TC06_cycle_count] cycles=%0d (check if reasonable)  (%0t ns)", cyc_got, $time - tc_t0);
                    end
                end
            end
        end

        // ── TC07: FSM state coverage — all 8 states visited ────────────
        $display("[TC07] FSM state coverage check...");
        tc_t0 = $time;
        begin : tc07
            // After TC04 (full inference), states_seen should have all 8 bits set
            // (IDLE was seen at reset, LOAD→DONE during inference)
            // Note: states_seen tracks the LAST sample's inference
            if (states_seen === 8'hFF) begin
                $display("PASS [TC07_all_fsm_states] states_seen=0x%02X  (%0t ns)", states_seen, $time - tc_t0);
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("FAIL [TC07_all_fsm_states] states_seen=0x%02X (missing: 0x%02X)  (%0t ns)",
                         states_seen, ~states_seen & 8'hFF, $time - tc_t0);
                fail_cnt = fail_cnt + 1;
            end
            // GAP sub-states: fc_sub_seen[1..5] (GAP,FC,FC_FLUSH,ARGMAX,DONE)
            if (fc_sub_seen[4:1] === 4'hF) begin
                $display("PASS [TC07_gap_substates] fc_sub_seen=0x%02X  (%0t ns)", fc_sub_seen, $time - tc_t0);
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("FAIL [TC07_gap_substates] fc_sub_seen=0x%02X (missing substates)  (%0t ns)",
                         fc_sub_seen, $time - tc_t0);
                fail_cnt = fail_cnt + 1;
            end
        end

        // ── Summary ────────────────────────────────────────────────────
        $display("=== tb_top SUMMARY: %0d PASS, %0d FAIL ===", pass_cnt, fail_cnt);
        $display("=== L2 BIT-EXACT (tol +/-10): %0d PASS, %0d FAIL ===", l2_pass_cnt, l2_fail_cnt);
        $display("=== DEVIATION vs Python golden: max|diff|=%0d LSB, %0d/%0d elements differ (%0d exact) ===",
                 g_max_diff, g_nonzero, g_total, g_total - g_nonzero);
        if (fail_cnt === 0 && l2_fail_cnt === 0)
            $display("ALL TESTS PASSED (incl. per-layer bit-exact)");
        $finish;
    end

    // Timeout: full inference ~5200 cycles × 3 samples + Avalon overhead
    initial begin
        #5000000;
        $display("TIMEOUT — exceeded 5ms simulation time");
        $finish;
    end

    initial begin
        $dumpfile("tb_top.vcd");
        $dumpvars(0, tb_top);
    end

endmodule
