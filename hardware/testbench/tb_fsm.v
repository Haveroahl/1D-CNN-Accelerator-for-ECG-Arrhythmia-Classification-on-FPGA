// tb_fsm.v — FSM transition-coverage testbench (controller, M3)
//
// Purpose: close the FSM-transition coverage gap left by single-shot tb_top.
// tb_top runs ONE inference (IDLE→LOAD→CONV1..4→GAP_FC→DONE). The streaming
// re-entry edge DONE_S→LOAD_INPUT (start again without rst) is never exercised.
//
// Strategy (simple but standard edge-coverage):
//   - Run TWO inferences back-to-back: first from IDLE (cold start), second by
//     re-starting from DONE_S (warm start, no rst) — this hits DONE_S→LOAD_INPUT
//     and re-traverses every CONV edge a second time.
//   - A monitor records each observed (prev_state -> state) edge into a seen[]
//     table. At the end we assert every EXPECTED main-FSM edge was seen.
//
// This TB checks ONLY control-flow (state sequence), not datapath values — those
// are covered bit-exact by tb_top. Uses a constant ECG pattern (data value is
// irrelevant to the FSM walk).
//
// Run: vsim -c -do run_tb_fsm.do   (coverage: cov_tb_fsm.do)

`timescale 1ns/1ps

module tb_fsm;

    // ── DUT I/O ─────────────────────────────────────────────────────────
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

    wire [2:0] layer_state = u_top.u_core.ctrl_layer_state;

    // FSM encoding (must match cnn_controller.v)
    localparam IDLE=3'd0, LOAD_INPUT=3'd1, CONV1=3'd2, CONV2=3'd3,
               CONV3=3'd4, CONV4=3'd5, GAP_FC_S=3'd6, DONE_S=3'd7;

    // ── Clock ───────────────────────────────────────────────────────────
    initial clk = 0;
    always #5 clk = ~clk;

    // ── Edge monitor: seen[from][to] = 1 when that transition is observed ─
    reg        seen [0:7][0:7];
    reg [2:0]  prev_state;
    integer    fi, ti;
    always @(posedge clk) begin
        if (rst) begin
            prev_state <= IDLE;
        end else begin
            if (layer_state !== prev_state)
                seen[prev_state][layer_state] <= 1'b1;
            prev_state <= layer_state;
        end
    end

    // ── Counters ────────────────────────────────────────────────────────
    integer pass_cnt, fail_cnt;

    // ── Avalon write helper ─────────────────────────────────────────────
    task avs_wr;
        input [4:0]  addr;
        input [31:0] data;
        begin
            @(negedge clk);
            avs_address = addr; avs_writedata = data; avs_write = 1;
            @(posedge clk); #1; avs_write = 0;
        end
    endtask

    task avs_rd;
        input  [4:0]  addr;
        output [31:0] data;
        begin
            @(negedge clk);
            avs_address = addr; avs_read = 1;
            @(posedge clk); #1; data = avs_readdata; avs_read = 0;
        end
    endtask

    // Load a constant ECG buffer (value irrelevant to FSM walk)
    task load_ecg_const;
        input [7:0] val;
        integer i;
        begin
            for (i = 0; i < 2500; i = i + 1) begin
                avs_wr(5'h00, {24'h0, val});
                avs_wr(5'h01, i[31:0]);
                avs_wr(5'h02, 32'd1);
            end
            @(posedge clk); #1;
        end
    endtask

    // Pulse START and wait until busy clears (one full inference)
    task run_inference;
        integer wd;
        reg [31:0] st;
        begin
            avs_wr(5'h03, 32'd1);          // START
            st = 1;
            for (wd = 0; wd < 8000 && st[0]; wd = wd + 1) begin
                @(posedge clk); #1;
                avs_rd(5'h04, st);          // status: bit0 = busy
            end
        end
    endtask

    // Check one expected edge
    task chk_edge;
        input [2:0] f;
        input [2:0] t;
        input [127:0] name;
        begin
            if (seen[f][t]) begin
                $display("PASS [edge %0s] (%0d->%0d) seen", name, f, t);
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("FAIL [edge %0s] (%0d->%0d) NOT seen", name, f, t);
                fail_cnt = fail_cnt + 1;
            end
        end
    endtask

    // ── Main ────────────────────────────────────────────────────────────
    integer a, b;
    initial begin
        pass_cnt = 0; fail_cnt = 0;
        for (a = 0; a < 8; a = a + 1)
            for (b = 0; b < 8; b = b + 1) seen[a][b] = 1'b0;

        $display("=== tb_fsm: FSM transition coverage ===");

        // Reset
        rst = 1; rst_n = 0;
        avs_address = 0; avs_write = 0; avs_read = 0; avs_writedata = 0;
        repeat (10) @(posedge clk); #1;
        rst = 0; rst_n = 1;
        @(posedge clk); #1;

        // Inference #1 — cold start from IDLE
        load_ecg_const(8'd5);
        run_inference;                 // IDLE→LOAD→CONV1..4→GAP_FC→DONE_S

        // Inference #2 — warm re-start from DONE_S (NO rst) → streaming edge
        load_ecg_const(8'd7);
        run_inference;                 // DONE_S→LOAD→...→DONE_S again

        // ── Assert every expected main-FSM edge was traversed ────────────
        chk_edge(IDLE,       LOAD_INPUT, "IDLE_LOAD");
        chk_edge(LOAD_INPUT, CONV1,      "LOAD_CONV1");
        chk_edge(CONV1,      CONV2,      "CONV1_CONV2");
        chk_edge(CONV2,      CONV3,      "CONV2_CONV3");
        chk_edge(CONV3,      CONV4,      "CONV3_CONV4");
        chk_edge(CONV4,      GAP_FC_S,   "CONV4_GAPFC");
        chk_edge(GAP_FC_S,   DONE_S,     "GAPFC_DONE");
        chk_edge(DONE_S,     LOAD_INPUT, "DONE_LOAD_restart");  // streaming edge

        $display("=== tb_fsm SUMMARY: %0d PASS, %0d FAIL ===", pass_cnt, fail_cnt);
        if (fail_cnt == 0) $display("ALL FSM EDGES COVERED");
        else               $display("SOME FSM EDGES MISSING");
        $finish;
    end

    // Safety timeout
    initial begin
        #2_000_000;
        $display("FAIL [tb_fsm] global timeout");
        $finish;
    end

endmodule
