// tb_gate.v — Gate-level (post-synthesis) functional simulation of
// ecg_accelerator_top, driving the Quartus EDA netlist (.vo).
//
// WHY a separate TB (cannot reuse tb_top.v):
//   The gate-level netlist flattens the hierarchy and renames internal nets
//   (e.g. \u_core|u_cpe|w_comb[0][24]~105_combout). tb_top.v reads the 21
//   bit-exact checkpoints via hierarchical references into DUT internals
//   (u_top.u_core.u_pp.mem_a_ch0[...] etc.) — those names do NOT exist in the
//   netlist, so tb_top would fail to elaborate. This TB is BLACK-BOX: it only
//   touches the external Avalon-MM ports, so it works against either RTL or
//   the synthesized netlist.
//
// WHAT it verifies:
//   The post-synthesis netlist (real LUT/DSP/M10K mapping; weights already
//   constant-folded into logic by Quartus) produces the correct argmax class
//   and the correct deterministic latency for the 3 reference samples — i.e.
//   synthesis did not alter functional behavior.
//
//   NOTE: this is FUNCTIONAL (zero-delay) gate-level sim. SDF timing
//   back-annotation is not available for Cyclone V under Quartus Prime Lite
//   (EDA Netlist Writer warning 10905: functional netlist is the only
//   supported type for this device). Timing closure is covered separately by
//   STA (TimeQuest, slack > 0 @100MHz).
//
// Requires (in CWD = hardware/fpga/simulation/questa):
//   ecg_accelerator_top.vo         — synthesized netlist (Quartus EDA writer)
//   ecg_sample0/1/2.hex            — INT8 ECG samples (2500 entries each)
//   expected_results.hex           — expected argmax class per sample (3,1,2)
//
// Weights are embedded in the netlist (constant-folded) — no weight .hex load.

`timescale 1ns/1ps

module tb_gate;

    // ── DUT ports (external only) ─────────────────────────────────────
    reg         clk, rst, rst_n;
    reg  [4:0]  avs_address;
    reg         avs_write, avs_read;
    reg  [31:0] avs_writedata;
    wire [31:0] avs_readdata;

    ecg_accelerator_top u_dut (
        .clk          (clk),
        .rst          (rst),
        .rst_n        (rst_n),
        .avs_address  (avs_address),
        .avs_write    (avs_write),
        .avs_read     (avs_read),
        .avs_writedata(avs_writedata),
        .avs_readdata (avs_readdata)
    );

    // ── Clock: 10 ns period (so cycle = elapsed_ns / 10) ──────────────
    initial clk = 0;
    always #5 clk = ~clk;

    // ── Avalon-MM helpers (identical protocol to tb_top.v) ────────────
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

    // Load ECG sample (2500 × INT8) over Avalon-MM.
    task load_ecg_hex;
        input [255:0] filename;
        reg [7:0] ecg [0:2499];
        integer i;
        begin
            $readmemh(filename, ecg);
            for (i = 0; i < 2500; i = i + 1) begin
                avs_wr(5'h00, {24'h0, ecg[i]});  // DATA_IN
                avs_wr(5'h01, i[31:0]);           // ADDR_IN
                avs_wr(5'h02, 32'd1);             // WR_EN
            end
            // let final WR_EN pulse commit to input_sram before next op
            @(posedge clk); #1;
        end
    endtask

    // Full inference: START → poll STATUS busy → read RESULT, count cycles.
    task run_inference;
        output [1:0] cls;
        output integer cycles;
        reg [31:0] status;
        time         t_start, t_end;
        integer      poll_iter;
        begin
            avs_wr(5'h03, 32'd1);   // START
            @(posedge clk); #1;
            t_start   = $time;
            poll_iter = 0;
            status    = 1;
            while (status[0] && poll_iter < 10000) begin
                @(posedge clk); #1;
                avs_rd(5'h04, status);   // STATUS: [0]=busy
                poll_iter = poll_iter + 1;
            end
            t_end  = $time;
            cycles = (t_end - t_start) / 10;  // 10 ns period
            avs_rd(5'h05, status);   // RESULT: [1:0]=class
            cls = status[1:0];
        end
    endtask

    // DUT reset (rst sync for core, rst_n async for bus). Hold long enough to
    // flush gate-level power-up X out of all registers (power_up="low").
    task apply_reset;
        begin
            rst = 1; rst_n = 0;
            avs_write = 0; avs_read = 0; avs_address = 0; avs_writedata = 0;
            repeat (4) @(posedge clk);
            #1;
            rst = 0; rst_n = 1;
            @(posedge clk); #1;
        end
    endtask

    // ── Test sequence ─────────────────────────────────────────────────
    integer    pass_cnt, fail_cnt;
    reg [7:0]  expected_results [0:2];
    reg [1:0]  cls_out;
    integer    cyc_out;
    integer    s;
    reg [255:0] ecg_filename;
    localparam integer EXPECTED_CYCLES = 5216;

    initial begin
        pass_cnt = 0; fail_cnt = 0;
        $display("==== tb_gate: gate-level (functional) sim of synthesized netlist ====");
        $readmemh("expected_results.hex", expected_results);

        for (s = 0; s < 3; s = s + 1) begin
            apply_reset;
            case (s)
                0: ecg_filename = "ecg_sample0.hex";
                1: ecg_filename = "ecg_sample1.hex";
                2: ecg_filename = "ecg_sample2.hex";
            endcase
            $display("[GLS%0d] loading %0s ...", s, ecg_filename);
            load_ecg_hex(ecg_filename);
            run_inference(cls_out, cyc_out);
            $display("[GLS%0d] result=%0d (expected=%0d), cycles=%0d (expected=%0d)",
                     s, cls_out, expected_results[s][1:0], cyc_out, EXPECTED_CYCLES);

            // argmax check
            if (cls_out === expected_results[s][1:0]) begin
                $display("PASS [GLS%0d_argmax] class=%0d", s, cls_out);
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("FAIL [GLS%0d_argmax] got=%0d expected=%0d", s, cls_out, expected_results[s][1:0]);
                fail_cnt = fail_cnt + 1;
            end

            // latency check
            if (cyc_out == EXPECTED_CYCLES) begin
                $display("PASS [GLS%0d_latency] cycles=%0d", s, cyc_out);
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("FAIL [GLS%0d_latency] cycles=%0d expected=%0d", s, cyc_out, EXPECTED_CYCLES);
                fail_cnt = fail_cnt + 1;
            end
        end

        $display("=== tb_gate SUMMARY: %0d PASS, %0d FAIL ===", pass_cnt, fail_cnt);
        if (fail_cnt == 0)
            $display("ALL GATE-LEVEL TESTS PASSED (netlist argmax + latency match RTL)");
        else
            $display("GATE-LEVEL TESTS FAILED");
        $finish;
    end

    // Safety timeout
    initial begin
        #2_000_000;   // 2 ms >> 3 × ~52 us inference + Avalon load
        $display("FAIL [timeout] tb_gate did not finish");
        $finish;
    end

endmodule
