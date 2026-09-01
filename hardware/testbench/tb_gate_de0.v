// tb_gate_de0.v — Gate-level (post-fit) sim of the DE0-Nano netlist, with VCD
// dump for PowerPlay. Black-box: only external Avalon ports (works on netlist).
//
// Fork of tb_gate.v with two changes:
//   1. avs_address widened 5 -> 14 bits to match the current top (Phase B01
//      added the CONFIG/DATA windows; the basic load/start/read map is still
//      0x00..0x05, so the low bits are used exactly as before).
//   2. $dumpvars over the whole DUT to emit a gate-level VCD whose node names
//      match the Cyclone IV netlist 1:1 -> PowerPlay reads it at HIGH confidence
//      (RTL VCDs only cover registers, leaving combinational logic vectorless).
//
// Weights are constant-init in the netlist ($readmemh folded at synth) so no
// weight load over the bus is needed; the default Chapman topology runs.
//
// Requires (in CWD = hardware/fpga_de0/simulation/questa):
//   ecg_de0_100.vo, ecg_sample0/1/2.hex, expected_results.hex

`timescale 1ns/1ps

module tb_gate_de0;

    reg         clk, rst, rst_n;
    reg  [13:0] avs_address;
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

    initial clk = 0;
    always #5 clk = ~clk;   // 10 ns period (100 MHz) — matches the timing-check revision

    task avs_wr;
        input [13:0] addr;
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
        input  [13:0] addr;
        output [31:0] data;
        begin
            @(negedge clk);
            avs_address = addr;
            avs_read    = 1;
            // avs_readdata is REGISTERED (avalon_slave.v:179-184): it captures the
            // selected register on the posedge AT WHICH avs_read is sampled high, so
            // the data is valid only on the NEXT posedge. At RTL the read happens to
            // align on the first edge; at gate level (SDF delays) avs_read reaches the
            // readback register a few ns after that edge and misses it, so we must wait
            // one extra posedge before sampling — matching real Avalon readdatavalid.
            @(posedge clk);          // edge that latches avs_read=1 into readback reg
            @(posedge clk); #1;      // data now stable here
            data     = avs_readdata;
            avs_read = 0;
        end
    endtask

    task load_ecg_hex;
        input [255:0] filename;
        reg [7:0] ecg [0:2499];
        integer i;
        begin
            $readmemh(filename, ecg);
            // DATA WINDOW (avalon_slave.v:160-164): addr[12]=1 -> one write per SRAM
            // byte (din=writedata[7:0], sram_addr=addr[11:0], we auto). One bus write
            // per sample instead of the legacy 3 (DATA_IN/ADDR_IN/WR_EN), cutting the
            // load phase ~3x and shrinking the SDF run / VCD accordingly.
            for (i = 0; i < 2500; i = i + 1) begin
                avs_wr(14'h1000 | i[13:0], {24'h0, ecg[i]});
            end
            @(posedge clk); #1;
        end
    endtask

    // Free-running cycle counter armed at START, used to measure inference latency
    // independently of how many clocks each bus read consumes. Counting clocks via a
    // dedicated counter (instead of (t_end-t_start)/poll-rate) keeps the latency
    // number stable whether avs_rd takes 1 or 2 posedges per poll, at RTL or gate.
    integer cyc_cnt;
    reg     cyc_run;
    initial cyc_run = 1'b0;
    always @(posedge clk) if (cyc_run) cyc_cnt = cyc_cnt + 1;

    task run_inference;
        output [1:0] cls;
        output integer cycles;
        reg [31:0] status;
        begin
            avs_wr(14'h03, 32'd1);   // START
            // Arm the cycle counter on the edge right after START is accepted.
            @(posedge clk); #1;
            cyc_cnt = 0;
            cyc_run = 1'b1;
            status  = 1;
`ifdef DBG_POLL
            // Probe: read STATUS a few times right after START and print raw value,
            // so we can see whether busy ever rises at gate level.
            begin : dbg
                integer k; reg [31:0] st;
                for (k = 0; k < 8; k = k + 1) begin
                    avs_rd(14'h04, st);
                    $display("[DBG] cyc=%0d poll%0d STATUS=0x%08h busy=%b done=%b",
                             cyc_cnt, k, st, st[0], st[1]);
                end
            end
`endif
            // Poll STATUS.busy until it drops. avs_rd consumes 2 posedges (registered
            // readback), but cyc_cnt counts every clock so latency stays accurate. The
            // measured cycle of the FIRST busy==0 read is the readback-delayed view of
            // the FSM finishing; we report cyc_cnt at that point (FSM done == 5216).
            while (status[0] && cyc_cnt < 10000) begin
                avs_rd(14'h04, status);   // STATUS: [0]=busy
            end
            cyc_run = 1'b0;
            // Report raw cycles observed busy-high via the bus. This includes the
            // fixed readback latency of avs_rd, so it is a few cycles above the pure
            // FSM latency (5216, proven bit-exact on the DE10 tb_top run). The point
            // of THIS test is netlist correctness (argmax) + VCD switching activity,
            // not re-proving the cycle count — so we observe latency, not assert it.
            cycles  = cyc_cnt;
            avs_rd(14'h05, status);   // RESULT: [1:0]=class
            cls = status[1:0];
        end
    endtask

    task apply_reset;
        begin
            // Gate-level: hold reset asserted for MANY cycles so every dffeas
            // (power_up="low") settles to 0 and any residual X is flushed before
            // we start polling STATUS. The RTL run tolerates a short reset; the
            // flattened netlist needs a longer one.
            rst = 1; rst_n = 0;
            avs_write = 0; avs_read = 0; avs_address = 0; avs_writedata = 0;
            repeat (20) @(posedge clk);
            #1;
            rst = 0; rst_n = 1;
            repeat (2) @(posedge clk); #1;
        end
    endtask

    integer    pass_cnt, fail_cnt;
    reg [7:0]  expected_results [0:2];
    reg [1:0]  cls_out;
    integer    cyc_out;
    integer    s;
    reg [255:0] ecg_filename;
    localparam integer EXPECTED_CYCLES = 5216;

    // ── VCD dump for PowerPlay (gate-level node names == netlist) ──────────
    // Guarded: the RTL smoke-test run compiles with +define+NO_VCD to skip the
    // (huge) dump — it only checks functional correctness of this TB. The real
    // gate-level run omits the define so the VCD is produced for PowerPlay.
    //
    // VCD WINDOWING: dumping the whole 3×(load 7500 writes + inference) run made
    // a 1.9 GB file. PowerPlay only needs the COMPUTE activity of one inference.
    // So we arm the dump but keep it OFF, then $dumpon just around sample-0's
    // inference and $dumpoff after — the Avalon-load phase (mostly idle compute
    // logic) is excluded, giving a small VCD with the right switching activity.
    reg vcd_enabled;
    initial vcd_enabled = 1'b0;
`ifndef NO_VCD
    initial begin
        $dumpfile("tb_gate_de0.vcd");
        $dumpvars(0, tb_gate_de0.u_dut);   // arm: whole DUT (regs + comb nets)
        $dumpoff;                          // ...but start OFF; windowed below
        vcd_enabled = 1'b1;
    end
`endif

    initial begin
        pass_cnt = 0; fail_cnt = 0;
        $display("==== tb_gate_de0: gate-level sim (DE0 netlist) + VCD dump ====");
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
            // Window the VCD around the compute of sample 0 only (load excluded).
            // Print the exact $time at dumpon/off so the PowerPlay VCD window
            // (POWER_VCD_FILE_START/END_TIME) can be set to the compute interval.
            if (vcd_enabled && s == 0) begin $dumpon; $display("[VCD] dump ON  @ %0t", $time); end
            run_inference(cls_out, cyc_out);
            if (vcd_enabled && s == 0) begin $dumpoff; $display("[VCD] dump OFF @ %0t", $time); end
            $display("[GLS%0d] result=%0d (expected=%0d), cycles=%0d (expected=%0d)",
                     s, cls_out, expected_results[s][1:0], cyc_out, EXPECTED_CYCLES);

            if (cls_out === expected_results[s][1:0]) begin
                $display("PASS [GLS%0d_argmax] class=%0d", s, cls_out);
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("FAIL [GLS%0d_argmax] got=%0d expected=%0d", s, cls_out, expected_results[s][1:0]);
                fail_cnt = fail_cnt + 1;
            end

            // Latency observed via the bus = FSM latency (5216) + a few cycles of
            // registered-readback overhead. Accept a small window above 5216 rather
            // than an exact match; the exact cycle count is proven by tb_top on DE10.
            if (cyc_out >= EXPECTED_CYCLES && cyc_out <= EXPECTED_CYCLES + 4) begin
                $display("PASS [GLS%0d_latency] cycles=%0d (FSM 5216 + readback)", s, cyc_out);
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("FAIL [GLS%0d_latency] cycles=%0d expected ~%0d", s, cyc_out, EXPECTED_CYCLES);
                fail_cnt = fail_cnt + 1;
            end

            // Gate-level (VCD) run only needs one inference's activity for power —
            // stop after sample 0 to keep the VCD small and the run fast. The RTL
            // smoke test (NO_VCD) runs all 3 samples.
            if (vcd_enabled && s == 0) begin
                $display("[VCD] one inference captured — ending gate run early");
                s = 3;
            end
        end

        $display("=== tb_gate_de0 SUMMARY: %0d PASS, %0d FAIL ===", pass_cnt, fail_cnt);
        if (fail_cnt == 0)
            $display("ALL GATE-LEVEL TESTS PASSED");
        else
            $display("GATE-LEVEL TESTS FAILED");
        $finish;
    end

    initial begin
        #2_000_000;
        $display("FAIL [timeout] tb_gate_de0 did not finish");
        $finish;
    end

endmodule
