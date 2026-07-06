// tb_topo.v — full-inference bit-exact check for MIN/MAX channel topologies.
//
// Proves the runtime-reconfigurable RTL runs a COMPLETE inference correctly for
// channel layouts other than trained Chapman (1,4,4,8): the MIN (2,2,2,2) and
// MAX (8,8,8,8) output-channel cases. Golden = synthetic INT8 reference from
// software/python/gen_topo_golden.py (same bit-exact pipeline as generate_golden).
//
// Per topology:
//   1. $readmemh the topology's weights/bias/FC into the DUT arrays
//      (hierarchical — the bus weight-load path is separately proven by
//       tb_weight_load; here we focus on the datapath honoring the topology).
//   2. Write in_ch / cp_en / nb / layer_base via the Avalon CONFIG window.
//   3. Load the same ECG sample used to compute the golden, via Avalon.
//   4. Run inference, compare fc_acc[0..3] to golden logits — must be bit-exact.
//
// Compiled with +define+NO_WEIGHT_INIT so the default Chapman $readmemh is off.
// Golden dir layout (relative to sim cwd): topo_golden/<tag>/{w_ram*,conv_bias,
//   fc_weights,fc_bias,logits_fc}.

`timescale 1ns/1ps

module tb_topo;

    reg        clk, rst, rst_n;
    reg [13:0] avs_address;
    reg        avs_write, avs_read;
    reg [31:0] avs_writedata;
    wire [31:0] avs_readdata;

    ecg_accelerator_top u_top (
        .clk(clk), .rst(rst), .rst_n(rst_n),
        .avs_address(avs_address), .avs_write(avs_write), .avs_read(avs_read),
        .avs_writedata(avs_writedata), .avs_readdata(avs_readdata)
    );

    wire busy = u_top.u_core.busy;
    wire [2:0] layer_state = u_top.u_core.ctrl_layer_state;

    initial clk = 0;
    always #5 clk = ~clk;

    integer pass_cnt = 0, fail_cnt = 0;

    // ── Avalon helpers ────────────────────────────────────────────────────
    task bus_wr14;
        input [13:0] addr;
        input [31:0] data;
        begin
            @(negedge clk);
            avs_address = addr; avs_writedata = data; avs_write = 1;
            @(posedge clk); #1;
            avs_write = 0; avs_address = 0;
        end
    endtask

    // CONFIG window: addr[13]=1, addr[12:11]=11; field=addr[3:2], layer=addr[1:0]
    task cfg_wr;
        input [1:0] layer; input [1:0] field; input [31:0] data;
        begin
            bus_wr14(14'h3800 | (field << 2) | layer, data);
        end
    endtask

    task apply_reset;
        begin
            rst = 1; rst_n = 0; avs_write = 0; avs_read = 0; avs_address = 0; avs_writedata = 0;
            @(posedge clk); @(posedge clk); #1;
            rst = 0; rst_n = 1;
            @(posedge clk); #1;
        end
    endtask

    task load_ecg_hex;
        input [255:0] filename;
        reg [7:0] ecg [0:2499];
        integer i;
        begin
            $readmemh(filename, ecg);
            for (i = 0; i < 2500; i = i + 1) begin
                bus_wr14(14'h1000 | i[11:0], {24'h0, ecg[i]});  // ECG data window
            end
            @(posedge clk); #1;
        end
    endtask

    // Run inference; return compute latency in clock cycles. Latency is measured
    // from the START pulse to busy-deassert via $time (10 ns period), NOT poll
    // count — so it is the true cycle count and varies with the configured
    // topology (more input channels per layer ⇒ more accumulate cycles).
    task run_inference;
        output integer cycles;
        integer     poll;
        time        t0;
        begin
            bus_wr14(14'h0003, 32'd1);   // START (low reg 3)
            // Wait for busy to assert, then count cycles until it deasserts.
            // Poll the internal busy directly (no Avalon reads) so cycles is the
            // pure compute latency for this topology.
            @(posedge clk); #1;
            t0 = $time;
            poll = 0;
            while (u_top.u_core.busy && poll < 20000) begin
                @(posedge clk); #1;
                poll = poll + 1;
            end
            cycles = ($time - t0) / 10;
        end
    endtask

    // ── Per-topology golden + driver ──────────────────────────────────────
    reg [31:0] gold_logits [0:3];
    reg [7:0]  gold_gap   [0:7];

    // Load topo weights directly into the DUT arrays (hierarchical $readmemh).
    task load_topo_weights;
        input [255:0] tag_dir;   // e.g. "topo_golden/min2222/"
        begin
            $readmemh({tag_dir, "w_ram0.hex"},   u_top.u_core.u_cpe.u_wstore.w_ram0);
            $readmemh({tag_dir, "w_ram1.hex"},   u_top.u_core.u_cpe.u_wstore.w_ram1);
            $readmemh({tag_dir, "w_ram2.hex"},   u_top.u_core.u_cpe.u_wstore.w_ram2);
            $readmemh({tag_dir, "w_ram3.hex"},   u_top.u_core.u_cpe.u_wstore.w_ram3);
            $readmemh({tag_dir, "w_ram4.hex"},   u_top.u_core.u_cpe.u_wstore.w_ram4);
            $readmemh({tag_dir, "w_ram5.hex"},   u_top.u_core.u_cpe.u_wstore.w_ram5);
            $readmemh({tag_dir, "w_ram6.hex"},   u_top.u_core.u_cpe.u_wstore.w_ram6);
            $readmemh({tag_dir, "w_ram7.hex"},   u_top.u_core.u_cpe.u_wstore.w_ram7);
            $readmemh({tag_dir, "conv_bias.hex"},  u_top.u_core.u_cpe.u_wstore.b_store);
            $readmemh({tag_dir, "fc_weights.hex"}, u_top.u_core.u_gfa.u_fc.fc_w);
            $readmemh({tag_dir, "fc_bias.hex"},    u_top.u_core.u_gfa.u_fc.fc_b);
        end
    endtask

    // Run one topology end-to-end. in_ch/cp_en/nb/base are passed per layer.
    // ecg_file selects the INT8 input hex (most cases reuse the Chapman sample0;
    // the ptbxl case ships its own ptbxl_sample.hex generated by gen_ptbxl_golden).
    task run_topology;
        input [255:0] tag;
        input [255:0] golden_dir;   // dir with trailing slash
        input [3:0] ic0, ic1, ic2, ic3;
        input [7:0] ce0, ce1, ce2, ce3;
        input [4:0] nb0, nb1, nb2, nb3;
        input [4:0] bs0, bs1, bs2, bs3;
        input [255:0] ecg_file;
        integer k, ndiff, cyc;
        reg signed [31:0] got, exp;
        begin
            apply_reset;
            // weights into arrays (after reset so they aren't cleared)
            load_topo_weights(golden_dir);
            $readmemh({golden_dir, "logits_fc.mem"}, gold_logits);
            $readmemh({golden_dir, "after_gap.mem"},  gold_gap);
            // config via bus
            cfg_wr(2'd0, 2'd0, {28'h0, ic0}); cfg_wr(2'd1, 2'd0, {28'h0, ic1});
            cfg_wr(2'd2, 2'd0, {28'h0, ic2}); cfg_wr(2'd3, 2'd0, {28'h0, ic3});
            cfg_wr(2'd0, 2'd1, {24'h0, ce0}); cfg_wr(2'd1, 2'd1, {24'h0, ce1});
            cfg_wr(2'd2, 2'd1, {24'h0, ce2}); cfg_wr(2'd3, 2'd1, {24'h0, ce3});
            cfg_wr(2'd0, 2'd2, {27'h0, nb0}); cfg_wr(2'd1, 2'd2, {27'h0, nb1});
            cfg_wr(2'd2, 2'd2, {27'h0, nb2}); cfg_wr(2'd3, 2'd2, {27'h0, nb3});
            cfg_wr(2'd0, 2'd3, {27'h0, bs0}); cfg_wr(2'd1, 2'd3, {27'h0, bs1});
            cfg_wr(2'd2, 2'd3, {27'h0, bs2}); cfg_wr(2'd3, 2'd3, {27'h0, bs3});
            // ECG (already-INT8 hex — RTL consumes verbatim)
            load_ecg_hex(ecg_file);
            run_inference(cyc);
            // GAP check: gap_reg is the direct FC input, captured at a well-defined
            // point (gap_step==5). after_pool4→mem_a survives post-inference; pool3
            // (mem_b) too. pool1/pool2 are overwritten by pool3/pool4 so not checked.
            begin : gapchk
                integer gi, ndg;
                ndg = 0;
                for (gi = 0; gi < 8; gi = gi + 1)
                    if ($signed(u_top.u_core.u_gfa.u_gap.gap_reg[gi]) !==
                        $signed({gold_gap[gi][7], gold_gap[gi]})) ndg = ndg + 1;
                if (ndg != 0)
                    $display("    [%0s] WARN gap mismatch=%0d/8", tag, ndg);
            end
            // compare logits bit-exact (final verdict)
            ndiff = 0;
            for (k = 0; k < 4; k = k + 1) begin
                got = $signed(u_top.u_core.u_gfa.u_fc.fc_acc[k]);
                exp = $signed(gold_logits[k]);
                if (got !== exp) begin
                    ndiff = ndiff + 1;
                    $display("    [%0s] logit[%0d] got=%0d exp=%0d", tag, k, got, exp);
                end
            end
            if (ndiff == 0) begin
                $display("PASS [%0s] full inference logits bit-exact (4/4)  latency=%0d cycles (%0.2f us @100MHz)",
                         tag, cyc, cyc / 100.0);
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("FAIL [%0s] %0d/4 logits mismatch  latency=%0d cycles", tag, ndiff, cyc);
                fail_cnt = fail_cnt + 1;
            end
        end
    endtask

`ifdef VCD_MAX
    // PowerPlay VCD capture, single-topology so the two builds are comparable
    // (each VCD = exactly ONE inference + its config/ECG load, same structure).
    // Dump the DUT subtree so PowerPlay maps the VCD root to |ecg_accelerator_top.
    //   +define+VCD_MAX            → max8888 (8,8,8,8) → tb_topo_max.vcd
    //   +define+VCD_MAX+VCD_CHAP   → chapman (1,4,4,8) → tb_topo_chap.vcd
    `ifdef VCD_CHAP
    initial begin
        $dumpfile("tb_topo_chap.vcd");
        $dumpvars(0, u_top);
    end
    initial begin
        $display("=== tb_topo VCD: chapman only ===");
        run_topology("chapman", "topo_golden/chapman/",
                     4'd1, 4'd4, 4'd4, 4'd8,
                     8'h0F, 8'h0F, 8'hFF, 8'hFF,
                     5'd8, 5'd6, 5'd6, 5'd7,
                     5'd0, 5'd1, 5'd5, 5'd9, "ecg_sample0.hex");
        $display("=== VCD done: %0d PASS, %0d FAIL ===", pass_cnt, fail_cnt);
        $finish;
    end
    `else
    initial begin
        $dumpfile("tb_topo_max.vcd");
        $dumpvars(0, u_top);
    end
    initial begin
        $display("=== tb_topo VCD: max8888 only ===");
        run_topology("max8888", "topo_golden/max8888/",
                     4'd1, 4'd8, 4'd8, 4'd8,
                     8'hFF, 8'hFF, 8'hFF, 8'hFF,
                     5'd8, 5'd6, 5'd6, 5'd7,
                     5'd0, 5'd1, 5'd9, 5'd17, "ecg_sample0.hex");
        $display("=== VCD done: %0d PASS, %0d FAIL ===", pass_cnt, fail_cnt);
        $finish;
    end
    `endif
`else
    initial begin
        $display("=== tb_topo: MIN/MAX channel topology end-to-end ===");

        // HARNESS CHECK: Chapman (1,4,4,8) via this path must reproduce the
        // proven 21/21 golden — validates the tb_topo harness itself.
        run_topology("chapman", "topo_golden/chapman/",
                     4'd1, 4'd4, 4'd4, 4'd8,
                     8'h0F, 8'h0F, 8'hFF, 8'hFF,
                     5'd8, 5'd6, 5'd6, 5'd7,
                     5'd0, 5'd1, 5'd5, 5'd9, "ecg_sample0.hex");

        // MIN (2,2,2,2): in_ch=(1,2,2,2) cp_en=0x03 base=(0,1,3,5)
        run_topology("min2222", "topo_golden/min2222/",
                     4'd1, 4'd2, 4'd2, 4'd2,
                     8'h03, 8'h03, 8'h03, 8'h03,
                     5'd8, 5'd6, 5'd6, 5'd7,
                     5'd0, 5'd1, 5'd3, 5'd5, "ecg_sample0.hex");

        // MAX (8,8,8,8): in_ch=(1,8,8,8) cp_en=0xFF base=(0,1,9,17)
        run_topology("max8888", "topo_golden/max8888/",
                     4'd1, 4'd8, 4'd8, 4'd8,
                     8'hFF, 8'hFF, 8'hFF, 8'hFF,
                     5'd8, 5'd6, 5'd6, 5'd7,
                     5'd0, 5'd1, 5'd9, 5'd17, "ecg_sample0.hex");

        // ── Diagnostic sweep (varied channel counts, all must pass) ────────
        // t4488 = Chapman channel shape (1,4,4,8) w/ random weights.
        run_topology("t4488", "topo_golden/t4488/",
                     4'd1, 4'd4, 4'd4, 4'd8, 8'h0F, 8'h0F, 8'hFF, 8'hFF,
                     5'd8, 5'd6, 5'd6, 5'd7, 5'd0, 5'd1, 5'd5, 5'd9, "ecg_sample0.hex");
        // t4888 = Conv3 in_ch=8 (untested), rest Chapman-ish.
        run_topology("t4888", "topo_golden/t4888/",
                     4'd1, 4'd4, 4'd8, 4'd8, 8'h0F, 8'hFF, 8'hFF, 8'hFF,
                     5'd8, 5'd6, 5'd6, 5'd7, 5'd0, 5'd1, 5'd5, 5'd13, "ecg_sample0.hex");
        // t4448 = Conv4 in_ch=4 (untested), Conv4 out=8.
        run_topology("t4448", "topo_golden/t4448/",
                     4'd1, 4'd4, 4'd4, 4'd4, 8'h0F, 8'h0F, 8'h0F, 8'hFF,
                     5'd8, 5'd6, 5'd6, 5'd7, 5'd0, 5'd1, 5'd5, 5'd9, "ecg_sample0.hex");
        // t2448 = Conv2 in_ch=2 (untested), rest Chapman-ish.
        run_topology("t2448", "topo_golden/t2448/",
                     4'd1, 4'd2, 4'd4, 4'd4, 8'h03, 8'h0F, 8'h0F, 8'hFF,
                     5'd8, 5'd6, 5'd6, 5'd7, 5'd0, 5'd1, 5'd3, 5'd7, "ecg_sample0.hex");

        // ── Architectural-floor + odd-channel cases (prove 1..8 per layer, not
        //    just powers of two) ──────────────────────────────────────────────
        // min1111 = absolute MIN: out_ch=1 every layer. in_ch=(1,1,1,1) base=(0,1,2,3).
        run_topology("min1111", "topo_golden/min1111/",
                     4'd1, 4'd1, 4'd1, 4'd1, 8'h01, 8'h01, 8'h01, 8'h01,
                     5'd8, 5'd6, 5'd6, 5'd7, 5'd0, 5'd1, 5'd2, 5'd3, "ecg_sample0.hex");
        // t3456 = ascending odd/even mix out_ch=(3,4,5,6). in_ch=(1,3,4,5) base=(0,1,4,8).
        run_topology("t3456", "topo_golden/t3456/",
                     4'd1, 4'd3, 4'd4, 4'd5, 8'h07, 8'h0F, 8'h1F, 8'h3F,
                     5'd8, 5'd6, 5'd6, 5'd7, 5'd0, 5'd1, 5'd4, 5'd8, "ecg_sample0.hex");
        // t3577 = odd channels out_ch=(3,5,7,7). in_ch=(1,3,5,7) base=(0,1,4,9).
        run_topology("t3577", "topo_golden/t3577/",
                     4'd1, 4'd3, 4'd5, 4'd7, 8'h07, 8'h1F, 8'h7F, 8'h7F,
                     5'd8, 5'd6, 5'd6, 5'd7, 5'd0, 5'd1, 5'd4, 5'd9, "ecg_sample0.hex");

        // ── PTB-XL real weights (cross-dataset C3): SAME bitstream, runtime
        //    nb[Conv3]=7 (vs Chapman 6) + PTB-XL weights, real test sample.
        //    Golden from gen_ptbxl_golden.py (INT8 ref == this RTL).
        run_topology("ptbxl", "topo_golden/ptbxl/",
                     4'd1, 4'd4, 4'd4, 4'd8, 8'h0F, 8'h0F, 8'hFF, 8'hFF,
                     5'd8, 5'd6, 5'd7, 5'd7, 5'd0, 5'd1, 5'd5, 5'd9, "ptbxl_sample.hex");

        $display("=== tb_topo SUMMARY: %0d PASS, %0d FAIL ===", pass_cnt, fail_cnt);
        if (fail_cnt == 0) $display("ALL TOPOLOGY TESTS PASSED");
        $finish;
    end
`endif

    initial begin
        #5000000;
        $display("TIMEOUT");
        $finish;
    end

endmodule
