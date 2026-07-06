// tb_topo_sweep.v — manifest-driven coverage sweep over the channel-scalable space.
//
// Companion to tb_topo.v. Instead of hard-coding each topology, this TB reads
// software/python/gen_topo_golden.py's `topo_golden/topo_manifest.txt` and runs
// EVERY listed topology end-to-end, comparing fc_acc[0..3] to the golden logits
// bit-exact. The manifest guarantees every out_ch value 1..8 appears at every
// layer position, plus monotone / non-monotone / random shapes — i.e. it proves
// the runtime-reconfigurable RTL handles arbitrary per-layer 1..8 channels with
// a SINGLE bitstream, not just the trained Chapman (1,4,4,8) or powers of two.
//
// Manifest row (whitespace-separated, '#' comment lines skipped):
//   tag c1 c2 c3 c4  ic0 ic1 ic2 ic3  ce0 ce1 ce2 ce3  nb0..3  bs0..3  argmax
// Each tag has its own golden dir topo_golden/<tag>/ with w_ram*/conv_bias/
// fc_weights/fc_bias/logits_fc.mem/after_gap.mem.
//
// Compiled with +define+NO_WEIGHT_INIT so the default Chapman $readmemh is off.
// Run from the sim cwd that contains topo_golden/ and ecg_sample0.hex.

`timescale 1ns/1ps

module tb_topo_sweep;

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
            for (i = 0; i < 2500; i = i + 1)
                bus_wr14(14'h1000 | i[11:0], {24'h0, ecg[i]});
            @(posedge clk); #1;
        end
    endtask

    task run_inference;
        output integer cycles;
        integer poll; time t0;
        begin
            bus_wr14(14'h0003, 32'd1);   // START
            @(posedge clk); #1;
            t0 = $time; poll = 0;
            while (u_top.u_core.busy && poll < 20000) begin
                @(posedge clk); #1; poll = poll + 1;
            end
            cycles = ($time - t0) / 10;
        end
    endtask

    // ── Per-topology golden + driver (mirrors tb_topo.v run_topology) ──────
    reg [31:0] gold_logits [0:3];

    task load_topo_weights;
        input [1023:0] tag_dir;   // wide enough for "topo_golden/<tag>/"
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

    // One topology end-to-end from manifest-parsed parameters.
    task run_topology;
        input [1023:0] tag_dir;     // "topo_golden/<tag>/"
        input [255:0]  tag;         // display tag
        input [3:0] ic0, ic1, ic2, ic3;
        input [7:0] ce0, ce1, ce2, ce3;
        input [4:0] nb0, nb1, nb2, nb3;
        input [4:0] bs0, bs1, bs2, bs3;
        integer k, ndiff, cyc;
        reg signed [31:0] got, exp;
        begin
            apply_reset;
            load_topo_weights(tag_dir);
            $readmemh({tag_dir, "logits_fc.mem"}, gold_logits);
            cfg_wr(2'd0, 2'd0, {28'h0, ic0}); cfg_wr(2'd1, 2'd0, {28'h0, ic1});
            cfg_wr(2'd2, 2'd0, {28'h0, ic2}); cfg_wr(2'd3, 2'd0, {28'h0, ic3});
            cfg_wr(2'd0, 2'd1, {24'h0, ce0}); cfg_wr(2'd1, 2'd1, {24'h0, ce1});
            cfg_wr(2'd2, 2'd1, {24'h0, ce2}); cfg_wr(2'd3, 2'd1, {24'h0, ce3});
            cfg_wr(2'd0, 2'd2, {27'h0, nb0}); cfg_wr(2'd1, 2'd2, {27'h0, nb1});
            cfg_wr(2'd2, 2'd2, {27'h0, nb2}); cfg_wr(2'd3, 2'd2, {27'h0, nb3});
            cfg_wr(2'd0, 2'd3, {27'h0, bs0}); cfg_wr(2'd1, 2'd3, {27'h0, bs1});
            cfg_wr(2'd2, 2'd3, {27'h0, bs2}); cfg_wr(2'd3, 2'd3, {27'h0, bs3});
            load_ecg_hex("ecg_sample0.hex");
            run_inference(cyc);
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
                $display("PASS [%0s] 4/4 logits bit-exact  latency=%0d cy (%0.2f us)",
                         tag, cyc, cyc / 100.0);
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("FAIL [%0s] %0d/4 logits mismatch  latency=%0d cy", tag, ndiff, cyc);
                fail_cnt = fail_cnt + 1;
            end
        end
    endtask

    // ── Manifest reader ────────────────────────────────────────────────────
    integer fd, rc, nrun;
    reg [255:0] tag;
    reg [1023:0] tag_dir;
    integer c1, c2, c3, c4;
    integer ic0, ic1, ic2, ic3;
    integer ce0, ce1, ce2, ce3;
    integer nb0, nb1, nb2, nb3;
    integer bs0, bs1, bs2, bs3;
    integer am;
    reg [8*256-1:0] line;

    initial begin
        $display("=== tb_topo_sweep: manifest-driven channel-scalable coverage ===");
        fd = $fopen("topo_golden/topo_manifest.txt", "r");
        if (fd == 0) begin
            $display("FATAL: cannot open topo_golden/topo_manifest.txt");
            $finish;
        end
        nrun = 0;
        while (!$feof(fd)) begin
            rc = $fgets(line, fd);
            if (rc != 0) begin
                // Comment ('# ...') / blank lines make %s read "#" then %d fail,
                // so rc != 22 and the row is skipped naturally.
                rc = $sscanf(line,
                    "%s %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d",
                    tag, c1, c2, c3, c4,
                    ic0, ic1, ic2, ic3, ce0, ce1, ce2, ce3,
                    nb0, nb1, nb2, nb3, bs0, bs1, bs2, bs3, am);
                if (rc == 22) begin
                    tag_dir = {"topo_golden/", tag, "/"};
                    run_topology(tag_dir, tag,
                        ic0[3:0], ic1[3:0], ic2[3:0], ic3[3:0],
                        ce0[7:0], ce1[7:0], ce2[7:0], ce3[7:0],
                        nb0[4:0], nb1[4:0], nb2[4:0], nb3[4:0],
                        bs0[4:0], bs1[4:0], bs2[4:0], bs3[4:0]);
                    nrun = nrun + 1;
                end
            end
        end
        $fclose(fd);
        $display("=== tb_topo_sweep SUMMARY: %0d run, %0d PASS, %0d FAIL ===",
                 nrun, pass_cnt, fail_cnt);
        if (fail_cnt == 0 && nrun > 0)
            $display("ALL %0d CHANNEL-SCALABLE TOPOLOGIES PASSED", nrun);
        $finish;
    end

    initial begin
        #200000000;
        $display("TIMEOUT");
        $finish;
    end

endmodule
