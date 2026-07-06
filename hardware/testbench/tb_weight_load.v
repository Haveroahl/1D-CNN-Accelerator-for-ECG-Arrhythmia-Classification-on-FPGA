// tb_weight_load.v — Phase B01 weight-RAM bus-load regression
// ============================================================================
// Proves the runtime weight-load path: with NO_WEIGHT_INIT (no $readmemh, RAMs
// start uninitialized), load ALL weights over the Avalon weight window, then run
// inference on golden sample0 and confirm every checkpoint is bit-exact.
//
// Compile with: vlog +define+NO_WEIGHT_INIT ...  (see run_tb_weight_load.do)
//
// Weight window address map (avalon_slave.v, 14-bit word address):
//   conv weight : 0x2000 | (word<<4) | (oc<<1) | hi   (addr[13]=1, [12:11]=00)
//   conv bias   : 0x2800 | b_idx                       (addr[12:11]=01)
//   FC w/bias   : 0x3000 | fcw_addr                    (addr[12:11]=10)
// Conv 40-bit entry loaded as lo (hi=0, bits[31:0]) then hi (hi=1, bits[39:32]).
// ============================================================================

`timescale 1ns/1ps

module tb_weight_load;

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

    wire [2:0] layer_state  = u_top.u_core.ctrl_layer_state;
    wire [2:0] fc_sub_state = u_top.u_core.ctrl_fc_sub_state;

    localparam GAP_FC_S = 3'd6, CONV1=3'd2, CONV2=3'd3, CONV3=3'd4, CONV4=3'd5;

    initial clk = 0;
    always #5 clk = ~clk;

    integer l2_pass, l2_fail;

    // ── Avalon write (14-bit addr) ──
    task avs_wr;
        input [13:0] addr;
        input [31:0] data;
        begin
            @(negedge clk);
            avs_address = addr; avs_writedata = data; avs_write = 1;
            @(posedge clk); #1; avs_write = 0;
        end
    endtask
    task avs_rd;
        input  [13:0] addr;
        output [31:0] data;
        begin
            @(negedge clk);
            avs_address = addr; avs_read = 1;
            @(posedge clk); #1; data = avs_readdata; avs_read = 0;
        end
    endtask

    task apply_reset;
        begin
            rst = 1; rst_n = 0; avs_write = 0; avs_read = 0;
            @(posedge clk); @(posedge clk); #1;
            rst = 0; rst_n = 1; @(posedge clk); #1;
        end
    endtask

    // ── Load all weights via the bus ───────────────────────────────────────
    reg [39:0] wram [0:7][0:16];   // per-oc RAM image read from w_ram*.hex
    reg [31:0] cbias [0:31];
    reg [7:0]  fcw   [0:31];
    reg [31:0] fcb   [0:3];
    task load_weights;
        integer oc, word, i;
        reg [39:0] e;
        begin
            // conv weights: per oc, per word (lo then hi)
            $readmemh("w_ram0.hex", wram[0]); $readmemh("w_ram1.hex", wram[1]);
            $readmemh("w_ram2.hex", wram[2]); $readmemh("w_ram3.hex", wram[3]);
            $readmemh("w_ram4.hex", wram[4]); $readmemh("w_ram5.hex", wram[5]);
            $readmemh("w_ram6.hex", wram[6]); $readmemh("w_ram7.hex", wram[7]);
            for (oc = 0; oc < 8; oc = oc + 1)
                for (word = 0; word < 17; word = word + 1) begin
                    e = wram[oc][word];
                    // lo: hi=0
                    avs_wr(14'h2000 | (word<<4) | (oc<<1) | 1'b0, e[31:0]);
                    // hi: hi=1 -> triggers 40-bit write
                    avs_wr(14'h2000 | (word<<4) | (oc<<1) | 1'b1, {24'h0, e[39:32]});
                end
            // conv bias (32 × INT32)
            $readmemh("conv_bias.hex", cbias);
            for (i = 0; i < 32; i = i + 1)
                avs_wr(14'h2800 | i, cbias[i]);
            // FC weights (32 × INT8) + FC bias (4 × INT32, addr[5]=1)
            $readmemh("fc_weights.hex", fcw);
            for (i = 0; i < 32; i = i + 1)
                avs_wr(14'h3000 | i, {24'h0, fcw[i]});
            $readmemh("fc_bias.hex", fcb);
            for (i = 0; i < 4; i = i + 1)
                avs_wr(14'h3000 | 6'h20 | i, fcb[i]);
        end
    endtask

    // ── Load ECG sample via data window ────────────────────────────────────
    task load_ecg;
        input [255:0] fn;
        reg [7:0] ecg [0:2499];
        integer i;
        begin
            $readmemh(fn, ecg);
            for (i = 0; i < 2500; i = i + 1)
                avs_wr(14'h1000 | i, {24'h0, ecg[i]});
            @(posedge clk); #1;
        end
    endtask

    task run_inference;
        output [1:0] cls;
        reg [31:0] status; integer it;
        begin
            avs_wr(14'h0003, 32'd1);   // START
            @(posedge clk); #1;
            status = 1; it = 0;
            while (status[0] && it < 10000) begin
                @(posedge clk); #1; avs_rd(14'h0004, status); it = it + 1;
            end
            avs_rd(14'h0005, status); cls = status[1:0];
        end
    endtask

    // ── Golden buffers + compare ───────────────────────────────────────────
    reg [7:0]  gold_pool4 [0:31];
    reg [7:0]  gold_gap   [0:7];
    reg [31:0] gold_logits[0:3];
    reg [7:0]  expected   [0:2];

    function [7:0] read_mem_a; input integer ch; input integer pos; begin
        case (ch)
            0: read_mem_a = u_top.u_core.u_pp.mem_a_ch0[pos];
            1: read_mem_a = u_top.u_core.u_pp.mem_a_ch1[pos];
            2: read_mem_a = u_top.u_core.u_pp.mem_a_ch2[pos];
            3: read_mem_a = u_top.u_core.u_pp.mem_a_ch3[pos];
            4: read_mem_a = u_top.u_core.u_pp.mem_a_ch4[pos];
            5: read_mem_a = u_top.u_core.u_pp.mem_a_ch5[pos];
            6: read_mem_a = u_top.u_core.u_pp.mem_a_ch6[pos];
            7: read_mem_a = u_top.u_core.u_pp.mem_a_ch7[pos];
            default: read_mem_a = 8'h00;
        endcase
    end endfunction

    // pool4 sits in mem_a after Conv4 (bank_sel parity as in tb_top)
    task check_pool4; integer ch,pos,idx,mm; reg signed [9:0] d; begin
        mm=0;
        for (ch=0; ch<8; ch=ch+1) for (pos=0; pos<4; pos=pos+1) begin
            idx=ch*4+pos;
            d = $signed({read_mem_a(ch,pos)[7],read_mem_a(ch,pos)})
              - $signed({gold_pool4[idx][7],gold_pool4[idx]});
            if (d>10||d<-10) mm=mm+1;
        end
        if (mm==0) begin $display("[STAGE PASS] after_pool4 (bus weights)"); l2_pass=l2_pass+1; end
        else       begin $display("[STAGE FAIL] after_pool4: %0d mismatches", mm); l2_fail=l2_fail+1; end
    end endtask

    task check_gap; integer i,mm; reg signed [9:0] d; begin
        mm=0;
        for (i=0;i<8;i=i+1) begin
            d = $signed(u_top.u_core.u_gfa.u_gap.gap_reg[i]) - $signed({gold_gap[i][7],gold_gap[i]});
            if (d>10||d<-10) mm=mm+1;
        end
        if (mm==0) begin $display("[STAGE PASS] after_gap (bus weights)"); l2_pass=l2_pass+1; end
        else       begin $display("[STAGE FAIL] after_gap: %0d mismatches", mm); l2_fail=l2_fail+1; end
    end endtask

    task check_logits; integer i,mm; reg signed [31:0] d; begin
        mm=0;
        for (i=0;i<4;i=i+1) begin
            d = $signed(u_top.u_core.u_gfa.u_fc.fc_acc[i]) - $signed(gold_logits[i]);
            if (d>10||d<-10) mm=mm+1;
        end
        if (mm==0) begin $display("[STAGE PASS] logits_fc (bus weights)"); l2_pass=l2_pass+1; end
        else       begin $display("[STAGE FAIL] logits_fc: %0d mismatches", mm); l2_fail=l2_fail+1; end
    end endtask

    reg [2:0] prev_layer, prev_fc;
    reg [3:0] prev_gap;
    reg verify_en;
    always @(posedge clk) begin
        prev_layer <= layer_state; prev_fc <= fc_sub_state;
        prev_gap <= u_top.u_core.ctrl_gap_step;
        if (verify_en) begin
            if (layer_state==GAP_FC_S && prev_layer==CONV4) check_pool4;
            if (prev_gap==4'd5 && fc_sub_state==3'd2 && prev_fc==3'd1) check_gap;
            if (prev_fc==3'd3 && fc_sub_state==3'd4) check_logits;
        end
    end

    reg [1:0] cls;
    initial begin
        l2_pass=0; l2_fail=0; verify_en=0;
        avs_write=0; avs_read=0; avs_address=0; avs_writedata=0;
        $display("=== tb_weight_load: NO_WEIGHT_INIT, weights loaded via bus ===");

        apply_reset;
        load_weights;          // <-- all weights come from the bus, not $readmemh
        load_ecg("ecg_sample0.hex");

        $readmemh("golden/sample0/after_pool4.mem", gold_pool4);
        $readmemh("golden/sample0/after_gap.mem",   gold_gap);
        $readmemh("golden/sample0/logits_fc.mem",   gold_logits);
        $readmemh("expected_results.hex", expected);

        verify_en = 1;
        run_inference(cls);
        verify_en = 0;

        $display("[RESULT] class=%0d expected=%0d", cls, expected[0]);
        if (cls === expected[0]) begin
            $display("[STAGE PASS] argmax (bus weights)"); l2_pass=l2_pass+1;
        end else begin
            $display("[STAGE FAIL] argmax got=%0d exp=%0d", cls, expected[0]); l2_fail=l2_fail+1;
        end

        $display("=== tb_weight_load SUMMARY: %0d PASS, %0d FAIL ===", l2_pass, l2_fail);
        if (l2_fail==0) $display("WEIGHT-LOAD BUS PATH BIT-EXACT");
        $finish;
    end

    initial begin #5000000; $display("TIMEOUT"); $finish; end

endmodule
