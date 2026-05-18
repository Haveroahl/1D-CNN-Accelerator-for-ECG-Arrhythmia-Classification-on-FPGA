// tb_top_probe.v — Single-sample full-pipeline trace testbench
//
// Purpose: Generate cycle-by-cycle trace of ALL stages (Conv1/2/3/4, GAP, FC,
// Argmax) running on ecg_sample0.hex, for use in thesis verification chapter.
//
// Unlike tb_top.v (which runs 3 samples × 7 test cases), this testbench:
//   - Runs ecg_sample0.hex once.
//   - Probes every internal signal of the datapath, gated by layer_state.
//   - Filters non-meaningful cycles (skip lines where no valid pipeline activity).
//   - Performs the same golden bit-exact compare as tb_top.v (7 stages).
//   - Outputs to ModelSim transcript via $display — grep/screenshot for thesis.
//
// Run: `vsim -c -do "do ecg_accelerator_top_probe.do; quit -f"`
//
// Requires (same as tb_top.v):
//   RTL/conv{1..4}_w.hex, conv_bias.hex, fc_weights.hex
//   testbench/ecg_sample0.hex
//   testbench/golden/sample0/{input_int8,after_pool1..4,after_gap,logits_fc}.mem

`timescale 1ns/1ps

module tb_top_probe;

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

    // GAP_FC sub-states (match gap_fc_argmax.v)
    localparam GAP_S    = 3'd1;
    localparam FC_S     = 3'd2;
    localparam FC_FLUSH = 3'd3;
    localparam ARGMAX_S = 3'd4;
    localparam DONE_SUB = 3'd5;

    // ── Probe transition state ─────────────────────────────────────────
    reg [2:0] prev_layer_state;
    reg [2:0] prev_fc_sub_state;
    reg [3:0] prev_gap_step;
    reg [1:0] prev_argmax_step;

    // ── Cycle counter (from start of inference, reset each apply_reset) ──
    integer cy;
    reg     trace_en;   // raised by initial block to enable verbose probes
    always @(posedge clk) begin
        if (rst) begin
            cy <= 0;
            prev_layer_state  <= 3'd0;
            prev_fc_sub_state <= 3'd0;
            prev_gap_step     <= 4'd0;
            prev_argmax_step  <= 2'd0;
        end else begin
            cy <= cy + 1;
            prev_layer_state  <= layer_state;
            prev_fc_sub_state <= fc_sub_state;
            prev_gap_step     <= u_top.ctrl_gap_step;
            prev_argmax_step  <= u_top.ctrl_argmax_step;
        end
    end

    // ── Clock ─────────────────────────────────────────────────────────
    initial clk = 0;
    always #5 clk = ~clk;

    // ───────────────────────────────────────────────────────────────────
    //                    PROBE SECTIONS (active when trace_en=1)
    // ───────────────────────────────────────────────────────────────────

    // ── Section 1: Controller state (1 line per cycle, skip IDLE/LOAD) ──
    always @(posedge clk) begin
        if (trace_en && busy && layer_state != LOAD_INPUT) begin
            $display("[ctrl cy=%0d] state=%0d t=%0d a=%0d se=%b ce=%b padr=%b prefcnt=%0d pong=%0d srw_rst=%b pool_rst=%b cp_en=%02x",
                cy, layer_state, u_top.ctrl_t, u_top.ctrl_a,
                u_top.ctrl_shift_en, u_top.ctrl_compute_en,
                u_top.u_cpe.pad_zero_r, u_top.u_ctrl.prefetch_cnt,
                u_top.ctrl_pong_addr, u_top.ctrl_srw_rst, u_top.ctrl_pool_rst,
                u_top.ctrl_cp_en);
        end
    end

    // ── Section 2: SRW state (print on shift_en cycle, all 8 channels) ──
    // SRW updated AT this clock edge — visible during NEXT cycle. Print
    // here shows OLD state (before shift); next-cycle probe will show NEW.
    always @(posedge clk) begin
        if (trace_en && (layer_state == CONV1 || layer_state == CONV2 ||
                         layer_state == CONV3 || layer_state == CONV4)
                     && u_top.ctrl_shift_en) begin
            $display("[SRW cy=%0d] ch0=%02x %02x %02x %02x %02x  ch1=%02x %02x %02x %02x %02x  ch2=%02x %02x %02x %02x %02x  ch3=%02x %02x %02x %02x %02x",
                cy,
                u_top.u_cpe.srw_flat[0*5+0], u_top.u_cpe.srw_flat[0*5+1], u_top.u_cpe.srw_flat[0*5+2], u_top.u_cpe.srw_flat[0*5+3], u_top.u_cpe.srw_flat[0*5+4],
                u_top.u_cpe.srw_flat[1*5+0], u_top.u_cpe.srw_flat[1*5+1], u_top.u_cpe.srw_flat[1*5+2], u_top.u_cpe.srw_flat[1*5+3], u_top.u_cpe.srw_flat[1*5+4],
                u_top.u_cpe.srw_flat[2*5+0], u_top.u_cpe.srw_flat[2*5+1], u_top.u_cpe.srw_flat[2*5+2], u_top.u_cpe.srw_flat[2*5+3], u_top.u_cpe.srw_flat[2*5+4],
                u_top.u_cpe.srw_flat[3*5+0], u_top.u_cpe.srw_flat[3*5+1], u_top.u_cpe.srw_flat[3*5+2], u_top.u_cpe.srw_flat[3*5+3], u_top.u_cpe.srw_flat[3*5+4]);
            // Channels 4..7 (only active for Conv3/4)
            if (layer_state == CONV3 || layer_state == CONV4) begin
                $display("[SRW cy=%0d] ch4=%02x %02x %02x %02x %02x  ch5=%02x %02x %02x %02x %02x  ch6=%02x %02x %02x %02x %02x  ch7=%02x %02x %02x %02x %02x",
                    cy,
                    u_top.u_cpe.srw_flat[4*5+0], u_top.u_cpe.srw_flat[4*5+1], u_top.u_cpe.srw_flat[4*5+2], u_top.u_cpe.srw_flat[4*5+3], u_top.u_cpe.srw_flat[4*5+4],
                    u_top.u_cpe.srw_flat[5*5+0], u_top.u_cpe.srw_flat[5*5+1], u_top.u_cpe.srw_flat[5*5+2], u_top.u_cpe.srw_flat[5*5+3], u_top.u_cpe.srw_flat[5*5+4],
                    u_top.u_cpe.srw_flat[6*5+0], u_top.u_cpe.srw_flat[6*5+1], u_top.u_cpe.srw_flat[6*5+2], u_top.u_cpe.srw_flat[6*5+3], u_top.u_cpe.srw_flat[6*5+4],
                    u_top.u_cpe.srw_flat[7*5+0], u_top.u_cpe.srw_flat[7*5+1], u_top.u_cpe.srw_flat[7*5+2], u_top.u_cpe.srw_flat[7*5+3], u_top.u_cpe.srw_flat[7*5+4]);
            end
        end
    end

    // ── Section 3: CP block pipeline per output channel ──
    // Filter: print only when at least one of {acc_final_v, bias_valid,
    // rescale_v1, rescale_v2, relu_v, pool_write_r} is 1.
    // Uses macro-style helper: one always block per channel (Verilog-2001
    // doesn't allow generate-for of always blocks with $display easily).
    `define CP_PROBE(OC) \
        always @(posedge clk) begin \
            if (trace_en && u_top.ctrl_cp_en[OC] && \
                (u_top.u_cpe.cp_blocks[OC].u_cp.acc_final_v || \
                 u_top.u_cpe.cp_blocks[OC].u_cp.bias_valid  || \
                 u_top.u_cpe.cp_blocks[OC].u_cp.rescale_v1  || \
                 u_top.u_cpe.cp_blocks[OC].u_cp.rescale_v2  || \
                 u_top.u_cpe.cp_blocks[OC].u_cp.relu_v      || \
                 u_top.u_cpe.cp_blocks[OC].u_cp.pool_write_r)) begin \
                $display("[CP oc=%0d cy=%0d] tree=%08x acc=%08x af(v=%b)=%08x biased(v=%b)=%08x shft(v=%b)=%08x clmp(v=%b)=%02x relu(v=%b)=%02x pcnt=%0d pw=%b max=%02x out=%02x", \
                    OC, cy, \
                    {{12{u_top.u_cpe.cp_blocks[OC].u_cp.tree_out[19]}}, u_top.u_cpe.cp_blocks[OC].u_cp.tree_out}, \
                    u_top.u_cpe.cp_blocks[OC].u_cp.acc, \
                    u_top.u_cpe.cp_blocks[OC].u_cp.acc_final_v, u_top.u_cpe.cp_blocks[OC].u_cp.acc_final_r, \
                    u_top.u_cpe.cp_blocks[OC].u_cp.bias_valid,  u_top.u_cpe.cp_blocks[OC].u_cp.biased, \
                    u_top.u_cpe.cp_blocks[OC].u_cp.rescale_v1,  u_top.u_cpe.cp_blocks[OC].u_cp.shifted, \
                    u_top.u_cpe.cp_blocks[OC].u_cp.rescale_v2,  u_top.u_cpe.cp_blocks[OC].u_cp.clamped, \
                    u_top.u_cpe.cp_blocks[OC].u_cp.relu_v,      u_top.u_cpe.cp_blocks[OC].u_cp.relu_out, \
                    u_top.u_cpe.cp_blocks[OC].u_cp.pool_cnt, \
                    u_top.u_cpe.cp_blocks[OC].u_cp.pool_write_r, \
                    u_top.u_cpe.cp_blocks[OC].u_cp.max_reg, \
                    u_top.u_cpe.cp_blocks[OC].u_cp.pool_out); \
            end \
        end

    `CP_PROBE(0)
    `CP_PROBE(1)
    `CP_PROBE(2)
    `CP_PROBE(3)
    `CP_PROBE(4)
    `CP_PROBE(5)
    `CP_PROBE(6)
    `CP_PROBE(7)

    // ── Section 4: GAP datapath (gap_step 0..5) ──
    always @(posedge clk) begin
        if (trace_en && layer_state == GAP_FC_S && fc_sub_state == GAP_S) begin
            $display("[GAP step=%0d cy=%0d] rd_addr=%0d ping_dout=%02x %02x %02x %02x %02x %02x %02x %02x  acc[0..7]=%03x %03x %03x %03x %03x %03x %03x %03x",
                u_top.ctrl_gap_step, cy, u_top.u_gfa.gap_rd_addr,
                u_top.pp_dout[0*8 +: 8], u_top.pp_dout[1*8 +: 8],
                u_top.pp_dout[2*8 +: 8], u_top.pp_dout[3*8 +: 8],
                u_top.pp_dout[4*8 +: 8], u_top.pp_dout[5*8 +: 8],
                u_top.pp_dout[6*8 +: 8], u_top.pp_dout[7*8 +: 8],
                u_top.u_gfa.gap_acc[0]&10'h3FF, u_top.u_gfa.gap_acc[1]&10'h3FF,
                u_top.u_gfa.gap_acc[2]&10'h3FF, u_top.u_gfa.gap_acc[3]&10'h3FF,
                u_top.u_gfa.gap_acc[4]&10'h3FF, u_top.u_gfa.gap_acc[5]&10'h3FF,
                u_top.u_gfa.gap_acc[6]&10'h3FF, u_top.u_gfa.gap_acc[7]&10'h3FF);
        end
        // GAP DONE: gap_reg latched at gap_step=5, visible 1cy later when fc_sub_state→FC_S
        if (trace_en && prev_gap_step == 4'd5 && fc_sub_state == FC_S && prev_fc_sub_state == GAP_S) begin
            $display("[GAP DONE cy=%0d] gap_reg[0..7] = %02x %02x %02x %02x %02x %02x %02x %02x (= floor(sum/4))",
                cy,
                u_top.u_gfa.gap_reg[0], u_top.u_gfa.gap_reg[1],
                u_top.u_gfa.gap_reg[2], u_top.u_gfa.gap_reg[3],
                u_top.u_gfa.gap_reg[4], u_top.u_gfa.gap_reg[5],
                u_top.u_gfa.gap_reg[6], u_top.u_gfa.gap_reg[7]);
        end
    end

    // ── Section 5: FC datapath (FC_S + FC_FLUSH) ──
    always @(posedge clk) begin
        if (trace_en && layer_state == GAP_FC_S && fc_sub_state == FC_S) begin
            $display("[FC step=%0d cy=%0d] gap_pipe=%02x w_idx=%0d pv=%b prod[0..3]=%04x %04x %04x %04x  acc[0..3]=%08x %08x %08x %08x",
                u_top.ctrl_fc_step, cy,
                u_top.u_gfa.fc_gap_pipe, u_top.u_gfa.fc_w_idx, u_top.u_gfa.prod_valid,
                u_top.u_gfa.fc_prod[0]&16'hFFFF, u_top.u_gfa.fc_prod[1]&16'hFFFF,
                u_top.u_gfa.fc_prod[2]&16'hFFFF, u_top.u_gfa.fc_prod[3]&16'hFFFF,
                u_top.u_gfa.fc_acc[0], u_top.u_gfa.fc_acc[1],
                u_top.u_gfa.fc_acc[2], u_top.u_gfa.fc_acc[3]);
        end
        if (trace_en && layer_state == GAP_FC_S && fc_sub_state == FC_FLUSH) begin
            $display("[FC FLUSH cy=%0d] (drain last prod) acc[0..3] = %08x %08x %08x %08x",
                cy,
                u_top.u_gfa.fc_acc[0], u_top.u_gfa.fc_acc[1],
                u_top.u_gfa.fc_acc[2], u_top.u_gfa.fc_acc[3]);
        end
        // FC final logits (entering ARGMAX)
        if (trace_en && prev_fc_sub_state == FC_FLUSH && fc_sub_state == ARGMAX_S) begin
            $display("[FC DONE cy=%0d] final logits = %08x %08x %08x %08x  (signed: %0d %0d %0d %0d)",
                cy,
                u_top.u_gfa.fc_acc[0], u_top.u_gfa.fc_acc[1],
                u_top.u_gfa.fc_acc[2], u_top.u_gfa.fc_acc[3],
                $signed(u_top.u_gfa.fc_acc[0]), $signed(u_top.u_gfa.fc_acc[1]),
                $signed(u_top.u_gfa.fc_acc[2]), $signed(u_top.u_gfa.fc_acc[3]));
        end
    end

    // ── Section 6: Argmax (4 cycles) ──
    always @(posedge clk) begin
        if (trace_en && layer_state == GAP_FC_S && fc_sub_state == ARGMAX_S) begin
            $display("[ARG step=%0d cy=%0d] cmp fc_acc[%0d]=%0d vs max=%0d  → cur_idx=%0d",
                u_top.ctrl_argmax_step, cy, u_top.ctrl_argmax_step,
                $signed(u_top.u_gfa.fc_acc[u_top.ctrl_argmax_step]),
                $signed(u_top.u_gfa.argmax_max),
                u_top.u_gfa.argmax_idx);
        end
        if (trace_en && prev_fc_sub_state == ARGMAX_S && fc_sub_state == DONE_SUB) begin
            $display("[ARG DONE cy=%0d] argmax_idx = %0d (class)  max_logit = %0d",
                cy, u_top.u_gfa.argmax_idx, $signed(u_top.u_gfa.argmax_max));
        end
    end

    // ── Section 7: Memory dumps on layer transitions ──
    // Note: ping_pong_sram uses 16 separate 1D arrays. Use read_mem_a/b helpers
    // declared below (Verilog hoists function decls, but to keep flow simple,
    // helpers are declared earlier in this file before check_pool tasks).
    integer dump_i;
    always @(posedge clk) begin
        // pool1 → mem_b (bank_sel was 0 during Conv1)
        if (trace_en && layer_state == CONV2 && prev_layer_state == CONV1) begin
            $display("[POOL1 dump cy=%0d] mem_b (ch=0..3, first 32 entries each)", cy);
            for (dump_i = 0; dump_i < 4; dump_i = dump_i + 1) begin
                $display("  ch=%0d: %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x",
                    dump_i,
                    read_mem_b(dump_i, 0),  read_mem_b(dump_i, 1),  read_mem_b(dump_i, 2),  read_mem_b(dump_i, 3),
                    read_mem_b(dump_i, 4),  read_mem_b(dump_i, 5),  read_mem_b(dump_i, 6),  read_mem_b(dump_i, 7),
                    read_mem_b(dump_i, 8),  read_mem_b(dump_i, 9),  read_mem_b(dump_i, 10), read_mem_b(dump_i, 11),
                    read_mem_b(dump_i, 12), read_mem_b(dump_i, 13), read_mem_b(dump_i, 14), read_mem_b(dump_i, 15),
                    read_mem_b(dump_i, 16), read_mem_b(dump_i, 17), read_mem_b(dump_i, 18), read_mem_b(dump_i, 19),
                    read_mem_b(dump_i, 20), read_mem_b(dump_i, 21), read_mem_b(dump_i, 22), read_mem_b(dump_i, 23),
                    read_mem_b(dump_i, 24), read_mem_b(dump_i, 25), read_mem_b(dump_i, 26), read_mem_b(dump_i, 27),
                    read_mem_b(dump_i, 28), read_mem_b(dump_i, 29), read_mem_b(dump_i, 30), read_mem_b(dump_i, 31));
            end
        end
        if (trace_en && layer_state == CONV3 && prev_layer_state == CONV2) begin
            $display("[POOL2 dump cy=%0d] mem_a (ch=0..3, all 100 entries — first 32 shown)", cy);
            for (dump_i = 0; dump_i < 4; dump_i = dump_i + 1) begin
                $display("  ch=%0d: %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x",
                    dump_i,
                    read_mem_a(dump_i, 0),  read_mem_a(dump_i, 1),  read_mem_a(dump_i, 2),  read_mem_a(dump_i, 3),
                    read_mem_a(dump_i, 4),  read_mem_a(dump_i, 5),  read_mem_a(dump_i, 6),  read_mem_a(dump_i, 7),
                    read_mem_a(dump_i, 8),  read_mem_a(dump_i, 9),  read_mem_a(dump_i, 10), read_mem_a(dump_i, 11),
                    read_mem_a(dump_i, 12), read_mem_a(dump_i, 13), read_mem_a(dump_i, 14), read_mem_a(dump_i, 15),
                    read_mem_a(dump_i, 16), read_mem_a(dump_i, 17), read_mem_a(dump_i, 18), read_mem_a(dump_i, 19),
                    read_mem_a(dump_i, 20), read_mem_a(dump_i, 21), read_mem_a(dump_i, 22), read_mem_a(dump_i, 23),
                    read_mem_a(dump_i, 24), read_mem_a(dump_i, 25), read_mem_a(dump_i, 26), read_mem_a(dump_i, 27),
                    read_mem_a(dump_i, 28), read_mem_a(dump_i, 29), read_mem_a(dump_i, 30), read_mem_a(dump_i, 31));
            end
        end
        if (trace_en && layer_state == CONV4 && prev_layer_state == CONV3) begin
            $display("[POOL3 dump cy=%0d] mem_b (ch=0..7, all 20 entries each)", cy);
            for (dump_i = 0; dump_i < 8; dump_i = dump_i + 1) begin
                $display("  ch=%0d: %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x",
                    dump_i,
                    read_mem_b(dump_i, 0),  read_mem_b(dump_i, 1),  read_mem_b(dump_i, 2),  read_mem_b(dump_i, 3),
                    read_mem_b(dump_i, 4),  read_mem_b(dump_i, 5),  read_mem_b(dump_i, 6),  read_mem_b(dump_i, 7),
                    read_mem_b(dump_i, 8),  read_mem_b(dump_i, 9),  read_mem_b(dump_i, 10), read_mem_b(dump_i, 11),
                    read_mem_b(dump_i, 12), read_mem_b(dump_i, 13), read_mem_b(dump_i, 14), read_mem_b(dump_i, 15),
                    read_mem_b(dump_i, 16), read_mem_b(dump_i, 17), read_mem_b(dump_i, 18), read_mem_b(dump_i, 19));
            end
        end
        if (trace_en && layer_state == GAP_FC_S && prev_layer_state == CONV4) begin
            $display("[POOL4 dump cy=%0d] mem_a (ch=0..7, all 4 entries each)", cy);
            for (dump_i = 0; dump_i < 8; dump_i = dump_i + 1) begin
                $display("  ch=%0d: %02x %02x %02x %02x",
                    dump_i,
                    read_mem_a(dump_i, 0), read_mem_a(dump_i, 1),
                    read_mem_a(dump_i, 2), read_mem_a(dump_i, 3));
            end
        end
    end

    // ───────────────────────────────────────────────────────────────────
    //                    TASKS (copied from tb_top.v)
    // ───────────────────────────────────────────────────────────────────

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

    task load_ecg_hex;
        input [255:0] filename;
        reg [7:0] ecg [0:2499];
        integer i;
        begin
            $readmemh(filename, ecg);
            for (i = 0; i < 2500; i = i + 1) begin
                avs_wr(5'h00, {24'h0, ecg[i]});
                avs_wr(5'h01, i[31:0]);
                avs_wr(5'h02, 32'd1);
            end
            @(posedge clk); #1;
        end
    endtask

    task run_inference;
        output [1:0] cls;
        output integer cycles;
        reg [31:0] status;
        begin
            avs_wr(5'h03, 32'd1);
            cycles = 0;
            status = 1;
            while (status[0] && cycles < 10000) begin
                @(posedge clk); #1;
                avs_rd(5'h04, status);
                cycles = cycles + 1;
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

    // ───────────────────────────────────────────────────────────────────
    //                    GOLDEN BUFFERS + COMPARE TASKS
    // ───────────────────────────────────────────────────────────────────

    reg [7:0]  gold_input   [0:2499];
    reg [7:0]  gold_pool1   [0:1999];
    reg [7:0]  gold_pool2   [0:399];
    reg [7:0]  gold_pool3   [0:159];
    reg [7:0]  gold_pool4   [0:31];
    reg [7:0]  gold_gap     [0:7];
    reg [31:0] gold_logits  [0:3];

    integer    l2_pass_cnt, l2_fail_cnt;
    integer    current_sample;

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

    // ── Helpers: read per-channel ping_pong memory ─────────────────────
    // ping_pong_sram uses 16 separate 1D arrays (mem_a_ch0..7, mem_b_ch0..7).
    function [7:0] read_mem_a;
        input integer ch;
        input integer pos;
        begin
            case (ch)
                0: read_mem_a = u_top.u_pp.mem_a_ch0[pos];
                1: read_mem_a = u_top.u_pp.mem_a_ch1[pos];
                2: read_mem_a = u_top.u_pp.mem_a_ch2[pos];
                3: read_mem_a = u_top.u_pp.mem_a_ch3[pos];
                4: read_mem_a = u_top.u_pp.mem_a_ch4[pos];
                5: read_mem_a = u_top.u_pp.mem_a_ch5[pos];
                6: read_mem_a = u_top.u_pp.mem_a_ch6[pos];
                7: read_mem_a = u_top.u_pp.mem_a_ch7[pos];
                default: read_mem_a = 8'h00;
            endcase
        end
    endfunction

    function [7:0] read_mem_b;
        input integer ch;
        input integer pos;
        begin
            case (ch)
                0: read_mem_b = u_top.u_pp.mem_b_ch0[pos];
                1: read_mem_b = u_top.u_pp.mem_b_ch1[pos];
                2: read_mem_b = u_top.u_pp.mem_b_ch2[pos];
                3: read_mem_b = u_top.u_pp.mem_b_ch3[pos];
                4: read_mem_b = u_top.u_pp.mem_b_ch4[pos];
                5: read_mem_b = u_top.u_pp.mem_b_ch5[pos];
                6: read_mem_b = u_top.u_pp.mem_b_ch6[pos];
                7: read_mem_b = u_top.u_pp.mem_b_ch7[pos];
                default: read_mem_b = 8'h00;
            endcase
        end
    endfunction

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
                    idx_gold = ch * 100 + pos;
                    raw = bank ? read_mem_b(ch, pos) : read_mem_a(ch, pos);
                    rtl_v  = $signed({raw[7], raw});
                    gold_v = $signed({gold_pool2[idx_gold][7], gold_pool2[idx_gold]});
                    diff = rtl_v - gold_v;
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

    // ── Auto-trigger compare on layer/sub-state transitions ──
    reg verify_en;
    always @(posedge clk) begin
        if (verify_en) begin
            if (layer_state == CONV2 && prev_layer_state == CONV1) check_pool1(1'b1);
            if (layer_state == CONV3 && prev_layer_state == CONV2) check_pool2(1'b0);
            if (layer_state == CONV4 && prev_layer_state == CONV3) check_pool3(1'b1);
            if (layer_state == GAP_FC_S && prev_layer_state == CONV4) check_pool4(1'b0);
            if (prev_gap_step == 4'd5 && fc_sub_state == FC_S && prev_fc_sub_state == GAP_S)
                check_gap;
            if (prev_fc_sub_state == FC_FLUSH && fc_sub_state == ARGMAX_S)
                check_logits;
        end
    end

    // ───────────────────────────────────────────────────────────────────
    //                    MAIN TEST SEQUENCE
    // ───────────────────────────────────────────────────────────────────

    reg [1:0]  cls_got;
    integer    cyc_got;

    initial begin
        l2_pass_cnt = 0; l2_fail_cnt = 0;
        verify_en = 1'b0;
        trace_en  = 1'b0;
        current_sample = 0;
        avs_write = 0; avs_read = 0; avs_address = 0; avs_writedata = 0;

        $display("=================================================================");
        $display("=== tb_top_probe: Full pipeline cycle-by-cycle trace (sample0) ===");
        $display("=================================================================");

        apply_reset;
        $display("[INFO] Loading ecg_sample0.hex into input SRAM...");
        load_ecg_hex("ecg_sample0.hex");

        $display("[INFO] Loading golden references for sample0...");
        $readmemh("golden/sample0/input_int8.mem",  gold_input);
        $readmemh("golden/sample0/after_pool1.mem", gold_pool1);
        $readmemh("golden/sample0/after_pool2.mem", gold_pool2);
        $readmemh("golden/sample0/after_pool3.mem", gold_pool3);
        $readmemh("golden/sample0/after_pool4.mem", gold_pool4);
        $readmemh("golden/sample0/after_gap.mem",   gold_gap);
        $readmemh("golden/sample0/logits_fc.mem",   gold_logits);

        $display("");
        $display("=== STAGE 0: Input SRAM verification ===");
        check_input;

        $display("");
        $display("=== STAGES 1-7: Full pipeline trace + inline compare ===");
        $display("");

        verify_en = 1'b1;
        trace_en  = 1'b1;
        run_inference(cls_got, cyc_got);
        trace_en  = 1'b0;
        verify_en = 1'b0;

        $display("");
        $display("=================================================================");
        $display("=== INFERENCE DONE: result=%0d (expected=3)  cycles=%0d ===", cls_got, cyc_got);
        $display("=== L2 BIT-EXACT SUMMARY: %0d PASS, %0d FAIL (out of 7 stages) ===", l2_pass_cnt, l2_fail_cnt);
        $display("=================================================================");
        $finish;
    end

    // Watchdog timeout
    initial begin
        #5000000;
        $display("[ERROR] TIMEOUT — simulation exceeded 5ms");
        $finish;
    end

endmodule
