// tb_cpb_cycle_probe.v — dump cp_block ch0 datapath registers each cycle
// around the start of Conv1, to capture the REAL run-time timing including the
// SRW priming / zero-padding phase. Read-only probe; reuses ecg_accelerator_top.
`timescale 1ns/1ps

module tb_cpb_cycle_probe;
    reg        clk, rst, rst_n;
    reg [12:0] avs_address;
    reg        avs_write, avs_read;
    reg [31:0] avs_writedata;
    wire [31:0] avs_readdata;

    ecg_accelerator_top u_top (
        .clk(clk), .rst(rst), .rst_n(rst_n),
        .avs_address(avs_address), .avs_write(avs_write), .avs_read(avs_read),
        .avs_writedata(avs_writedata), .avs_readdata(avs_readdata)
    );

    // Hierarchy into core / cp_engine / cp_block[0]
    wire [2:0] layer_state = u_top.u_core.ctrl_layer_state;
    wire [3:0] a           = u_top.u_core.u_cpe.a;
    wire       shift_en    = u_top.u_core.u_cpe.shift_en;
    wire       srw_rst     = u_top.u_core.u_cpe.srw_rst;
    wire       compute_en  = u_top.u_core.u_cpe.compute_en;
    wire       pad_zero_r  = u_top.u_core.u_cpe.pad_zero_r;
    wire [11:0] rd_addr_in = u_top.u_core.u_cpe.sram_rd_addr_in;
    wire signed [7:0] srw0 = u_top.u_core.u_cpe.srw_flat[0]; // ch0 slot0 (newest)
    wire signed [7:0] srw4 = u_top.u_core.u_cpe.srw_flat[4]; // ch0 slot4 (oldest)
    wire [39:0] mux_s1     = u_top.u_core.u_cpe.mux_s1;
    wire [3:0] a_d5        = u_top.u_core.u_cpe.a_d5;
    wire       ce_d5       = u_top.u_core.u_cpe.ce_d5;
    // cp_block[0] internals
    wire signed [19:0] tree_out = u_top.u_core.u_cpe.cp_blocks[0].u_cp.tree_out;
    wire signed [31:0] acc      = u_top.u_core.u_cpe.cp_blocks[0].u_cp.u_accres.acc;
    wire signed [31:0] accf     = u_top.u_core.u_cpe.cp_blocks[0].u_cp.u_accres.acc_final_r;
    wire       accf_v  = u_top.u_core.u_cpe.cp_blocks[0].u_cp.u_accres.acc_final_v;
    wire signed [31:0] shifted  = u_top.u_core.u_cpe.cp_blocks[0].u_cp.u_accres.shifted;
    wire signed [7:0]  clamped  = u_top.u_core.u_cpe.cp_blocks[0].u_cp.u_accres.clamped;
    wire signed [7:0]  relu_out = u_top.u_core.u_cpe.cp_blocks[0].u_cp.u_accres.relu_out;
    wire       relu_v   = u_top.u_core.u_cpe.cp_blocks[0].u_cp.u_accres.relu_v;
    wire       pool_wr  = u_top.u_core.u_cpe.cp_blocks[0].u_cp.pool_write;
    wire signed [7:0] pool_out = u_top.u_core.u_cpe.cp_blocks[0].u_cp.pool_out;

    localparam CONV1 = 3'd2;
    localparam CONV4 = 3'd5;
`ifdef PROBE_CONV4
    localparam PROBE_LS = CONV4;
`else
    localparam PROBE_LS = CONV1;
`endif

    // clock
    initial clk = 0;
    always #5 clk = ~clk;

    integer i, cyc;
    integer started;

    // Avalon write task
    task avs_wr(input [12:0] addr, input [31:0] data);
        begin
            @(negedge clk);
            avs_address = addr; avs_writedata = data; avs_write = 1; avs_read = 0;
            @(negedge clk);
            avs_write = 0;
        end
    endtask

    // load ecg sample into input SRAM via Avalon DATA window
    reg [7:0] ecg [0:2499];

    initial begin
        rst = 1; rst_n = 0;
        avs_address = 0; avs_write = 0; avs_read = 0; avs_writedata = 0;
        started = 0; cyc = 0;
        repeat (4) @(negedge clk);
        rst = 0; rst_n = 1;
        @(negedge clk);

        // load input ECG (addr window 0x1000..): word address = 0x1000 | idx
        $readmemh("ecg_sample0.hex", ecg);
        for (i = 0; i < 2500; i = i + 1)
            avs_wr(13'h1000 | i[11:0], {24'd0, ecg[i]});
        @(negedge clk); #1;

        // START (control reg 0x0003 bit0)
        avs_wr(13'h0003, 32'h1);

        // run long enough: Conv1 alone is ~2500 cy; to reach Conv4 need ~5000 cy
        repeat (5200) @(posedge clk);
        $display("=== PROBE DONE ===");
        $finish;
    end

    // per-cycle dump: only while relevant (from first PROBE_LS to +60 cycles)
    always @(posedge clk) begin
        if (layer_state == PROBE_LS && started == 0) begin
            started = 1; cyc = 0;
            $display("");
            $display("cyc | ls a sh srwR ce pad rdIn | srw0 srw4 mux_s1        a5 ce5 | tree_out    acc         accf av | shft   clmp relu rv | pw pout");
            $display("----+------------------------- +--------------------------------+------------------------------------+--------------------+--------");
        end
        if (started == 1 && cyc <= 60) begin
            $display("%3d |  %0d %0d  %0d   %0d  %0d  %0d  %4d | %4d %4d %010h %0d  %0d  | %8d %11d %11d %0d | %6d %4d %4d %0d | %0d %4d",
                cyc, layer_state, a, shift_en, srw_rst, compute_en, pad_zero_r, rd_addr_in,
                srw0, srw4, mux_s1, a_d5, ce_d5,
                tree_out, acc, accf, accf_v,
                shifted, clamped, relu_out, relu_v, pool_wr, pool_out);
            cyc = cyc + 1;
        end
    end
endmodule
