// ============================================================================
// tb_demo_bus.v — replay the ON-BOARD demo driver's bus protocol in simulation.
//
// Purpose: the JTAG demo (soc/ecg_jtag_rom.tcl) talks to the ROM build through
// the SAME avalon_slave register map that this testbench drives. Running it
// proves the driver's transaction sequence is correct BEFORE going to the board:
//   1. block-write 2500 bytes through the DATA WINDOW (word 0x1000 + i)
//   2. write 1 to START (word 0x0003)
//   3. poll STATUS (word 0x0004) until bit1 (done_latched)
//   4. read RESULT (word 0x0005)
// Any mismatch here (window base, done bit, result latch) would be a driver bug
// that costs a board session to find.
//
// Data: 40 stratified samples from the Chapman test set (all 4 classes), the
// SAME bytes demo_data/ningba_test_ecg_int8.bin ships to the board (the on-disk
// file name keeps its ningba_* prefix; "Chapman" is the display/thesis name).
// Reference: demo_pred40.hex = Python INT8 bit-exact predictions (floor GAP).
// PASS = RTL class == Python class for all 40.
// ============================================================================
`timescale 1ns / 1ps

module tb_demo_bus;

    localparam N_SAMPLES = 40;
    localparam SAMPLE_LEN = 2500;

    // Word addresses (avalon_slave.v)
    localparam [13:0] A_START  = 14'h0003;
    localparam [13:0] A_STAT   = 14'h0004;
    localparam [13:0] A_RES    = 14'h0005;
    localparam [13:0] A_WINDOW = 14'h1000;

    reg         clk = 0;
    reg         rst = 1;
    reg         rst_n = 0;
    reg  [13:0] avs_address = 0;
    reg         avs_write = 0;
    reg         avs_read = 0;
    reg  [31:0] avs_writedata = 0;
    wire [31:0] avs_readdata;

    always #5 clk = ~clk;   // 100 MHz

    ecg_accelerator_top dut (
        .clk(clk), .rst(rst), .rst_n(rst_n),
        .avs_address(avs_address), .avs_write(avs_write), .avs_read(avs_read),
        .avs_writedata(avs_writedata), .avs_readdata(avs_readdata)
    );

    // Stimulus / reference
    reg [7:0] ecg_all [0:N_SAMPLES*SAMPLE_LEN-1];
    reg [7:0] labels  [0:N_SAMPLES-1];
    reg [7:0] sw_pred [0:N_SAMPLES-1];

    integer s, i, pass, agree, correct;
    integer cyc0, cyc_total;
    reg [1:0] pred;
    reg [31:0] status;

    // ── Bus helpers: one write / one read, matching master_write_32 semantics ──
    task bus_write(input [13:0] addr, input [31:0] data);
        begin
            @(negedge clk);
            avs_address   = addr;
            avs_writedata = data;
            avs_write     = 1;
            @(negedge clk);
            avs_write     = 0;
        end
    endtask

    task bus_read(input [13:0] addr, output [31:0] data);
        begin
            @(negedge clk);
            avs_address = addr;
            avs_read    = 1;
            @(negedge clk);
            avs_read    = 0;
            @(negedge clk);          // readdata is registered -> valid next cycle
            data        = avs_readdata;
        end
    endtask

    // Step 1 of the driver: ship one sample through the DATA WINDOW.
    task load_sample(input integer idx);
        begin
            for (i = 0; i < SAMPLE_LEN; i = i + 1)
                bus_write(A_WINDOW + i[13:0],
                          {24'b0, ecg_all[idx*SAMPLE_LEN + i]});
            @(posedge clk); #1;      // let the final SRAM write commit
        end
    endtask

    initial begin
        $readmemh("demo_ecg40.hex",  ecg_all);
        $readmemh("demo_lbl40.hex",  labels);
        $readmemh("demo_pred40.hex", sw_pred);

        $display("========================================================");
        $display(" tb_demo_bus — on-board driver protocol replay (ROM build)");
        $display(" %0d stratified Chapman samples, all 4 classes", N_SAMPLES);
        $display("========================================================");

        // Reset
        rst = 1; rst_n = 0;
        repeat (5) @(posedge clk);
        rst = 0; rst_n = 1;
        repeat (5) @(posedge clk);

        pass = 1; agree = 0; correct = 0; cyc_total = 0;

        for (s = 0; s < N_SAMPLES; s = s + 1) begin
            load_sample(s);                      // 1. DATA WINDOW block write

            cyc0 = $time / 10;
            bus_write(A_START, 32'h1);           // 2. START

            status = 0;                          // 3. poll done_latched (bit1)
            while ((status & 32'h2) == 0)
                bus_read(A_STAT, status);
            cyc_total = cyc_total + (($time/10) - cyc0);

            bus_read(A_RES, status);             // 4. RESULT
            pred = status[1:0];

            if (pred === sw_pred[s][1:0]) agree = agree + 1;
            else begin
                pass = 0;
                $display("  [MISMATCH] sample %0d: RTL=%0d  Python=%0d",
                         s, pred, sw_pred[s][1:0]);
            end
            if (pred === labels[s][1:0]) correct = correct + 1;

            if (s < 5 || s % 10 == 0)
                $display("  sample %2d : pred=%0d truth=%0d %s",
                         s, pred, labels[s][1:0],
                         (pred === labels[s][1:0]) ? "OK" : "X");
        end

        $display("--------------------------------------------------------");
        $display(" RTL vs Python (bit-exact) : %0d/%0d agree", agree, N_SAMPLES);
        $display(" RTL vs ground truth       : %0d/%0d correct (%0.1f%%)",
                 correct, N_SAMPLES, 100.0*correct/N_SAMPLES);
        $display(" Avg cycles/inference      : %0d", cyc_total / N_SAMPLES);
        if (pass && agree == N_SAMPLES)
            $display(" RESULT: PASS — driver bus protocol verified, %0d/%0d bit-exact",
                     agree, N_SAMPLES);
        else
            $display(" RESULT: FAIL — %0d mismatches", N_SAMPLES - agree);
        $display("========================================================");
        $finish;
    end

endmodule
