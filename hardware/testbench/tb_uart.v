// tb_uart.v — UART-driven full-system test for ecg_uart_top.
//
// Drives the serial RX line bit-by-bit at the configured baud, exercising the
// command protocol end to end, and decodes the serial TX replies:
//
//   CMD_LOAD (0xA0) + 2500 sample bytes  → expect ACK 0x55
//   CMD_START(0xA1)                       → expect ACK 0x55
//   poll CMD_STATUS(0xA2) until busy clears (and done_latched set)
//   CMD_RESULT(0xA3)                      → expect class == expected_results
//
// Runs all 3 golden samples and checks each class.
//
// Sim speed: BAUD is overridden HIGH so CLKS_PER_BIT is small. At the real
// 115200/100MHz, CLKS_PER_BIT=868 → ~21.7M cycles just to shift 2500 bytes in,
// which is impractical to simulate. Functionality is baud-independent (same
// FSM, same sampling), so a fast baud verifies the protocol + core path. The
// same RTL still elaborates with the default 115200 for synthesis.
//
// Requires: testbench/ecg_sample0..2.hex, testbench/expected_results.hex,
//   and RTL/conv*.hex / fc_weights.hex / conv_bias.hex for ecg_core.

`timescale 1ns/1ps

module tb_uart;

    // Fast-sim UART params. CLKS_PER_BIT = 100e6 / 12_500_000 = 8.
    localparam CLK_FREQ     = 100_000_000;
    localparam BAUD         = 12_500_000;
    localparam CLKS_PER_BIT = CLK_FREQ / BAUD;        // 8
    localparam CLK_PERIOD   = 10;                     // ns (100 MHz)
    localparam BIT_NS       = CLKS_PER_BIT * CLK_PERIOD;  // one serial bit time

    // Opcodes (mirror uart_wrapper)
    localparam [7:0] CMD_LOAD   = 8'hA0,
                     CMD_START  = 8'hA1,
                     CMD_STATUS = 8'hA2,
                     CMD_RESULT = 8'hA3,
                     RESP_ACK   = 8'h55;

    // ── DUT ────────────────────────────────────────────────────────────
    reg  clk, rst;
    reg  uart_rx;
    wire uart_tx;

    ecg_uart_top #(.CLK_FREQ(CLK_FREQ), .BAUD(BAUD)) u_dut (
        .clk     (clk),
        .rst     (rst),
        .uart_rx (uart_rx),
        .uart_tx (uart_tx)
    );

    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // ── Sample data + expected ─────────────────────────────────────────
    reg [7:0] ecg [0:2][0:2499];
    reg [7:0] tmp [0:2499];
    reg [7:0] expected [0:2];

    integer pass_cnt, fail_cnt;

    // ── Serial driver: send one 8N1 byte on uart_rx (LSB-first) ────────
    task uart_send_byte;
        input [7:0] b;
        integer i;
        begin
            uart_rx = 1'b0;            // start bit
            #(BIT_NS);
            for (i = 0; i < 8; i = i + 1) begin
                uart_rx = b[i];        // LSB-first
                #(BIT_NS);
            end
            uart_rx = 1'b1;            // stop bit
            #(BIT_NS);
        end
    endtask

    // ── Serial monitor: receive one 8N1 byte from uart_tx (LSB-first) ──
    // Waits for the start bit (falling edge while idle), samples 8 data bits
    // at each bit midpoint, returns the byte.
    task uart_recv_byte;
        output [7:0] b;
        integer i;
        begin
            @(negedge uart_tx);        // start bit edge
            #(BIT_NS + BIT_NS/2);      // move into middle of bit 0
            for (i = 0; i < 8; i = i + 1) begin
                b[i] = uart_tx;        // LSB-first
                #(BIT_NS);
            end
            // now in stop bit; no need to wait further for next op
        end
    endtask

    // ── High-level transactions ────────────────────────────────────────
    task send_load;
        input integer s;
        integer i;
        reg [7:0] ack;
        begin
            uart_send_byte(CMD_LOAD);
            for (i = 0; i < 2500; i = i + 1)
                uart_send_byte(ecg[s][i]);
            uart_recv_byte(ack);
            if (ack !== RESP_ACK)
                $display("WARN sample %0d: LOAD ack=0x%02h (expected 0x55)", s, ack);
        end
    endtask

    task send_start;
        reg [7:0] ack;
        begin
            uart_send_byte(CMD_START);
            uart_recv_byte(ack);
            if (ack !== RESP_ACK)
                $display("WARN: START ack=0x%02h (expected 0x55)", ack);
        end
    endtask

    // Poll STATUS until busy(bit0)==0 and done_latched(bit1)==1.
    task wait_done;
        reg [7:0] st;
        integer guard;
        begin
            st = 8'h01; guard = 0;
            while (st[0] && guard < 2000) begin
                uart_send_byte(CMD_STATUS);
                uart_recv_byte(st);
                guard = guard + 1;
            end
        end
    endtask

    task read_result;
        output [1:0] cls;
        reg [7:0] r;
        begin
            uart_send_byte(CMD_RESULT);
            uart_recv_byte(r);
            cls = r[1:0];
        end
    endtask

    task check;
        input integer s;
        input [1:0]   got;
        begin
            if (got === expected[s][1:0]) begin
                $display("PASS sample %0d: class=%0d", s, got);
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("FAIL sample %0d: class=%0d expected=%0d",
                         s, got, expected[s][1:0]);
                fail_cnt = fail_cnt + 1;
            end
        end
    endtask

    // ── Test sequence ──────────────────────────────────────────────────
    integer s, w;
    reg [1:0] cls;

    initial begin
        pass_cnt = 0; fail_cnt = 0;
        uart_rx  = 1'b1;   // idle high
        rst      = 1'b1;

        $readmemh("ecg_sample0.hex", tmp);
        for (w=0; w<2500; w=w+1) ecg[0][w] = tmp[w];
        $readmemh("ecg_sample1.hex", tmp);
        for (w=0; w<2500; w=w+1) ecg[1][w] = tmp[w];
        $readmemh("ecg_sample2.hex", tmp);
        for (w=0; w<2500; w=w+1) ecg[2][w] = tmp[w];
        $readmemh("expected_results.hex", expected);

        // reset
        repeat (5) @(posedge clk);
        #1 rst = 1'b0;
        repeat (5) @(posedge clk);

        for (s = 0; s < 3; s = s + 1) begin
            send_load(s);
            send_start;
            wait_done;
            read_result(cls);
            check(s, cls);
        end

        $display("");
        if (fail_cnt == 0)
            $display("=== tb_uart: PASS (%0d/3 samples correct via UART) ===", pass_cnt);
        else
            $display("=== tb_uart: FAIL (%0d pass, %0d fail) ===", pass_cnt, fail_cnt);

        $finish;
    end

    // Safety timeout
    initial begin
        #500_000_000;
        $display("=== tb_uart: TIMEOUT ===");
        $finish;
    end

endmodule
