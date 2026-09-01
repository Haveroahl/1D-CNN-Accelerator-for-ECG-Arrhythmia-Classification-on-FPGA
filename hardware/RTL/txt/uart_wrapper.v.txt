// uart_wrapper.v
// UART bus adapter for ecg_core — drop-in alternative to avalon_slave for
// boards driven from a PC over a serial line (no Nios V / HPS needed).
//
//   PC ──RX/TX──► uart_wrapper ──8 wires──► ecg_core
//                 (this module)             (datapath + FSM, unchanged)
//
// The 8-wire core interface is IDENTICAL to what avalon_slave drives, so
// ecg_core.v is reused verbatim (no RTL change to the verified core).
//
// ── Command-based protocol (host → FPGA), all bytes 8N1 ──────────────────
//
//   CMD_LOAD  (0xA0) + 2500 data bytes
//       Streams the next 2500 ECG samples into input_sram at addresses
//       0..2499 (auto-incrementing). After the 2500th byte the wrapper
//       returns ACK (0x55). Any other count is not enforced — host must send
//       exactly 2500 bytes after the opcode.
//
//   CMD_START (0xA1)
//       Pulses `start` for one cycle (kicks off inference). Returns ACK.
//
//   CMD_STATUS(0xA2)
//       Returns one status byte: {5'b0, isram_free, done_latched, busy}.
//       done_latched is set when the core asserts `done`, cleared by CMD_START.
//
//   CMD_RESULT(0xA3)
//       Returns one byte {6'b0, result[1:0]} — the argmax class 0..3.
//
// Each opcode is one byte. CMD_LOAD is the only multi-byte transaction; the
// wrapper counts the 2500 payload bytes itself. Unknown opcodes are ignored.
//
// Reset: synchronous active-high `rst` for the core path (matches ecg_core),
// but rx is double-flopped here for async-pin safety.

module uart_wrapper #(
    parameter CLK_FREQ = 100_000_000,
    parameter BAUD     = 115_200
) (
    input  wire        clk,
    input  wire        rst,        // synchronous, active high (core domain)

    // ── Physical UART pins ─────────────────────────────────────────────
    input  wire        uart_rx,    // async serial in
    output wire        uart_tx,    // serial out

    // ── 8-wire core interface (same as avalon_slave drives) ────────────
    output reg  [11:0] sram_wr_addr,
    output reg  [7:0]  sram_din,
    output reg         sram_we,
    output reg         start,
    input  wire        busy,
    input  wire        done,
    input  wire [1:0]  result,
    input  wire        isram_free
);

    // ── Opcodes ────────────────────────────────────────────────────────
    localparam [7:0] CMD_LOAD   = 8'hA0,
                     CMD_START  = 8'hA1,
                     CMD_STATUS = 8'hA2,
                     CMD_RESULT = 8'hA3,
                     RESP_ACK   = 8'h55;

    // ── RX synchronizer (async pin → core clock) ───────────────────────
    reg rx_meta, rx_sync;
    always @(posedge clk) begin
        if (rst) begin
            rx_meta <= 1'b1;
            rx_sync <= 1'b1;
        end else begin
            rx_meta <= uart_rx;
            rx_sync <= rx_meta;
        end
    end

    // ── UART RX/TX primitives ──────────────────────────────────────────
    wire       rx_valid;
    wire [7:0] rx_data;
    reg        tx_start;
    reg  [7:0] tx_data;
    wire       tx_busy;
    wire       tx_done;

    uart_rx #(.CLK_FREQ(CLK_FREQ), .BAUD(BAUD)) u_rx (
        .clk(clk), .rst(rst),
        .rx(rx_sync), .rx_valid(rx_valid), .rx_data(rx_data)
    );

    uart_tx #(.CLK_FREQ(CLK_FREQ), .BAUD(BAUD)) u_tx (
        .clk(clk), .rst(rst),
        .tx_start(tx_start), .tx_data(tx_data),
        .tx(uart_tx), .tx_busy(tx_busy), .tx_done(tx_done)
    );

    // ── done latch (mirrors avalon_slave done_latched semantics) ───────
    reg done_latched;

    // ── Command FSM ────────────────────────────────────────────────────
    localparam [1:0] W_OP   = 2'd0,   // waiting for an opcode byte
                     W_LOAD = 2'd1,   // streaming 2500 payload bytes
                     W_RESP = 2'd2;   // waiting for tx_done after a reply

    reg [1:0]  state;
    reg [11:0] load_cnt;   // 0..2499

    always @(posedge clk) begin
        if (rst) begin
            sram_wr_addr <= 12'd0;
            sram_din     <= 8'd0;
            sram_we      <= 1'b0;
            start        <= 1'b0;
            tx_start     <= 1'b0;
            tx_data      <= 8'd0;
            done_latched <= 1'b0;
            state        <= W_OP;
            load_cnt     <= 12'd0;
        end else begin
            // default 1-cycle pulses
            sram_we  <= 1'b0;
            start    <= 1'b0;
            tx_start <= 1'b0;

            if (done)
                done_latched <= 1'b1;

            case (state)
                // ── Decode opcode ─────────────────────────────────────
                W_OP: begin
                    if (rx_valid) begin
                        case (rx_data)
                            CMD_LOAD: begin
                                load_cnt <= 12'd0;
                                state    <= W_LOAD;
                            end
                            CMD_START: begin
                                start        <= 1'b1;
                                done_latched <= 1'b0;
                                tx_data      <= RESP_ACK;
                                tx_start     <= 1'b1;
                                state        <= W_RESP;
                            end
                            CMD_STATUS: begin
                                tx_data  <= {5'b0, isram_free, done_latched, busy};
                                tx_start <= 1'b1;
                                state    <= W_RESP;
                            end
                            CMD_RESULT: begin
                                tx_data  <= {6'b0, result};
                                tx_start <= 1'b1;
                                state    <= W_RESP;
                            end
                            default: ; // unknown opcode → ignore, stay W_OP
                        endcase
                    end
                end

                // ── Stream 2500 payload bytes into input_sram ─────────
                W_LOAD: begin
                    if (rx_valid) begin
                        sram_wr_addr <= load_cnt;
                        sram_din     <= rx_data;
                        sram_we      <= 1'b1;
                        if (load_cnt == 12'd2499) begin
                            tx_data  <= RESP_ACK;   // queue ACK after last byte
                            tx_start <= 1'b1;
                            state    <= W_RESP;
                        end else begin
                            load_cnt <= load_cnt + 12'd1;
                        end
                    end
                end

                // ── Wait for the reply frame to finish transmitting ───
                W_RESP: begin
                    if (tx_done)
                        state <= W_OP;
                end

                default: state <= W_OP;
            endcase
        end
    end

endmodule
