// uart_tx.v
// 8N1 UART transmitter. On tx_start (1-cycle pulse) latches tx_data, sends
// start + 8 data (LSB-first) + 1 stop bit at the configured baud rate.
// tx_busy is high for the whole frame; tx_done pulses 1 cycle at end.
//
// Parameterized by CLK_FREQ / BAUD; default 100 MHz, 115200 baud.
// Mirror of uart_rx timing (CLKS_PER_BIT = CLK_FREQ / BAUD).

module uart_tx #(
    parameter CLK_FREQ = 100_000_000,
    parameter BAUD     = 115_200
) (
    input  wire       clk,
    input  wire       rst,        // synchronous, active high
    input  wire       tx_start,   // 1-cycle pulse to begin a frame
    input  wire [7:0] tx_data,
    output reg        tx,         // serial out (idle high)
    output reg        tx_busy,
    output reg        tx_done     // 1-cycle pulse when frame complete
);

    localparam integer CLKS_PER_BIT = CLK_FREQ / BAUD;

    localparam [2:0] S_IDLE  = 3'd0,
                     S_START = 3'd1,
                     S_DATA  = 3'd2,
                     S_STOP  = 3'd3,
                     S_DONE  = 3'd4;

    reg [2:0]  state;
    reg [15:0] clk_cnt;
    reg [2:0]  bit_idx;
    reg [7:0]  shifter;

    always @(posedge clk) begin
        if (rst) begin
            state   <= S_IDLE;
            clk_cnt <= 16'd0;
            bit_idx <= 3'd0;
            shifter <= 8'd0;
            tx      <= 1'b1;    // idle line high
            tx_busy <= 1'b0;
            tx_done <= 1'b0;
        end else begin
            tx_done <= 1'b0;    // default: deassert pulse

            case (state)
                S_IDLE: begin
                    tx      <= 1'b1;
                    tx_busy <= 1'b0;
                    clk_cnt <= 16'd0;
                    bit_idx <= 3'd0;
                    if (tx_start) begin
                        shifter <= tx_data;
                        tx_busy <= 1'b1;
                        tx      <= 1'b0;    // start bit
                        state   <= S_START;
                    end
                end

                // ── Hold start bit for one bit time ────────────────────
                S_START: begin
                    if (clk_cnt < CLKS_PER_BIT-1) begin
                        clk_cnt <= clk_cnt + 16'd1;
                    end else begin
                        clk_cnt <= 16'd0;
                        tx      <= shifter[0];   // first data bit (LSB)
                        state   <= S_DATA;
                    end
                end

                // ── Shift out 8 data bits, LSB-first ───────────────────
                S_DATA: begin
                    if (clk_cnt < CLKS_PER_BIT-1) begin
                        clk_cnt <= clk_cnt + 16'd1;
                    end else begin
                        clk_cnt <= 16'd0;
                        if (bit_idx == 3'd7) begin
                            bit_idx <= 3'd0;
                            tx      <= 1'b1;     // stop bit
                            state   <= S_STOP;
                        end else begin
                            bit_idx <= bit_idx + 3'd1;
                            tx      <= shifter[bit_idx + 3'd1];
                        end
                    end
                end

                // ── Hold stop bit for one bit time ─────────────────────
                S_STOP: begin
                    if (clk_cnt < CLKS_PER_BIT-1) begin
                        clk_cnt <= clk_cnt + 16'd1;
                    end else begin
                        clk_cnt <= 16'd0;
                        state   <= S_DONE;
                    end
                end

                S_DONE: begin
                    tx_busy <= 1'b0;
                    tx_done <= 1'b1;
                    state   <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
