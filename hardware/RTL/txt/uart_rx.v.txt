// uart_rx.v
// 8N1 UART receiver. Samples RX at the configured baud rate, deserializes one
// start + 8 data + 1 stop frame, pulses rx_valid for 1 cycle with rx_data.
//
// Parameterized by CLK_FREQ / BAUD; default 100 MHz, 115200 baud.
// CLKS_PER_BIT = CLK_FREQ / BAUD = 100e6 / 115200 = 868.
//
// LSB-first (standard UART). No parity. rx must be synchronized externally if
// it comes from an async pin — uart_wrapper double-flops it before driving here.

module uart_rx #(
    parameter CLK_FREQ = 100_000_000,
    parameter BAUD     = 115_200
) (
    input  wire       clk,
    input  wire       rst,        // synchronous, active high
    input  wire       rx,         // serial in (already synchronized)
    output reg        rx_valid,   // 1-cycle pulse when rx_data is ready
    output reg  [7:0] rx_data
);

    localparam integer CLKS_PER_BIT = CLK_FREQ / BAUD;

    localparam [2:0] S_IDLE  = 3'd0,
                     S_START = 3'd1,
                     S_DATA  = 3'd2,
                     S_STOP  = 3'd3,
                     S_DONE  = 3'd4;

    reg [2:0]  state;
    reg [15:0] clk_cnt;   // counts clocks within a bit (max ~868)
    reg [2:0]  bit_idx;   // 0..7
    reg [7:0]  shifter;

    always @(posedge clk) begin
        if (rst) begin
            state    <= S_IDLE;
            clk_cnt  <= 16'd0;
            bit_idx  <= 3'd0;
            shifter  <= 8'd0;
            rx_valid <= 1'b0;
            rx_data  <= 8'd0;
        end else begin
            rx_valid <= 1'b0;   // default: deassert pulse

            case (state)
                // ── Wait for falling edge (start bit) ──────────────────
                S_IDLE: begin
                    clk_cnt <= 16'd0;
                    bit_idx <= 3'd0;
                    if (rx == 1'b0)         // start bit detected
                        state <= S_START;
                end

                // ── Confirm start bit at its midpoint ──────────────────
                S_START: begin
                    if (clk_cnt == (CLKS_PER_BIT-1)/2) begin
                        if (rx == 1'b0) begin   // still low → valid start
                            clk_cnt <= 16'd0;
                            state   <= S_DATA;
                        end else begin
                            state <= S_IDLE;    // glitch
                        end
                    end else begin
                        clk_cnt <= clk_cnt + 16'd1;
                    end
                end

                // ── Sample 8 data bits at each bit midpoint ────────────
                S_DATA: begin
                    if (clk_cnt < CLKS_PER_BIT-1) begin
                        clk_cnt <= clk_cnt + 16'd1;
                    end else begin
                        clk_cnt        <= 16'd0;
                        shifter[bit_idx] <= rx;     // LSB-first
                        if (bit_idx == 3'd7) begin
                            bit_idx <= 3'd0;
                            state   <= S_STOP;
                        end else begin
                            bit_idx <= bit_idx + 3'd1;
                        end
                    end
                end

                // ── Stop bit (1 bit time), then emit ───────────────────
                S_STOP: begin
                    if (clk_cnt < CLKS_PER_BIT-1) begin
                        clk_cnt <= clk_cnt + 16'd1;
                    end else begin
                        clk_cnt  <= 16'd0;
                        rx_data  <= shifter;
                        rx_valid <= 1'b1;
                        state    <= S_DONE;
                    end
                end

                // ── One cycle settle, return to idle ───────────────────
                S_DONE: begin
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
