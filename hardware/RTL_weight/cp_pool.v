// cp_pool.v
// MaxPool rolling comparator (S9)
//
// Rolling max over a window of 5 valid samples. pool_cnt counts 0..4; on the 5th
// sample (pool_cnt==4) it emits pool_write and resets. Gated by relu_v (pipeline
// valid) AND compute_en_in (the original ce_d5 gate) so junk from the SRW priming
// phase is never counted — identical gating to the original cp_block S9.

module cp_pool (
    input  wire                  clk,
    input  wire                  rst,
    input  wire                  pool_rst,

    // From cp_accumulate_rescale
    input  wire signed [7:0]     relu_out,
    input  wire                  relu_v,

    // Original gate (ce_d5) — pool only counts when pipeline carries valid data
    input  wire                  compute_en_in,

    // Outputs
    output wire                  pool_write,      // write strobe (AND cp_en externally)
    output wire signed [7:0]     pool_out         // max-pooled value to Pong SRAM
);

    reg signed [7:0] max_reg;
    reg [2:0]        pool_cnt;
    reg              pool_write_r;

    always @(posedge clk) begin
        pool_write_r <= 1'b0;
        if (rst || pool_rst) begin
            pool_cnt     <= 3'd0;
            pool_write_r <= 1'b0;
            max_reg      <= 8'sd0;   // clear max so X-state never propagates post-reset
        end else if (relu_v && compute_en_in) begin
            // Gate pool_cnt with compute_en_in (= ce_d6): only count when pipeline
            // carries valid data, not junk from SRW priming phase.
            if (pool_cnt == 3'd0)
                max_reg <= relu_out;
            else if (relu_out > max_reg)
                max_reg <= relu_out;

            if (pool_cnt == 3'd4) begin
                pool_cnt     <= 3'd0;
                pool_write_r <= 1'b1;
            end else begin
                pool_cnt <= pool_cnt + 3'd1;
            end
        end
    end

    assign pool_write = pool_write_r;
    assign pool_out   = max_reg;

endmodule
