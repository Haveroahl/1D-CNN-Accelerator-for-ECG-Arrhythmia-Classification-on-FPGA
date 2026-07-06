// avalon_slave.v
// Avalon-MM slave interface for ECG accelerator on DE10-Standard.
// Bridges HPS Lightweight bridge / JTAG-to-Avalon master to:
//   Input SRAM write port + control/status registers + runtime weight load.
//
// Address map (WORDS, 14-bit address):
//   Low registers (unchanged, byte-at-a-time path — used by tb_top.v):
//     0x0000 W : sram_din      [7:0]
//     0x0001 W : sram_wr_addr  [11:0]
//     0x0002 W : sram_we       [0]
//     0x0003 W : start         [0]   (clears done_latched)
//     0x0004 R : status        {done_latched, busy}
//     0x0005 R : result        [1:0]
//   ECG DATA WINDOW (addr[13:12]=01, burst path — used by JTAG host):
//     0x1000..0x19C3 W : write word -> sram_din<=wd[7:0], sram_wr_addr<=(addr-0x1000),
//                        sram_we<=1.  index = addr[11:0] (0..2499).
//   WEIGHT WINDOW (addr[13]=1, Phase B01 runtime weight reload):
//     region = addr[12:11]
//       00  conv weight : offset = addr[10:0] = (word*8 + oc)*2 + hi
//             hi=off[0], oc=off[3:1], word=off[8:4].  A 40-bit entry = 2 writes:
//             lo (bits[31:0], hi=0) then hi (bits[39:32], hi=1) -> w_wr_en pulse.
//       01  conv bias   : b_addr  = addr[4:0]  (0..31), data[31:0] = INT32  -> b_wr_en
//       10  FC w/bias   : fcw_addr= addr[5:0]  ([5]=1 bias k=addr[1:0], else fc_w idx) -> fcw_wr_en
//       11  TOPOLOGY CONFIG (runtime channel/nb reconfig — see below):
//             layer = addr[1:0] (0..3 = Conv1..4), field = addr[3:2]:
//               0: in_ch       [3:0]   (1..8)
//               1: cp_en       [7:0]   (active output-channel bitmask)
//               2: nb          [4:0]   (rescale shift)
//               3: layer_base  [4:0]   (weight-RAM word base for this layer)
//             Config registers RESET to the default Chapman topology
//             (in_ch=1,4,4,8 / cp_en=0F,0F,FF,FF / nb=8,6,6,7 / base=0,1,5,9),
//             so with NO config writes the DUT behaves exactly as before
//             (tb_top.v 21/21 bit-exact unaffected). The driver overrides these
//             only when loading a different topology + matching weights.
// The ECG window and low-register paths are byte-identical to the previous
// 13-bit slave, so tb_top.v (21/21 bit-exact) is unaffected.

module avalon_slave (
    input  wire        clk,
    input  wire        rst_n,

    // Avalon-MM Slave
    input  wire [13:0] avs_address,
    input  wire        avs_write,
    input  wire        avs_read,
    input  wire [31:0] avs_writedata,
    output reg  [31:0] avs_readdata,

    // Input SRAM write port
    output reg  [11:0] sram_wr_addr,
    output reg  [7:0]  sram_din,
    output reg         sram_we,

    // Control/status
    output reg         start,
    input  wire        busy,
    input  wire        done,
    input  wire [1:0]  result,

    // ── Weight load ports (to cp_engine) ────────────────────────────────────
    output reg         w_wr_en,
    output reg  [2:0]  w_wr_oc,
    output reg  [4:0]  w_wr_word,
    output reg  [39:0] w_wr_data,
    output reg         b_wr_en,
    output reg  [4:0]  b_wr_addr,
    output reg  [31:0] b_wr_data,

    // ── Weight load ports (to gap_fc_argmax) ────────────────────────────────
    output reg         fcw_wr_en,
    output reg  [5:0]  fcw_wr_addr,
    output reg  [31:0] fcw_wr_data,

    // ── Topology config (to cnn_controller + cp_engine, packed per-layer) ────
    // layer L occupies bit-slice [L*W +: W]. Reset = default Chapman topology.
    output reg  [15:0] cfg_in_ch,    // 4 × 4-bit  in_ch  per layer
    output reg  [31:0] cfg_cp_en,    // 4 × 8-bit  cp_en  per layer
    output reg  [19:0] cfg_nb,       // 4 × 5-bit  nb     per layer
    output reg  [19:0] cfg_base      // 4 × 5-bit  layer_base per layer
);

    reg done_latched;
    reg [31:0] w_lo;   // holds conv-weight low 32 bits between the lo/hi write pair

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            done_latched <= 1'b0;
            start        <= 1'b0;
            sram_we      <= 1'b0;
            sram_din     <= 8'h00;
            sram_wr_addr <= 12'h000;
            avs_readdata <= 32'h0;
            w_wr_en      <= 1'b0;
            w_wr_oc      <= 3'd0;
            w_wr_word    <= 5'd0;
            w_wr_data    <= 40'd0;
            w_lo         <= 32'd0;
            b_wr_en      <= 1'b0;
            b_wr_addr    <= 5'd0;
            b_wr_data    <= 32'd0;
            fcw_wr_en    <= 1'b0;
            fcw_wr_addr  <= 6'd0;
            fcw_wr_data  <= 32'd0;
            // Default Chapman topology (Conv1..4). Packed LSB = Conv1.
            cfg_in_ch    <= {4'd8, 4'd4, 4'd4, 4'd1};                 // L3,L2,L1,L0
            cfg_cp_en    <= {8'hFF, 8'hFF, 8'h0F, 8'h0F};
            cfg_nb       <= {5'd7, 5'd6, 5'd6, 5'd8};
            cfg_base     <= {5'd9, 5'd5, 5'd1, 5'd0};
        end else begin
            // 1-cycle strobes default low
            start     <= 1'b0;
            sram_we   <= 1'b0;
            w_wr_en   <= 1'b0;
            b_wr_en   <= 1'b0;
            fcw_wr_en <= 1'b0;

            if (done)
                done_latched <= 1'b1;

            if (avs_write) begin
                if (avs_address[13]) begin
                    // ── WEIGHT WINDOW ──────────────────────────────────────
                    case (avs_address[12:11])
                        2'b00: begin
                            // Conv weight: off = (word*8+oc)*2 + hi
                            // off[0]=hi, off[3:1]=oc, off[8:4]=word
                            if (avs_address[0]) begin
                                // hi half -> assemble 40-bit and pulse write
                                w_wr_data <= {avs_writedata[7:0], w_lo};
                                w_wr_oc   <= avs_address[3:1];
                                w_wr_word <= avs_address[8:4];
                                w_wr_en   <= 1'b1;
                            end else begin
                                // lo half -> latch
                                w_lo <= avs_writedata;
                            end
                        end
                        2'b01: begin
                            // Conv bias: 1 write per INT32 entry
                            b_wr_addr <= avs_address[4:0];
                            b_wr_data <= avs_writedata;
                            b_wr_en   <= 1'b1;
                        end
                        2'b10: begin
                            // FC weight/bias
                            fcw_wr_addr <= avs_address[5:0];
                            fcw_wr_data <= avs_writedata;
                            fcw_wr_en   <= 1'b1;
                        end
                        2'b11: begin
                            // TOPOLOGY CONFIG: layer=addr[1:0], field=addr[3:2]
                            case (avs_address[3:2])
                                2'd0: cfg_in_ch[avs_address[1:0]*4 +: 4] <= avs_writedata[3:0];
                                2'd1: cfg_cp_en[avs_address[1:0]*8 +: 8] <= avs_writedata[7:0];
                                2'd2: cfg_nb   [avs_address[1:0]*5 +: 5] <= avs_writedata[4:0];
                                2'd3: cfg_base [avs_address[1:0]*5 +: 5] <= avs_writedata[4:0];
                            endcase
                        end
                    endcase
                end else if (avs_address[12]) begin
                    // ── ECG DATA WINDOW: one word = one SRAM byte ──
                    sram_din     <= avs_writedata[7:0];
                    sram_wr_addr <= avs_address[11:0];
                    sram_we      <= 1'b1;
                end else begin
                    // ── Low registers (legacy byte-at-a-time path) ──
                    case (avs_address[2:0])
                        3'h0: sram_din     <= avs_writedata[7:0];
                        3'h1: sram_wr_addr <= avs_writedata[11:0];
                        3'h2: sram_we      <= avs_writedata[0];
                        3'h3: begin
                            start        <= avs_writedata[0];
                            done_latched <= 1'b0;
                        end
                    endcase
                end
            end

            if (avs_read)
                case ({avs_address[12], avs_address[2:0]})
                    4'h4: avs_readdata <= {30'b0, done_latched, busy};
                    4'h5: avs_readdata <= {30'b0, result};
                    default: avs_readdata <= 32'b0;
                endcase
        end
    end

endmodule
