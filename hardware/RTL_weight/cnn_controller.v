// cnn_controller.v
// Unified FSM controlling the entire CNN accelerator pipeline.
// Drives cp_engine (Conv1..4) and gap_fc_argmax (GAP/FC/Argmax) sequentially.
//
// NB values per layer (from QAT calibration — python_log.txt):
//   NB1=8 (Conv1), NB2=6 (Conv2), NB3=6 (Conv3), NB4=7 (Conv4)

module cnn_controller (
    input  wire        clk,
    input  wire        rst,       // synchronous reset
    input  wire        start,     // 1-cycle pulse from avalon_slave

    // pool_write from cp_engine (any active cp_block)
    // All active blocks write simultaneously — use ch0 as representative
    input  wire        pool_write,

    // ── Topology config (packed per-layer, from avalon_slave) ──────────
    // Layer L = bit-slice [L*W +: W]. L0=Conv1 .. L3=Conv4. Reset defaults
    // in avalon_slave reproduce the original (1,4,4,8)/(0F,0F,FF,FF)/(8,6,6,7).
    input  wire [15:0] cfg_in_ch,    // 4 × 4-bit
    input  wire [31:0] cfg_cp_en,    // 4 × 8-bit
    input  wire [19:0] cfg_nb,       // 4 × 5-bit

    // Outputs to cp_engine
    output reg  [3:0]  a,             // channel counter 0..IN_CH-1
    output reg  [11:0] t,             // output position counter
    output wire        shift_en,      // = (a == in_ch - 1)
    output reg         srw_rst,       // SRW clear pulse (1 cycle)
    output reg         compute_en,    // pipeline enable
    output reg  [3:0]  in_ch,         // current layer IN_CH
    output reg  [11:0] in_len,        // current layer IN_LEN (Conv1=2500, Conv2=500, Conv3=100, Conv4=20)
    output reg  [3:0]  nb,            // rescale shift (0..15; max used = 8)
    output reg         relu_en,       // ReLU enable (Conv4 only)
    output reg  [7:0]  cp_en,         // active output channel bitmask
    output reg         bank_sel,      // Ping/Pong bank selector
    output reg  [11:0] pong_addr,     // Pong SRAM write address
    output reg         pool_rst,      // reset pool_cnt on layer transition

    // Outputs to gap_fc_argmax
    output reg  [2:0]  fc_sub_state,
    output reg  [3:0]  gap_step,
    output reg  [3:0]  fc_step,
    output reg  [1:0]  argmax_step,

    // Input from gap_fc_argmax
    input  wire [1:0]  argmax_result,

    // Top-level outputs
    output reg  [2:0]  layer_state,   // exposed for cp_engine MUX and top-level
    output reg         busy,          // 1 while not IDLE/DONE
    output reg         done,          // 1-cycle pulse
    output reg  [1:0]  result         // latched argmax class index
);

    // ── FSM state encoding ─────────────────────────────────────────────────
    localparam IDLE       = 3'd0,
               LOAD_INPUT = 3'd1,
               CONV1      = 3'd2,
               CONV2      = 3'd3,
               CONV3      = 3'd4,
               CONV4      = 3'd5,
               GAP_FC_S   = 3'd6,
               DONE_S     = 3'd7;

    // GAP_FC sub-states (match gap_fc_argmax.v)
    localparam GAP_SUB    = 3'd1,
               FC_SUB     = 3'd2,
               FC_FLUSH_S = 3'd3,
               ARGMAX_SUB = 3'd4,
               DONE_SUB   = 3'd5;

    // NB / in_ch / cp_en are now runtime-config (see cfg_* ports + accessors
    // below). Default Chapman values (nb=8,6,6,7) come from avalon_slave reset.

    // ── Per-layer config accessors (li = 0..3 for Conv1..4) ────────────────
    // Replaces the old hard-coded in_ch / cp_en / nb literals. The transition
    // block selects li for the NEXT layer; LOAD_INPUT uses li=0 (Conv1).
    function [3:0] cfg_in_ch_of; input [1:0] li; cfg_in_ch_of = cfg_in_ch[li*4 +: 4]; endfunction
    function [7:0] cfg_cp_en_of; input [1:0] li; cfg_cp_en_of = cfg_cp_en[li*8 +: 8]; endfunction
    // cfg_nb is packed 5-bit/layer in avalon_slave; nb max used = 8 → take low 4 bits.
    function [3:0] cfg_nb_of;    input [1:0] li; cfg_nb_of    = cfg_nb   [li*5 +: 4]; endfunction

    // ── Derived signals ────────────────────────────────────────────────────
    // IN_LEN per layer: Conv1=2500, Conv2=500, Conv3=100, Conv4=20
    // rp = t - 2 (read position with padding offset of 2)
    // sram_rd_addr issued when sram_addr_en=1: addr = rp = t - 2
    // For pre-fetch and boundary: controller issues addr, srw_din=0 if out of range (handled in cp_engine)

    assign shift_en    = (a == in_ch - 4'd1);

    // sram_rd_addr: output position t minus padding (2 taps ahead)
    // Exposed as wire so top-level can route to both input_sram and ping_pong_sram
    // The cp_engine uses t to track output position; read address = t - 2 + tap_offset
    // Simplified: controller outputs t; cp_engine or top computes actual rd_addr
    // Here: output t directly, let top-level wire t as rd_addr base
    // (cp_engine handles out-of-range → srw_din=0 for padding)

    // ── layer_done ─────────────────────────────────────────────────────────
    // pong_addr holds current write address (increments AFTER pool_write non-blocking).
    // layer_done fires on the LAST pool_write: pong_addr == out_len-1.
    // The transition block resets pong_addr=0, so no conflict with the increment.
    reg  [11:0] out_len;
    wire        layer_done = (pong_addr == out_len - 12'd1) && pool_write;

    // Pre-fetch counter: gates compute_en until pipeline (6 stages) + SRW prime.
    // Per controller_fsm.md:64-68 — count 6 shift_en pulses before raising compute_en.
    reg [2:0]   prefetch_cnt;
    // gap_fc_done removed — was unused (FSM transitions via fc_sub_state == DONE_SUB directly)

    // ── Main FSM ───────────────────────────────────────────────────────────
    always @(posedge clk) begin
        if (rst) begin
            layer_state  <= IDLE;
            done         <= 1'b0;
            busy         <= 1'b0;
            result       <= 2'b0;
            compute_en   <= 1'b0;
            srw_rst      <= 1'b0;
            pool_rst     <= 1'b0;
            cp_en        <= 8'h00;
            bank_sel     <= 1'b0;
            a            <= 4'd0;
            t            <= 12'd0;
            in_len       <= 12'd0;
            pong_addr    <= 12'd0;
            fc_sub_state <= 3'd0;
            gap_step     <= 4'd0;
            fc_step      <= 4'd0;
            argmax_step  <= 2'd0;
            prefetch_cnt <= 3'd0;
        end else begin
            // Defaults (overridden below)
            srw_rst  <= 1'b0;
            pool_rst <= 1'b0;
            done     <= 1'b0;

            case (layer_state)
                // ── IDLE ───────────────────────────────────────────────
                IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        layer_state <= LOAD_INPUT;
                        busy        <= 1'b1;
                    end
                end

                // ── LOAD_INPUT ─────────────────────────────────────────
                // Input SRAM already loaded by HPS before start pulse.
                // Immediately transition to CONV1.
                LOAD_INPUT: begin
                    layer_state  <= CONV1;
                    in_ch        <= cfg_in_ch_of(2'd0);
                    in_len       <= 12'd2500;
                    out_len      <= 12'd500;
                    nb           <= cfg_nb_of(2'd0);
                    relu_en      <= 1'b0;
                    cp_en        <= cfg_cp_en_of(2'd0);
                    bank_sel     <= 1'b0;
                    a            <= 4'd0;
                    t            <= 12'd0;
                    pong_addr    <= 12'd0;
                    compute_en   <= 1'b0;   // pre-fetch first; raised after 6 shift_en pulses
                    prefetch_cnt <= 3'd0;
                    srw_rst      <= 1'b1;
                    pool_rst     <= 1'b1;
                end

                // ── CONV1..4 ───────────────────────────────────────────
                CONV1, CONV2, CONV3, CONV4: begin
                    // a counter: 0..IN_CH-1
                    if (a == in_ch - 4'd1)
                        a <= 4'd0;
                    else
                        a <= a + 4'd1;

                    // t counter: increment when a wraps (= shift_en)
                    if (shift_en)
                        t <= t + 12'd1;

                    // pong_addr: increment on pool_write
                    // On layer_done (pool_write at addr out_len-1), the transition block
                    // below resets pong_addr=0, overriding this increment (non-blocking
                    // last-assignment-wins). That is correct: no more writes needed.
                    if (pool_write)
                        pong_addr <= pong_addr + 12'd1;

                    // Pre-fetch: count REAL SRW shifts (gated by !srw_rst).
                    // Need 5 shifts to fill SRW correctly:
                    //   shift 0: slot0 ← pad (t=0 < 2)
                    //   shift 1: slot0 ← pad (t=1 < 2)
                    //   shift 2: slot0 ← data[0]
                    //   shift 3: slot0 ← data[1]
                    //   shift 4: slot0 ← data[2]
                    // After 5 shifts, SRW = [data[2],data[1],data[0],0,0].
                    // MUX a=0 → [0,0,data[0],data[1],data[2]] = output position 0. ✓
                    //
                    // srw_rst cycle: shift_en may be 1 (Conv1) but SRW does NOT shift.
                    // Must gate prefetch_cnt by !srw_rst to avoid over-counting.
                    if (!compute_en && shift_en && !srw_rst) begin
                        if (prefetch_cnt == 3'd4)
                            compute_en <= 1'b1;
                        else
                            prefetch_cnt <= prefetch_cnt + 3'd1;
                    end

                    // Layer transition
                    if (layer_done) begin
                        srw_rst      <= 1'b1;
                        pool_rst     <= 1'b1;
                        compute_en   <= 1'b0;
                        prefetch_cnt <= 3'd0;
                        a            <= 4'd0;
                        t            <= 12'd0;
                        pong_addr    <= 12'd0;
                        bank_sel     <= ~bank_sel;

                        case (layer_state)
                            CONV1: begin
                                layer_state <= CONV2;
                                in_ch       <= cfg_in_ch_of(2'd1);
                                in_len      <= 12'd500;
                                out_len     <= 12'd100;
                                nb          <= cfg_nb_of(2'd1);
                                relu_en     <= 1'b0;
                                cp_en       <= cfg_cp_en_of(2'd1);
                            end
                            CONV2: begin
                                layer_state <= CONV3;
                                in_ch       <= cfg_in_ch_of(2'd2);
                                in_len      <= 12'd100;
                                out_len     <= 12'd20;
                                nb          <= cfg_nb_of(2'd2);
                                relu_en     <= 1'b0;
                                cp_en       <= cfg_cp_en_of(2'd2);
                            end
                            CONV3: begin
                                layer_state <= CONV4;
                                in_ch       <= cfg_in_ch_of(2'd3);
                                in_len      <= 12'd20;
                                out_len     <= 12'd4;
                                nb          <= cfg_nb_of(2'd3);
                                relu_en     <= 1'b1;
                                cp_en       <= cfg_cp_en_of(2'd3);
                            end
                            CONV4: begin
                                layer_state  <= GAP_FC_S;
                                cp_en        <= 8'h00;
                                fc_sub_state <= GAP_SUB;
                                gap_step     <= 4'd0;
                            end
                            default: ;
                        endcase
                    end
                end

                // ── GAP_FC_S ───────────────────────────────────────────
                GAP_FC_S: begin
                    case (fc_sub_state)
                        GAP_SUB: begin
                            gap_step <= gap_step + 4'd1;
                            if (gap_step == 4'd5) begin
                                fc_sub_state <= FC_SUB;
                                fc_step      <= 4'd0;
                            end
                        end
                        FC_SUB: begin
                            fc_step <= fc_step + 4'd1;
                            if (fc_step == 4'd9)
                                fc_sub_state <= FC_FLUSH_S;
                        end
                        FC_FLUSH_S: begin
                            fc_sub_state <= ARGMAX_SUB;
                            argmax_step  <= 2'd0;
                        end
                        ARGMAX_SUB: begin
                            argmax_step <= argmax_step + 2'd1;
                            if (argmax_step == 2'd3)
                                fc_sub_state <= DONE_SUB;
                        end
                        DONE_SUB: begin
                            layer_state <= DONE_S;
                            done        <= 1'b1;
                            result      <= argmax_result;
                        end
                        default: ;
                    endcase
                end

                // ── DONE_S ─────────────────────────────────────────────
                // Hold result until next rst or start. Accept new start pulse
                // to enable streaming use case (back-to-back windows without rst).
                DONE_S: begin
                    busy <= 1'b0;
                    if (start) begin
                        layer_state <= LOAD_INPUT;
                        busy        <= 1'b1;
                    end
                end

                default: layer_state <= IDLE;
            endcase
        end
    end

endmodule
