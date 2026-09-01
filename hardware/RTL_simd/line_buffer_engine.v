// line_buffer_engine.v
// 8 line-buffers (1 per input channel), each a depth-(L+4)=24 INT8 shift register.
// Provides the 20 static 5-tap windows for the selected input channel `a`.
//
// SIMD.md §4 / §7. Position-parallel: for a block starting at output position
// block_base, buffer[ic] holds input positions [block_base-2 .. block_base+21]
// (24 values). Lane l of output channel oc needs input window
// [ (block_base+l)-2 .. (block_base+l)+2 ] of channel a.
//
// SHIFT MODEL (matches production SRW convention — ascending stream, newest in
// slot 0): one new position shifts into slot 0 per `shift_en`; existing values move
// slot s → s+1; slot DEPTH-1 drops off. Positions stream in INCREASING order.
//   After streaming through position P:  slot j = position (P - j).
//   For block_base, the compute phase begins once position (block_base+21) is in,
//   so slot j = position (block_base+21-j).
//   Lane l (output t=block_base+l) tap k wants x[t-2+k]=x[block_base+l-2+k], held in
//   slot j with block_base+21-j = block_base+l-2+k  →  j = (DEPTH-1) - (l+k).
//   So: tap k of lane l = slot (DEPTH-1 - l - k).  k=0 (oldest x[t-2]) → high slot,
//       k=4 (newest x[t+2]) → low slot. Matches production (oldest from high slot).
//
// A block boundary occurs every 20 shifts. Priming before block 0: 24 shifts fill
// the buffer (left zero-pad for positions < 0). Between blocks, 20 shifts advance.
//
// PADDING (SIMD.md §7), one pad_zero signal shared by all 8 buffers (pad is by
// position, not channel):
//   PP4 left  pad: srw_rst clears buffers to 0; the first real position lands such
//                  that the two below-zero positions read as 0 (handled by the
//                  controller priming with pad_zero=1 for pos<0).
//   PP1 right pad: pad_zero=1 when the position being loaded is >= in_len → force 0.
//
// All 8 channels shift in parallel from din_ch[ic] (controller picks the source:
// Conv1 = input stream, Conv2-4 = ping-pong channel-major dout). pad_zero forces 0
// on every channel for that shift.

module line_buffer_engine #(
    parameter L     = 20,
    parameter DEPTH = 24   // L + 4
) (
    input  wire             clk,
    input  wire             rst,
    input  wire             srw_rst,        // clear all buffers (layer transition, PP4 left pad)

    // Per-channel shift-in data (8 channels packed), and a single shift strobe.
    input  wire [63:0]      din_ch,         // din_ch[ic*8 +: 8], INT8 per channel
    input  wire             shift_en,       // 1 = shift all buffers by one slot
    input  wire             pad_zero,       // 1 = force 0 into slot 0 this shift (PP1/PP4)

    // Wide-load path (SF-3, SIMD.md §6 lợi phụ): load 4 consecutive positions per
    // cycle from a pong-wide word, exactly equivalent to 4 back-to-back 1-pos shifts.
    // Used for Conv2-4 priming/slide (pong source); Conv1 keeps the 1-pos shift path.
    //   din4_ch[ic*32 +: 32] = word {pos3,pos2,pos1,pos0} (byte0=oldest pos0, byte3=
    //   newest pos3). After load: slot0=pos3, slot1=pos2, slot2=pos1, slot3=pos0; old
    //   slot s → s+4. pad4_zero[j] forces position j (j=0 oldest .. 3 newest) to 0 for
    //   PP1 right-pad of the last block. wide_load and shift_en are mutually exclusive.
    input  wire             wide_load,      // 1 = load 4 positions this cycle
    input  wire [255:0]     din4_ch,        // 8 ch × 32b word, din4_ch[ic*32 +: 32]
    input  wire [3:0]       pad4_zero,      // per-position right-pad mask (bit j = pos j)

    // Channel select for the tap output (the in-channel `a` being accumulated).
    input  wire [3:0]       a,

    // 20 lane × 5-tap windows of buffer[a], packed taps_out[l*40 + k*8 +: 8].
    output wire [L*40-1:0]  taps_out
);

    // 8 buffers × DEPTH slots × 8b. buf_flat[ic*DEPTH + slot].
    reg signed [7:0] buf_flat [0:8*DEPTH-1];

    integer ic, s;
    // per-position bytes of a wide word: byte j = pos j (j=0 oldest .. 3 newest)
    function [7:0] w4byte;
        input [31:0] word; input [1:0] j; input pz;
        w4byte = pz ? 8'sd0 : word[j*8 +: 8];
    endfunction
    always @(posedge clk) begin
        if (srw_rst) begin
            for (ic = 0; ic < 8*DEPTH; ic = ic + 1)
                buf_flat[ic] <= 8'sd0;
        end else if (wide_load) begin
            // load 4 positions = 4 sequential 1-pos shifts collapsed into one cycle.
            // newest (pos3) → slot0; oldest (pos0) → slot3; old slot s → s+4.
            for (ic = 0; ic < 8; ic = ic + 1) begin
                for (s = DEPTH-1; s >= 4; s = s - 1)
                    buf_flat[ic*DEPTH + s] <= buf_flat[ic*DEPTH + (s-4)];
                buf_flat[ic*DEPTH + 0] <= $signed(w4byte(din4_ch[ic*32 +: 32], 2'd3, pad4_zero[3]));
                buf_flat[ic*DEPTH + 1] <= $signed(w4byte(din4_ch[ic*32 +: 32], 2'd2, pad4_zero[2]));
                buf_flat[ic*DEPTH + 2] <= $signed(w4byte(din4_ch[ic*32 +: 32], 2'd1, pad4_zero[1]));
                buf_flat[ic*DEPTH + 3] <= $signed(w4byte(din4_ch[ic*32 +: 32], 2'd0, pad4_zero[0]));
            end
        end else if (shift_en) begin
            for (ic = 0; ic < 8; ic = ic + 1) begin
                // shift slot s → s+1 (newest enters slot 0)
                for (s = DEPTH-1; s > 0; s = s - 1)
                    buf_flat[ic*DEPTH + s] <= buf_flat[ic*DEPTH + (s-1)];
                buf_flat[ic*DEPTH + 0] <= pad_zero ? 8'sd0 : $signed(din_ch[ic*8 +: 8]);
            end
        end
    end

    // ── Static tap windows for selected channel a ────────────────────────────
    // lane l, tap k → buffer[a] slot (DEPTH-1 - l - k). Pure index select (no
    // dynamic MUX across slots); only dynamic select is channel `a` (8:1, 1 LUT lvl).
    genvar l, k;
    generate
        for (l = 0; l < L; l = l + 1) begin : lanes
            for (k = 0; k < 5; k = k + 1) begin : taps
                assign taps_out[l*40 + k*8 +: 8] =
                    buf_flat[a[2:0]*DEPTH + ((DEPTH-1) - l - k)];
            end
        end
    endgenerate

endmodule
