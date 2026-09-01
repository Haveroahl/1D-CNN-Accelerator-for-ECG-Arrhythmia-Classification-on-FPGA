// simd_weight_rom.v
// Weight + bias ROM for the SIMD core. Unlike production cp_engine (which fans 8 oc
// in parallel), the SIMD core processes ONE output channel at a time, so this ROM
// selects a SINGLE (oc, a) entry combinationally.
//
// Weight hex layout is IDENTICAL to production (SIMD.md §12):
//   conv1_w.hex [0:3]   4 oc × 1 ic   addr = oc
//   conv2_w.hex [0:15]  4 oc × 4 ic   addr = oc*4 + a
//   conv3_w.hex [0:31]  8 oc × 4 ic   addr = oc*4 + a
//   conv4_w.hex [0:63]  8 oc × 8 ic   addr = oc*8 + a
//   each line = 40b packed tap4..tap0 (MSB→LSB); w_packed[tap*8+:8] = tap.
// Bias conv_bias.hex [0:31] INT32 LE, addr = oc*4 + layer_idx (Conv1..4 → 0..3).
//
// CONTRACT (matches simd_lane_array): w_packed and bias_cur are COMBINATIONAL from
// (layer_state, oc, a). The lane array registers w once (w_lane) to align with its
// internal taps_s1 register; bias is used directly at the acc edge and is constant
// across the a-sweep for a given oc, so a combinational read is stable.

module simd_weight_rom (
    input  wire        clk,           // for $readmemh init only (arrays are FF)
    input  wire [2:0]  layer_state,   // CONV1=2..CONV4=5
    input  wire [3:0]  oc,
    input  wire [3:0]  a,
    output wire [39:0] w_packed,
    output wire signed [31:0] bias_cur
);
    localparam CONV1=3'd2, CONV2=3'd3, CONV3=3'd4, CONV4=3'd5;

    reg [39:0] w_rom_conv1 [0:3];
    reg [39:0] w_rom_conv2 [0:15];
    reg [39:0] w_rom_conv3 [0:31];
    reg [39:0] w_rom_conv4 [0:63];
    (* ramstyle = "MLAB" *) reg signed [31:0] b_store [0:31];

    initial begin
        $readmemh("conv1_w.hex", w_rom_conv1);
        $readmemh("conv2_w.hex", w_rom_conv2);
        $readmemh("conv3_w.hex", w_rom_conv3);
        $readmemh("conv4_w.hex", w_rom_conv4);
        $readmemh("conv_bias.hex", b_store);
    end

    wire [1:0] layer_idx = layer_state[1:0] - 2'd2;  // CONV1→0 .. CONV4→3

    assign w_packed =
        (layer_state==CONV1) ? w_rom_conv1[oc[1:0]]                :
        (layer_state==CONV2) ? w_rom_conv2[{oc[1:0], a[1:0]}]      :
        (layer_state==CONV3) ? w_rom_conv3[{oc[2:0], a[1:0]}]      :
        (layer_state==CONV4) ? w_rom_conv4[{oc[2:0], a[2:0]}]      : 40'd0;

    assign bias_cur = b_store[{oc[2:0], layer_idx}];
endmodule
