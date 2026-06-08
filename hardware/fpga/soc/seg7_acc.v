// seg7_acc.v
// ============================================================================
// Drives 3 of the DE10-Standard 7-segment displays (HEX2 HEX1 HEX0) to show an
// accuracy value 0..100 (%), fed as a 7-bit unsigned binary number from the HPS
// over a Qsys PIO. RTL does the binary→3-digit-BCD split and 7-seg decode, so
// the HPS just writes the raw number (no segment table on the software side).
//
//   acc_in = 0..100  → HEX2 = hundreds (0 or 1), HEX1 = tens, HEX0 = units
//   e.g. 94 → HEX2='0' HEX1='9' HEX0='4'; 100 → '1''0''0'
//
// Each HEX output is 7 bits, ACTIVE-LOW (segment on = 0), bit order {g,f,e,d,c,b,a}
// matching the Terasic DE10-Standard convention (HEXn[0]=a … HEXn[6]=g).
//
// Pure combinational. acc_in > 100 is clamped to 100 (defensive; HPS never sends >100).
// ============================================================================

module seg7_acc (
    input  wire [6:0] acc_in,    // accuracy 0..100 (binary), from PIO
    output wire [6:0] hex0,      // units   (active-low)
    output wire [6:0] hex1,      // tens
    output wire [6:0] hex2       // hundreds (0 or 1)
);

    // Clamp to 100 (defensive)
    wire [6:0] acc = (acc_in > 7'd100) ? 7'd100 : acc_in;

    // ── Binary → BCD (range 0..100, so hundreds is 0 or 1) ─────────────────
    wire [3:0] hundreds = (acc >= 7'd100) ? 4'd1 : 4'd0;
    wire [6:0] rem      = (acc >= 7'd100) ? (acc - 7'd100) : acc;   // 0..99 (0 when acc==100)
    wire [3:0] tens     = rem / 7'd10;
    wire [3:0] units    = rem % 7'd10;

    // ── 7-seg decode (active-low, {g,f,e,d,c,b,a}) ─────────────────────────
    function [6:0] seg7;
        input [3:0] d;
        begin
            case (d)
                4'd0: seg7 = 7'b1000000;
                4'd1: seg7 = 7'b1111001;
                4'd2: seg7 = 7'b0100100;
                4'd3: seg7 = 7'b0110000;
                4'd4: seg7 = 7'b0011001;
                4'd5: seg7 = 7'b0010010;
                4'd6: seg7 = 7'b0000010;
                4'd7: seg7 = 7'b1111000;
                4'd8: seg7 = 7'b0000000;
                4'd9: seg7 = 7'b0010000;
                default: seg7 = 7'b1111111;   // blank
            endcase
        end
    endfunction

    assign hex0 = seg7(units);
    assign hex1 = seg7(tens);
    assign hex2 = seg7(hundreds);

endmodule
