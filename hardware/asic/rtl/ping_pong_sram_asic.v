// ping_pong_sram_asic.v  —  ASIC (Sky130/OpenLane) variant of ping_pong_sram.
//
// Packed for SRAM-macro mapping: the FPGA version uses 16 separate 512x8 arrays
// (8 channels x 2 banks) so each fits one M10K. For ASIC we map each BANK to a
// single 512x64 SRAM macro (8 channels packed into one 64-bit word) -> 2 macros
// total instead of 16.
//
// Port-level behavior is IDENTICAL to ping_pong_sram.v (bit-exact):
//   - din/dout are 64-bit packed: [ch*8 +: 8]
//   - we is per-channel [7:0]   -> PER-BYTE write mask on the 64-bit word
//   - 1-cycle synchronous read, combinational bank-select mux on registered data
//
// Per-channel write is preserved via a byte write-mask: only channels whose we
// bit is set update their byte; inactive channels keep their old byte. This is
// exactly the FPGA semantics and maps to an SRAM macro with a byte write-mask
// (OpenRAM `wmask`). We do NOT read-modify-write in logic (that blew up area on
// the FPGA pack experiment); the macro's wmask does the masking internally. For
// pre-macro RTL simulation the per-byte updates below model wmask behavior.

module ping_pong_sram_asic (
    input  wire        clk,
    input  wire        bank_sel,   // 0: A=Ping B=Pong | 1: B=Ping A=Pong

    // Write port - Pong set (from cp_engine pool output)
    input  wire [8:0]  wr_addr,    // 0..499
    input  wire [63:0] din,        // 8 channels packed: din[ch*8+:8]
    input  wire [7:0]  we,         // per-channel write enable (= per-byte wmask)

    // Read port - Ping set (to cp_engine SRW / GAP engine)
    input  wire [8:0]  rd_addr,    // 0..499
    output reg  [63:0] dout        // 8 channels packed: dout[ch*8+:8]
);

    // ── 2 packed 512x64 arrays (one per bank) — map to 2 SRAM macros ───────
    reg [63:0] mem_a [0:511];
    reg [63:0] mem_b [0:511];

    reg [63:0] rd_a, rd_b;
    reg        bank_sel_d;

    // ── Bank-gated per-channel write enables (= per-byte wmask) ────────────
    // bank_sel=0 -> Pong=B, bank_sel=1 -> Pong=A (matches ping_pong_sram.v)
    wire [7:0] we_a = bank_sel ? we : 8'h00;
    wire [7:0] we_b = bank_sel ? 8'h00 : we;

    // ── Bank A: 512x64 macro with per-byte write mask, 1-cy sync read ──────
    integer ia;
    always @(posedge clk) begin
        for (ia = 0; ia < 8; ia = ia + 1)
            if (we_a[ia]) mem_a[wr_addr][ia*8 +: 8] <= din[ia*8 +: 8];
        rd_a <= mem_a[rd_addr];
    end

    // ── Bank B ─────────────────────────────────────────────────────────────
    integer ib;
    always @(posedge clk) begin
        for (ib = 0; ib < 8; ib = ib + 1)
            if (we_b[ib]) mem_b[wr_addr][ib*8 +: 8] <= din[ib*8 +: 8];
        rd_b <= mem_b[rd_addr];
    end

    // ── Bank_sel delay 1cy to align with sync-read latency ─────────────────
    always @(posedge clk) bank_sel_d <= bank_sel;

    // ── Output mux: combinational select on registered read data ──────────
    always @(*) begin
        dout = bank_sel_d ? rd_b : rd_a;
    end

endmodule
