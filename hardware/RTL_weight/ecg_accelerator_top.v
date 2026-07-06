// ecg_accelerator_top.v
// Thin wrapper: bus adapter (avalon_slave) + bus-agnostic core (ecg_core).
//
// Port list and behavior are IDENTICAL to the previous monolithic top, so
// tb_top.v and the Quartus project see no change. The only difference is
// internal: the core is now a separate module (ecg_core.v) reusable under
// other wrappers (e.g. soc_top.v for Phase D Qsys/HPS).
//
//   HPS ──avs_*──► avalon_slave ──8 wires──► ecg_core
//                  (bus adapter)             (datapath + FSM, verified)

module ecg_accelerator_top (
    input  wire        clk,
    input  wire        rst,         // synchronous reset (active high, for RTL)
    input  wire        rst_n,       // async active-low reset (for avalon_slave)

    // Avalon-MM Slave port (from HPS Lightweight bridge)
    input  wire [13:0] avs_address,
    input  wire        avs_write,
    input  wire        avs_read,
    input  wire [31:0] avs_writedata,
    output wire [31:0] avs_readdata
);

    // ── 8-wire core interface (driven/consumed by avalon_slave) ────────
    wire [11:0] sram_wr_addr;
    wire [7:0]  sram_din;
    wire        sram_we;
    wire        start;
    wire        busy;
    wire        done;
    wire [1:0]  result;

    // ── input_sram read port (core ↔ wrapper-resident input_sram) ──────
    wire [11:0] input_rd_addr;
    wire [7:0]  input_dout;

    // ── Weight-load wires (avalon_slave → ecg_core, Phase B01) ─────────
    wire        w_wr_en;
    wire [2:0]  w_wr_oc;
    wire [4:0]  w_wr_word;
    wire [39:0] w_wr_data;
    wire        b_wr_en;
    wire [4:0]  b_wr_addr;
    wire [31:0] b_wr_data;
    wire        fcw_wr_en;
    wire [5:0]  fcw_wr_addr;
    wire [31:0] fcw_wr_data;

    // ── Topology config (avalon_slave → ecg_core) ─────────────────────
    wire [15:0] cfg_in_ch;
    wire [31:0] cfg_cp_en;
    wire [19:0] cfg_nb;
    wire [19:0] cfg_base;

    // ── avalon_slave (bus adapter) ─────────────────────────────────────
    avalon_slave u_avs (
        .clk          (clk),
        .rst_n        (rst_n),
        .avs_address  (avs_address),
        .avs_write    (avs_write),
        .avs_read     (avs_read),
        .avs_writedata(avs_writedata),
        .avs_readdata (avs_readdata),
        .sram_wr_addr (sram_wr_addr),
        .sram_din     (sram_din),
        .sram_we      (sram_we),
        .start        (start),
        .busy         (busy),
        .done         (done),
        .result       (result),
        .w_wr_en      (w_wr_en),
        .w_wr_oc      (w_wr_oc),
        .w_wr_word    (w_wr_word),
        .w_wr_data    (w_wr_data),
        .b_wr_en      (b_wr_en),
        .b_wr_addr    (b_wr_addr),
        .b_wr_data    (b_wr_data),
        .fcw_wr_en    (fcw_wr_en),
        .fcw_wr_addr  (fcw_wr_addr),
        .fcw_wr_data  (fcw_wr_data),
        .cfg_in_ch    (cfg_in_ch),
        .cfg_cp_en    (cfg_cp_en),
        .cfg_nb       (cfg_nb),
        .cfg_base     (cfg_base)
    );

    // ── input_sram (input I/O buffer — host writes, core reads) ────────
    input_sram u_isram (
        .clk    (clk),
        .wr_addr(sram_wr_addr),
        .din    (sram_din),
        .we     (sram_we),
        .rd_addr(input_rd_addr),
        .dout   (input_dout)
    );

    // ── ecg_core (bus-agnostic accelerator) ────────────────────────────
    ecg_core u_core (
        .clk          (clk),
        .rst          (rst),
        .input_rd_addr(input_rd_addr),
        .input_dout   (input_dout),
        .start        (start),
        .busy         (busy),
        .done         (done),
        .result       (result),
        .w_wr_en      (w_wr_en),
        .w_wr_oc      (w_wr_oc),
        .w_wr_word    (w_wr_word),
        .w_wr_data    (w_wr_data),
        .b_wr_en      (b_wr_en),
        .b_wr_addr    (b_wr_addr),
        .b_wr_data    (b_wr_data),
        .fcw_wr_en    (fcw_wr_en),
        .fcw_wr_addr  (fcw_wr_addr),
        .fcw_wr_data  (fcw_wr_data),
        .cfg_in_ch    (cfg_in_ch),
        .cfg_cp_en    (cfg_cp_en),
        .cfg_nb       (cfg_nb),
        .cfg_base     (cfg_base)
    );

endmodule
