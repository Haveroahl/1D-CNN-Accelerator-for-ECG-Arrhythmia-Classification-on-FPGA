// tb_srw.v — TB minh hoa SRW (shift-register-window) hoat dong trong cp_engine
// ---------------------------------------------------------------------------
// Muc dich: cho thay SRW TRUOT that su. cp_engine chua 8 SRW 5-tap; o Conv1
// (in_ch=1) chi SRW[0] hoat dong, va MOI cycle shift_en=1 -> cua so truot 1
// mau moi cycle. Ta nap mot chuoi ECG ngan de doc "song" chay qua 5 slot.
//
// SRW la thanh phan cua cp_engine, KHONG phai cp_block. TB nay drive cp_engine
// standalone bang cac tin hieu controller toi thieu cho Conv1:
//   a=0 (in_ch=1) -> shift_en = (a==in_ch-1) = 1 moi cycle
//   sram_rd_addr_in = t (bo dem vi tri); cp_engine tru 2 (pad) noi bo
//   input_sram_dout = mau ECG tuong ung dia chi da doc (do TB gia lap SRAM)
//   compute_en bat sau khi SRW da moi (prime) 5 slot
//
// Quan sat tren GUI (wave_tb_srw.do):
//   dut/srw_flat[0..4]  -> 5 slot truot: slot0=moi nhat, slot4=cu nhat
//   dut/mux_comb[0..4]  -> 5 tap sau khi re-index (oldest->newest)
//   pong_din[7:0], pong_we[0] -> pool_out + pool_write cua channel 0
//
// Run:  vsim -c -do "do run_tb_srw.do; quit -f"
// GUI:  vsim -gui -do wave_tb_srw.do   (roi: run -all)

`timescale 1ns/1ps

module tb_srw;

    // ── DUT ports ───────────────────────────────────────────────────────
    reg         clk, rst;
    reg  [3:0]  a, in_ch;
    reg  [11:0] in_len;
    reg         srw_rst, compute_en;
    reg  [3:0]  nb;
    reg         relu_en;
    reg  [7:0]  cp_en;
    reg  [2:0]  layer_state;
    reg         pool_rst;
    reg  [7:0]  input_sram_dout;
    reg  [63:0] ping_dout;
    reg  [11:0] sram_rd_addr_in;

    wire [63:0] pong_din;
    wire [7:0]  pong_we;
    wire [11:0] sram_rd_addr;

    // shift_en = (a == in_ch-1). Voi in_ch=1, a=0 -> luon 1.
    wire shift_en = (a == in_ch - 4'd1);

    cp_engine dut (
        .clk            (clk),
        .rst            (rst),
        .a              (a),
        .in_ch          (in_ch),
        .in_len         (in_len),
        .shift_en       (shift_en),
        .srw_rst        (srw_rst),
        .compute_en     (compute_en),
        .nb             (nb),
        .relu_en        (relu_en),
        .cp_en          (cp_en),
        .layer_state    (layer_state),
        .pool_rst       (pool_rst),
        .input_sram_dout(input_sram_dout),
        .ping_dout      (ping_dout),
        .pong_din       (pong_din),
        .pong_we        (pong_we),
        .sram_rd_addr   (sram_rd_addr),
        .sram_rd_addr_in(sram_rd_addr_in)
    );

    // ── Clock ───────────────────────────────────────────────────────────
    initial clk = 0;
    always #5 clk = ~clk;

    // ── Gia lap Input SRAM ──────────────────────────────────────────────
    // Chuoi ECG ngan, de nhan dang khi truot qua SRW.
    localparam integer N = 20;
    reg signed [7:0] ecg [0:N-1];
    integer i;

    // input_sram_dout = du lieu tai dia chi sram_rd_addr (do cp_engine phat, da
    // tru 2). Mo phong SRAM 1-cycle: capture rd_addr, tra ve o cycle sau.
    reg [11:0] rd_addr_q;
    always @(posedge clk) rd_addr_q <= sram_rd_addr;
    always @(*) begin
        if (rd_addr_q < N) input_sram_dout = ecg[rd_addr_q];
        else               input_sram_dout = 8'sd0;
    end

    // ── t counter: dia chi vi tri dau ra (0..) ; sram_rd_addr_in = t ────────
    integer t;

    initial begin
        // ECG mau: mot xung nhon de de thay truot (0,0,10,40,20,5,0,...)
        for (i = 0; i < N; i = i + 1) ecg[i] = 8'sd0;
        ecg[2]  = 8'sd10;
        ecg[3]  = 8'sd40;   // dinh
        ecg[4]  = 8'sd20;
        ecg[5]  = 8'sd5;
        ecg[8]  = 8'sd30;
        ecg[9]  = 8'sd15;

        // Static config: Conv1
        in_ch = 4'd1; in_len = 12'd20; nb = 4'd6; relu_en = 1'b0;
        cp_en = 8'h0F; layer_state = 3'd2 /*CONV1*/;
        ping_dout = 64'd0;
        a = 4'd0; compute_en = 1'b0; sram_rd_addr_in = 12'd0; t = 0;

        // Reset + SRW clear
        rst = 1'b1; srw_rst = 1'b1; pool_rst = 1'b1;
        repeat (4) @(posedge clk); #1;
        rst = 1'b0;
        @(posedge clk); #1;
        srw_rst = 1'b0; pool_rst = 1'b0;

        $display("=== tb_srw: SRW truot qua %0d mau (Conv1, in_ch=1) ===", N);

        // Prime: chay SRW ~5 cycle truoc khi bat compute_en (giong prefetch).
        // Sau do bat compute_en va chay het chuoi.
        for (t = 0; t < N + 6; t = t + 1) begin
            sram_rd_addr_in = t[11:0];
            a = 4'd0;                         // in_ch=1 -> a luon 0, shift moi cycle
            if (t >= 5) compute_en = 1'b1;    // sau khi SRW moi day
            @(posedge clk); #1;
            $display("t=%0d rd_addr=%0d din=%0d | slot[0..4]=%0d %0d %0d %0d %0d | tap[0..4]=%0d %0d %0d %0d %0d | pong_we0=%0b pool_out=%0d",
                t, sram_rd_addr, $signed(input_sram_dout),
                $signed(dut.srw_flat[0]), $signed(dut.srw_flat[1]), $signed(dut.srw_flat[2]),
                $signed(dut.srw_flat[3]), $signed(dut.srw_flat[4]),
                $signed(dut.mux_comb[0]), $signed(dut.mux_comb[1]), $signed(dut.mux_comb[2]),
                $signed(dut.mux_comb[3]), $signed(dut.mux_comb[4]),
                pong_we[0], $signed(pong_din[7:0]));
        end

        // Drain
        compute_en = 1'b0;
        repeat (12) @(posedge clk); #1;

        $display("=== done (xem srw_flat truot tren GUI) ===");
        $finish;
    end

    initial begin
        #100000;
        $display("TIMEOUT");
        $finish;
    end

endmodule
