// tb_timing.v — Per-layer cycle breakdown (Chương 4, Bảng 4.13)
//
// Không kiểm tra giá trị (đã có tb_bitexact1 lo bit-exactness). Việc duy nhất:
// đếm số chu kỳ đồng hồ mà máy trạng thái ở mỗi lớp, để đối chiếu số chu kỳ
// đo được với số chu kỳ lý thuyết in_len × in_ch.
//
// Chạy: vsim -c -do run_tb_timing.do

`timescale 1ns/1ps

module tb_timing;

    reg        clk, rst, rst_n;
    reg [13:0] avs_address;
    reg        avs_write, avs_read;
    reg [31:0] avs_writedata;
    wire [31:0] avs_readdata;

    ecg_accelerator_top u_top (
        .clk(clk), .rst(rst), .rst_n(rst_n),
        .avs_address(avs_address), .avs_write(avs_write), .avs_read(avs_read),
        .avs_writedata(avs_writedata), .avs_readdata(avs_readdata)
    );

    wire [2:0] layer_state  = u_top.u_core.ctrl_layer_state;
    wire [2:0] fc_sub_state = u_top.u_core.ctrl_fc_sub_state;

    localparam IDLE = 3'd0, LOAD_INPUT = 3'd1, CONV1 = 3'd2, CONV2 = 3'd3,
               CONV3 = 3'd4, CONV4 = 3'd5, GAP_FC_S = 3'd6, DONE_S = 3'd7;
    localparam GAP_SUB = 3'd1, FC_SUB = 3'd2, FC_FLUSH_S = 3'd3,
               ARGMAX_SUB = 3'd4, DONE_SUB = 3'd5;

    // ── Cycle counters ────────────────────────────────────────────────
    integer c_load, c_conv1, c_conv2, c_conv3, c_conv4, c_gapfc, c_done;
    integer c_gap, c_fc, c_flush, c_argmax, c_total;
    reg     counting;

    initial begin
        c_load=0; c_conv1=0; c_conv2=0; c_conv3=0; c_conv4=0; c_gapfc=0;
        c_done=0; c_gap=0; c_fc=0; c_flush=0; c_argmax=0; c_total=0;
        counting=0;
    end

    always @(posedge clk) if (!rst && counting) begin
        c_total = c_total + 1;
        case (layer_state)
            LOAD_INPUT: c_load  = c_load  + 1;
            CONV1:      c_conv1 = c_conv1 + 1;
            CONV2:      c_conv2 = c_conv2 + 1;
            CONV3:      c_conv3 = c_conv3 + 1;
            CONV4:      c_conv4 = c_conv4 + 1;
            GAP_FC_S: begin
                c_gapfc = c_gapfc + 1;
                case (fc_sub_state)
                    GAP_SUB:    c_gap    = c_gap    + 1;
                    FC_SUB:     c_fc     = c_fc     + 1;
                    FC_FLUSH_S: c_flush  = c_flush  + 1;
                    ARGMAX_SUB: c_argmax = c_argmax + 1;
                    default: ;
                endcase
            end
            DONE_S:     c_done  = c_done  + 1;
            default: ;
        endcase
    end

    initial clk = 0;
    always #5 clk = ~clk;

    task avs_wr;
        input [4:0] addr; input [31:0] data;
        begin
            @(negedge clk);
            avs_address = addr; avs_writedata = data; avs_write = 1;
            @(posedge clk); #1; avs_write = 0;
        end
    endtask

    task avs_rd;
        input [4:0] addr; output [31:0] data;
        begin
            @(negedge clk);
            avs_address = addr; avs_read = 1;
            @(posedge clk); #1; data = avs_readdata; avs_read = 0;
        end
    endtask

    task load_ecg_hex;
        input [255:0] filename;
        reg [7:0] ecg [0:2499];
        integer i;
        begin
            $readmemh(filename, ecg);
            for (i = 0; i < 2500; i = i + 1) begin
                avs_wr(5'h00, {24'h0, ecg[i]});
                avs_wr(5'h01, i[31:0]);
                avs_wr(5'h02, 32'd1);
            end
            @(posedge clk); #1;
        end
    endtask

    integer poll;
    reg [31:0] status;

    initial begin
        rst = 1; rst_n = 0; avs_write = 0; avs_read = 0;
        avs_address = 0; avs_writedata = 0;
        @(posedge clk); @(posedge clk); #1;
        rst = 0; rst_n = 1;
        @(posedge clk); #1;

        load_ecg_hex("ecg_sample0.hex");

        // START → đếm cho tới khi hết busy
        counting = 1;
        avs_wr(5'h03, 32'd1);
        @(posedge clk); #1;
        status = 1; poll = 0;
        while (status[0] && poll < 20000) begin
            @(posedge clk); #1;
            avs_rd(5'h04, status);
            poll = poll + 1;
        end
        counting = 0;

        $display("");
        $display("=== PHAN TICH SO CHU KY THEO TUNG LOP ===");
        $display("%-22s %10s %10s", "Trang thai", "Do chu ky", "Ly thuyet");
        $display("%-22s %10d %10s", "LOAD_INPUT",  c_load,  "-");
        $display("%-22s %10d %10d", "CONV1 + pool", c_conv1, 2500);
        $display("%-22s %10d %10d", "CONV2 + pool", c_conv2, 2000);
        $display("%-22s %10d %10d", "CONV3 + pool", c_conv3,  400);
        $display("%-22s %10d %10d", "CONV4 + pool", c_conv4,  160);
        $display("%-22s %10d %10d", "GAP/FC/ARGMAX", c_gapfc,  22);
        $display("     - GAP           %10d", c_gap);
        $display("     - FC            %10d", c_fc);
        $display("     - FC_FLUSH      %10d", c_flush);
        $display("     - ARGMAX        %10d", c_argmax);
        $display("%-22s %10d %10s", "DONE",        c_done,  "-");
        $display("%-22s %10d %10d", "TONG",        c_total, 5082);
        $display("");
        $display("Do tre @100MHz : %0d ns", c_total * 10);
        $finish;
    end

endmodule
