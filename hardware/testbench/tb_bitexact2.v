// tb_bitexact2.v — Bit-exact checkpoint test cho tập kiểm tra chéo GEORGIA
//
// Giống tb_bitexact1 nhưng MẶC ĐỊNH = Georgia (SAMPLE=8): zero-shot trên cùng
// weight ROM ningba (không reload). So 7 checkpoint bit-exact với golden Python:
//   input_int8, after_pool1..4, after_gap, logits_fc  → 7/7.
//
// Mặc định SAMPLE=8 (Georgia). Vẫn đổi được qua plusarg:
//   vsim ... +SAMPLE=9   → ningba;  +SAMPLE=0/1/2 → Chapman cũ (nếu ROM Chapman)
//
// Requires (đường dẫn tương đối từ thư mục sim):
//   RTL weight/bias hex (qua $readmemh trong RTL, bản ROM cố định)
//   testbench/ecg_sample<N>.hex        — INT8 ECG, 2500 entries
//   golden/sample<N>/{input_int8,after_pool1..4,after_gap,logits_fc}.mem
// Sinh golden: software/python/generate_golden.py

`timescale 1ns/1ps

module tb_bitexact2;

    // ── DUT signals ───────────────────────────────────────────────────
    reg        clk, rst, rst_n;
    reg [13:0] avs_address;
    reg        avs_write, avs_read;
    reg [31:0] avs_writedata;
    wire [31:0] avs_readdata;

    ecg_accelerator_top u_top (
        .clk          (clk),
        .rst          (rst),
        .rst_n        (rst_n),
        .avs_address  (avs_address),
        .avs_write    (avs_write),
        .avs_read     (avs_read),
        .avs_writedata(avs_writedata),
        .avs_readdata (avs_readdata)
    );

    // ── Hierarchical aliases ──────────────────────────────────────────
    wire [2:0]  layer_state  = u_top.u_core.ctrl_layer_state;
    wire [2:0]  fc_sub_state = u_top.u_core.ctrl_fc_sub_state;

    // FSM state encoding
    localparam IDLE       = 3'd0;
    localparam CONV1      = 3'd2;
    localparam CONV2      = 3'd3;
    localparam CONV3      = 3'd4;
    localparam CONV4      = 3'd5;
    localparam GAP_FC_S   = 3'd6;

    // Transition-detection registers for the auto-trigger block
    reg [2:0] prev_layer_state;
    reg [2:0] prev_fc_sub_state;
    reg [3:0] prev_gap_step;
    always @(posedge clk) begin
        if (rst) begin
            prev_layer_state  <= 3'd0;
            prev_fc_sub_state <= 3'd0;
            prev_gap_step     <= 4'd0;
        end else begin
            prev_layer_state  <= layer_state;
            prev_fc_sub_state <= fc_sub_state;
            prev_gap_step     <= u_top.u_core.ctrl_gap_step;
        end
    end

    // ── Clock ─────────────────────────────────────────────────────────
    initial clk = 0;
    always #5 clk = ~clk;

    // ── Avalon-MM helpers ─────────────────────────────────────────────
    task avs_wr;
        input [4:0]  addr;
        input [31:0] data;
        begin
            @(negedge clk);
            avs_address   = addr;
            avs_writedata = data;
            avs_write     = 1;
            @(posedge clk); #1;
            avs_write = 0;
        end
    endtask

    task avs_rd;
        input  [4:0]  addr;
        output [31:0] data;
        begin
            @(negedge clk);
            avs_address = addr;
            avs_read    = 1;
            @(posedge clk); #1;
            data     = avs_readdata;
            avs_read = 0;
        end
    endtask

    // Probe: giá trị ECG signed đang nạp, để xem SÓNG ECG chạy theo thời gian
    // trên Wave (add wave -analog sim:/tb_bitexact2/ecg_probe). Mảng SRAM không
    // vẽ được thành sóng — cần tín hiệu đơn [7:0] signed thay đổi theo thời gian.
    reg signed [7:0] ecg_probe;

    // Load ECG from hex file (2500 × INT8) via Avalon-MM
    task load_ecg_hex;
        input [255:0] filename;
        reg [7:0] ecg [0:2499];
        integer i;
        begin
            $readmemh(filename, ecg);
            for (i = 0; i < 2500; i = i + 1) begin
                ecg_probe = $signed(ecg[i]);       // sóng ECG chạy theo thời gian
                avs_wr(5'h00, {24'h0, ecg[i]});  // DATA_IN
                avs_wr(5'h01, i[31:0]);            // ADDR_IN
                avs_wr(5'h02, 32'd1);              // WR_EN
            end
            @(posedge clk); #1;   // let final WR_EN pulse commit to input_sram
        end
    endtask

    // Run full inference: START → poll STATUS busy → return class + cycles
    task run_inference;
        output [1:0] cls;
        output integer cycles;
        reg [31:0] status;
        time         t_start, t_end;
        integer      poll_iter;
        begin
            avs_wr(5'h03, 32'd1);   // START
            @(posedge clk); #1;
            t_start   = $time;
            poll_iter = 0;
            status    = 1;
            while (status[0] && poll_iter < 10000) begin
                @(posedge clk); #1;
                avs_rd(5'h04, status);  // STATUS[0]=busy
                poll_iter = poll_iter + 1;
            end
            t_end  = $time;
            cycles = (t_end - t_start) / 10;
            avs_rd(5'h05, status);   // RESULT
            cls = status[1:0];
        end
    endtask

    task apply_reset;
        begin
            rst = 1; rst_n = 0;
            avs_write = 0; avs_read = 0;
            @(posedge clk); @(posedge clk); #1;
            rst = 0; rst_n = 1;
            @(posedge clk); #1;
        end
    endtask

    // ── Golden buffers ────────────────────────────────────────────────
    reg [7:0]  gold_input   [0:2499];
    reg [7:0]  gold_pool1   [0:1999];
    reg [7:0]  gold_pool2   [0:399];
    reg [7:0]  gold_pool3   [0:159];
    reg [7:0]  gold_pool4   [0:31];
    reg [7:0]  gold_gap     [0:7];
    reg [31:0] gold_logits  [0:3];

    integer l2_pass_cnt, l2_fail_cnt;
    integer g_nonzero, g_total;
    reg signed [31:0] g_max_diff;

    // ── Per-channel ping-pong memory read ─────────────────────────────
    function [7:0] read_mem_a;
        input integer ch;
        input integer pos;
        begin
            case (ch)
                0: read_mem_a = u_top.u_core.u_pp.mem_a_ch0[pos];
                1: read_mem_a = u_top.u_core.u_pp.mem_a_ch1[pos];
                2: read_mem_a = u_top.u_core.u_pp.mem_a_ch2[pos];
                3: read_mem_a = u_top.u_core.u_pp.mem_a_ch3[pos];
                4: read_mem_a = u_top.u_core.u_pp.mem_a_ch4[pos];
                5: read_mem_a = u_top.u_core.u_pp.mem_a_ch5[pos];
                6: read_mem_a = u_top.u_core.u_pp.mem_a_ch6[pos];
                7: read_mem_a = u_top.u_core.u_pp.mem_a_ch7[pos];
                default: read_mem_a = 8'h00;
            endcase
        end
    endfunction

    function [7:0] read_mem_b;
        input integer ch;
        input integer pos;
        begin
            case (ch)
                0: read_mem_b = u_top.u_core.u_pp.mem_b_ch0[pos];
                1: read_mem_b = u_top.u_core.u_pp.mem_b_ch1[pos];
                2: read_mem_b = u_top.u_core.u_pp.mem_b_ch2[pos];
                3: read_mem_b = u_top.u_core.u_pp.mem_b_ch3[pos];
                4: read_mem_b = u_top.u_core.u_pp.mem_b_ch4[pos];
                5: read_mem_b = u_top.u_core.u_pp.mem_b_ch5[pos];
                6: read_mem_b = u_top.u_core.u_pp.mem_b_ch6[pos];
                7: read_mem_b = u_top.u_core.u_pp.mem_b_ch7[pos];
                default: read_mem_b = 8'h00;
            endcase
        end
    endfunction

    // ── Checkpoint compare tasks (tol +/-10; deviation tracked exactly) ──
    task check_input;
        integer i, mismatches, first_bad, first_rtl, first_gold;
        reg signed [9:0] diff;
        reg signed [8:0] rtl_v, gold_v;
        begin
            mismatches = 0; first_bad = -1; first_rtl = 0; first_gold = 0;
            for (i = 0; i < 2500; i = i + 1) begin
                rtl_v  = $signed({u_top.u_isram.mem[i][7], u_top.u_isram.mem[i]});
                gold_v = $signed({gold_input[i][7], gold_input[i]});
                diff = rtl_v - gold_v;
                g_total = g_total + 1;
                if (diff != 0) g_nonzero = g_nonzero + 1;
                if (diff  >  g_max_diff) g_max_diff =  diff;
                if (-diff >  g_max_diff) g_max_diff = -diff;
                if (diff > 10 || diff < -10) begin
                    mismatches = mismatches + 1;
                    if (first_bad < 0) begin first_bad = i; first_rtl = rtl_v; first_gold = gold_v; end
                end
            end
            if (mismatches == 0) begin
                $display("[STAGE PASS] input_int8 (2500 elems)");
                l2_pass_cnt = l2_pass_cnt + 1;
            end else begin
                $display("[STAGE FAIL] input_int8: %0d/2500 mismatches, first @ idx=%0d rtl=%0d gold=%0d",
                         mismatches, first_bad, first_rtl, first_gold);
                l2_fail_cnt = l2_fail_cnt + 1;
            end
        end
    endtask

    // Generic pool compare — nch channels × plen positions, golden stride = plen.
    task check_pool;
        input         bank;
        input integer nch;
        input integer plen;
        input [63:0]  name;     // short ascii label, e.g. "pool1"
        integer ch, pos, idx, mismatches, first_bad, first_rtl, first_gold;
        reg signed [9:0] diff;
        reg signed [8:0] rtl_v, gold_v;
        reg [7:0] raw, gold_raw;
        begin
            mismatches = 0; first_bad = -1; first_rtl = 0; first_gold = 0;
            for (ch = 0; ch < nch; ch = ch + 1) begin
                for (pos = 0; pos < plen; pos = pos + 1) begin
                    idx = ch * plen + pos;
                    raw = bank ? read_mem_b(ch, pos) : read_mem_a(ch, pos);
                    case (plen)
                        500: gold_raw = gold_pool1[idx];
                        100: gold_raw = gold_pool2[idx];
                        20:  gold_raw = gold_pool3[idx];
                        default: gold_raw = gold_pool4[idx];  // plen==4
                    endcase
                    rtl_v  = $signed({raw[7], raw});
                    gold_v = $signed({gold_raw[7], gold_raw});
                    diff = rtl_v - gold_v;
                    g_total = g_total + 1;
                    if (diff != 0) g_nonzero = g_nonzero + 1;
                    if (diff  >  g_max_diff) g_max_diff =  diff;
                    if (-diff >  g_max_diff) g_max_diff = -diff;
                    if (diff > 10 || diff < -10) begin
                        mismatches = mismatches + 1;
                        if (first_bad < 0) begin first_bad = idx; first_rtl = rtl_v; first_gold = gold_v; end
                    end
                end
            end
            if (mismatches == 0) begin
                $display("[STAGE PASS] %0s (%0d elems)", name, nch*plen);
                l2_pass_cnt = l2_pass_cnt + 1;
            end else begin
                $display("[STAGE FAIL] %0s: %0d/%0d mismatches, first @ ch=%0d pos=%0d rtl=%0d gold=%0d",
                         name, mismatches, nch*plen, first_bad/plen, first_bad%plen, first_rtl, first_gold);
                l2_fail_cnt = l2_fail_cnt + 1;
            end
        end
    endtask

    task check_gap;
        integer i, mismatches, first_bad, first_rtl, first_gold;
        reg signed [9:0] diff;
        reg signed [8:0] rtl_v, gold_v;
        begin
            mismatches = 0; first_bad = -1; first_rtl = 0; first_gold = 0;
            for (i = 0; i < 8; i = i + 1) begin
                rtl_v  = $signed(u_top.u_core.u_gfa.u_gap.gap_reg[i]);
                gold_v = $signed({gold_gap[i][7], gold_gap[i]});
                diff = rtl_v - gold_v;
                g_total = g_total + 1;
                if (diff != 0) g_nonzero = g_nonzero + 1;
                if (diff  >  g_max_diff) g_max_diff =  diff;
                if (-diff >  g_max_diff) g_max_diff = -diff;
                if (diff > 10 || diff < -10) begin
                    mismatches = mismatches + 1;
                    if (first_bad < 0) begin first_bad = i; first_rtl = rtl_v; first_gold = gold_v; end
                end
            end
            if (mismatches == 0) begin
                $display("[STAGE PASS] after_gap (8 elems)");
                l2_pass_cnt = l2_pass_cnt + 1;
            end else begin
                $display("[STAGE FAIL] after_gap: %0d/8 mismatches, first @ idx=%0d rtl=%0d gold=%0d",
                         mismatches, first_bad, first_rtl, first_gold);
                l2_fail_cnt = l2_fail_cnt + 1;
            end
        end
    endtask

    task check_logits;
        integer i, mismatches, first_bad;
        reg signed [31:0] rtl_v, gold_v, diff, first_rtl, first_gold;
        begin
            mismatches = 0; first_bad = -1; first_rtl = 0; first_gold = 0;
            for (i = 0; i < 4; i = i + 1) begin
                rtl_v  = $signed(u_top.u_core.u_gfa.u_fc.fc_acc[i]);
                gold_v = $signed(gold_logits[i]);
                diff = rtl_v - gold_v;
                g_total = g_total + 1;
                if (diff != 0) g_nonzero = g_nonzero + 1;
                if (diff  >  g_max_diff) g_max_diff =  diff;
                if (-diff >  g_max_diff) g_max_diff = -diff;
                if (diff > 10 || diff < -10) begin
                    mismatches = mismatches + 1;
                    if (first_bad < 0) begin first_bad = i; first_rtl = rtl_v; first_gold = gold_v; end
                end
            end
            if (mismatches == 0) begin
                $display("[STAGE PASS] logits_fc (4 elems)");
                l2_pass_cnt = l2_pass_cnt + 1;
            end else begin
                $display("[STAGE FAIL] logits_fc: %0d/4 mismatches, first @ idx=%0d rtl=%0d gold=%0d",
                         mismatches, first_bad, first_rtl, first_gold);
                l2_fail_cnt = l2_fail_cnt + 1;
            end
        end
    endtask

    // ── Auto-trigger checkpoint compares on FSM transitions ────────────
    //   after_pool1 → mem_b (bank=1)   after_pool2 → mem_a (bank=0)
    //   after_pool3 → mem_b (bank=1)   after_pool4 → mem_a (bank=0)
    reg verify_en;
    always @(posedge clk) begin
        if (verify_en) begin
            if (layer_state == CONV2 && prev_layer_state == CONV1) check_pool(1'b1, 4, 500, "pool1");
            if (layer_state == CONV3 && prev_layer_state == CONV2) check_pool(1'b0, 4, 100, "pool2");
            if (layer_state == CONV4 && prev_layer_state == CONV3) check_pool(1'b1, 8, 20,  "pool3");
            if (layer_state == GAP_FC_S && prev_layer_state == CONV4) check_pool(1'b0, 8, 4, "pool4");
            if (prev_gap_step == 4'd5 && fc_sub_state == 3'd2 && prev_fc_sub_state == 3'd1)
                check_gap;
            if (prev_fc_sub_state == 3'd3 && fc_sub_state == 3'd4)
                check_logits;
        end
    end

    // ── Main sequence ─────────────────────────────────────────────────
    reg [1:0]     cls_got;
    integer       cyc_got;
    integer       sample_n;
    reg [255:0]   ecg_file, gdir;

    initial begin
        l2_pass_cnt = 0; l2_fail_cnt = 0;
        g_nonzero = 0; g_total = 0; g_max_diff = 0;
        verify_en = 1'b0;
        avs_write = 0; avs_read = 0; avs_address = 0; avs_writedata = 0;

        // Default = ningba bit-exact sample (nb Conv2=7, khớp weight ROM ningba
        // của RTL/ hiện tại). SAMPLE=0/1/2 = golden Chapman cũ (nb Conv2=6) —
        // chỉ đúng nếu ROM là weight Chapman.
        if (!$value$plusargs("SAMPLE=%d", sample_n)) sample_n = 8;
        case (sample_n)
            0: begin ecg_file = "testbench/ecg_sample0.hex"; gdir = "golden/sample0"; end
            1: begin ecg_file = "testbench/ecg_sample1.hex"; gdir = "golden/sample1"; end
            2: begin ecg_file = "testbench/ecg_sample2.hex"; gdir = "golden/sample2"; end
            8: begin ecg_file = "testbench/ecg_georgia0.hex"; gdir = "golden/georgia_bitexact"; end
            default: begin ecg_file = "testbench/ecg_ningba0.hex"; gdir = "golden/ningba_bitexact"; end
        endcase

        $display("=== tb_bitexact2: 7-checkpoint bit-exact, sample %0d ===", sample_n);

        apply_reset;

        // Load golden for this sample
        $readmemh({gdir, "/input_int8.mem"},  gold_input);
        $readmemh({gdir, "/after_pool1.mem"}, gold_pool1);
        $readmemh({gdir, "/after_pool2.mem"}, gold_pool2);
        $readmemh({gdir, "/after_pool3.mem"}, gold_pool3);
        $readmemh({gdir, "/after_pool4.mem"}, gold_pool4);
        $readmemh({gdir, "/after_gap.mem"},   gold_gap);
        $readmemh({gdir, "/logits_fc.mem"},   gold_logits);

        // Load ECG, verify input SRAM, then run with auto-checks enabled
        load_ecg_hex(ecg_file);
        check_input;                       // checkpoint 1

        verify_en = 1'b1;                  // checkpoints 2-7 auto-trigger
        run_inference(cls_got, cyc_got);
        verify_en = 1'b0;

        $display("=== inference done: class=%0d, cycles=%0d ===", cls_got, cyc_got);
        $display("=== BIT-EXACT: %0d/7 checkpoints PASS, %0d FAIL ===", l2_pass_cnt, l2_fail_cnt);
        $display("=== DEVIATION vs Python golden: max|diff|=%0d LSB, %0d/%0d elems differ (%0d exact) ===",
                 g_max_diff, g_nonzero, g_total, g_total - g_nonzero);
        if (l2_fail_cnt == 0 && g_max_diff == 0)
            $display("ALL 7 CHECKPOINTS BIT-EXACT (max|diff|=0 LSB)");
        else if (l2_fail_cnt == 0)
            $display("ALL 7 CHECKPOINTS PASS within tol (max|diff|=%0d LSB)", g_max_diff);
        $finish;
    end

    // Timeout guard: one inference ~5200 cycles + Avalon load overhead
    initial begin
        #2000000;
        $display("TIMEOUT — exceeded 2ms simulation time");
        $finish;
    end

endmodule
