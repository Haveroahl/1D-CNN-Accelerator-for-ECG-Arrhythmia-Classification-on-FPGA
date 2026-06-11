// tb_niosv_system.v
// ============================================================================
// System-level testbench for the Nios V/m + ECG accelerator SoC (Phase D, stage 2).
//
// This drives NOTHING on the Avalon bus itself — the Nios V/m soft-core, booting
// from on-chip RAM (initialised with the compiled main.c firmware), is the master.
// The testbench only provides clock + reset and lets the CPU run. main.c loads the
// three embedded ECG samples into the accelerator, runs inference, and prints the
// predicted class through the JTAG UART (which appears on the ModelSim console via
// the JTAG UART simulation FIFO model).
//
// Pass criterion: the firmware prints "Result: 3/3 match golden" (classes 3/1/2),
// observed on the console. As an independent check we also probe the core's result
// register through the design hierarchy and report each inference.
//
// Run (from hardware/fpga/soc/nios_system/simulation/mentor):
//   vsim -c -do ../../../../testbench/run_tb_niosv.do
// ============================================================================
`timescale 1ns/1ps

module tb_niosv_system;

    reg clk = 1'b0;
    reg rst_n = 1'b0;

    // 100 MHz core clock (10 ns period)
    always #5 clk = ~clk;

    wire irq;   // jtag_uart irq export (left open)

    // ── DUT: the generated Qsys system ──────────────────────────────────────
    // Ports (from nios_system/synthesis/nios_system.v):
    //   clk_clk, reset_reset_n, ecg_reset_h_reset, intel_niosv_m_0platform_irq_rx_irq
    nios_system u_sys (
        .clk_clk                          (clk),
        .reset_reset_n                    (rst_n),
        .ecg_reset_h_reset                (~rst_n),   // core sync active-high reset
        .intel_niosv_m_0platform_irq_rx_irq (irq)
    );

    // ── Reset release ────────────────────────────────────────────────────────
    initial begin
        rst_n = 1'b0;
        repeat (20) @(posedge clk);
        rst_n = 1'b1;
        $display("[TB] reset released, Nios V booting from on-chip RAM...");
    end

    // ── Probe the Avalon transactions reaching the core (debug) ─────────────
    // Hierarchy: u_sys.ecg_core_0 (= ecg_accelerator_top) .avs_* ports.
    wire [4:0]  p_addr  = u_sys.ecg_core_0.avs_address;
    wire        p_write = u_sys.ecg_core_0.avs_write;
    wire        p_read  = u_sys.ecg_core_0.avs_read;
    wire [31:0] p_wdata = u_sys.ecg_core_0.avs_writedata;
    wire [31:0] p_rdata = u_sys.ecg_core_0.avs_readdata;
    // core internal 8-wire interface (inside ecg_accelerator_top: u_avs -> u_core)
    wire        p_we    = u_sys.ecg_core_0.sram_we;
    wire        p_start = u_sys.ecg_core_0.start;
    wire        p_busy  = u_sys.ecg_core_0.busy;
    wire        p_done  = u_sys.ecg_core_0.done;
    wire [1:0]  p_res   = u_sys.ecg_core_0.result;
    wire        p_rst   = u_sys.ecg_core_0.rst;     // core sync reset (ecg_reset_h)
    wire        p_rst_n = u_sys.ecg_core_0.rst_n;   // bus async reset-n

    // Track start pulse width + busy behaviour around the first inference.
    integer start_cycles = 0;
    always @(posedge clk) if (rst_n && p_start) start_cycles = start_cycles + 1;
    reg p_start_d = 0, p_busy_d = 0;
    always @(posedge clk) begin
        p_start_d <= p_start; p_busy_d <= p_busy;
        if (rst_n && p_start && !p_start_d)
            $display("[START edge t=%0t] core_rst=%b rst_n=%b busy=%b", $time, p_rst, p_rst_n, p_busy);
        if (rst_n && p_busy && !p_busy_d)
            $display("[BUSY rise t=%0t]", $time);
        if (rst_n && !p_busy && p_busy_d)
            $display("[BUSY fall t=%0t] result=%0d", $time, p_res);
    end

    integer we_count = 0;
    always @(posedge clk) if (rst_n && p_we) we_count = we_count + 1;

    // Report only START (word 3) writes + done edges + running we count.
    always @(posedge clk) if (rst_n && p_write && p_addr == 5'h03)
        $display("[CORE START t=%0t] we_count(bytes loaded)=%0d", $time, we_count);
    reg p_done_d = 1'b0;
    always @(posedge clk) begin
        p_done_d <= p_done;
        if (rst_n && p_done && !p_done_d)
            $display("[CORE DONE t=%0t] result=%0d", $time, p_res);
    end

    // On the first START, dump the first/last few input_sram bytes so we can
    // compare against golden ecg_sample0 (00, FD, FE, ... at index 0,1,2).
    reg dumped = 1'b0;
    always @(posedge clk) if (rst_n && p_start && !dumped) begin
        dumped <= 1'b1;
        $display("[SRAM DUMP] mem[0..5] = %02h %02h %02h %02h %02h %02h",
            u_sys.ecg_core_0.u_core.u_isram.mem[0],
            u_sys.ecg_core_0.u_core.u_isram.mem[1],
            u_sys.ecg_core_0.u_core.u_isram.mem[2],
            u_sys.ecg_core_0.u_core.u_isram.mem[3],
            u_sys.ecg_core_0.u_core.u_isram.mem[4],
            u_sys.ecg_core_0.u_core.u_isram.mem[5]);
        $display("[SRAM DUMP] mem[2497..2499] = %02h %02h %02h",
            u_sys.ecg_core_0.u_core.u_isram.mem[2497],
            u_sys.ecg_core_0.u_core.u_isram.mem[2498],
            u_sys.ecg_core_0.u_core.u_isram.mem[2499]);
    end

    // ── Watchdog: the firmware ends in an infinite halt loop after printing.
    // Three inferences over 2500-byte loads at 3 writes/byte ≈ a few hundred k
    // CPU instructions + 3×5216 accelerator cycles. Give it generous headroom. ──
    initial begin
        #20_000_000;   // 20 ms sim time
        $display("[TB] watchdog timeout — stopping.");
        $stop;
    end

endmodule
