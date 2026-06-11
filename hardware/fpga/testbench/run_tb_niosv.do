# run_tb_niosv.do — system simulation of Nios V/m + ECG accelerator
# Run from: hardware/fpga/soc/nios_system/simulation/mentor
#   vsim -c -do ../../../../../testbench/run_tb_niosv.do
#
# Uses the Qsys-generated msim_setup.tcl to compile every system submodule
# (Nios V core, interconnect, on-chip RAM with firmware, JTAG UART, ecg_core),
# then compiles the testbench and elaborates with the TB as top.

# 1. Bring in the Qsys-generated compile/elaborate flow.
source msim_setup.tcl

# 2. Compile all IP + system sources (libraries, then the design).
dev_com
com

# 2b. The ecg_core submodules $readmemh() their weight/bias .hex relative to the
#     simulation CWD (this mentor/ dir). Qsys only copies onchip RAM + Nios mifs,
#     NOT the accelerator weights -> copy them here or every weight reads as X
#     and the CNN outputs class 0. (Root cause of the 0/3 mismatch.)
foreach h {conv1_w conv2_w conv3_w conv4_w conv_bias fc_weights fc_bias} {
    file copy -force D:/Thesis101/hardware/RTL/$h.hex ./$h.hex
}

# 3. Compile the system testbench into the work library.
vlog -work work D:/Thesis101/hardware/fpga/testbench/tb_niosv_system.v

# 4. Elaborate with the testbench as top (override the default nios_system top).
#    Use elab_debug (-voptargs=+acc) so the testbench can read internal memory
#    arrays (u_core.u_isram.mem[]) for the SRAM dump probe.
set TOP_LEVEL_NAME tb_niosv_system
elab_debug

# 5. Run until the firmware halts / watchdog fires.
run -all

quit -f
