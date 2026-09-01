transcript on
if {[file exists rtl_work]} {
	vdel -lib rtl_work -all
}
vlib rtl_work
vmap work rtl_work

# ── ASIC memory + core + sim wrapper (hardware/asic/rtl) ──────────────────
vlog -sv -work work {D:/Thesis101/hardware/asic/rtl/ping_pong_sram_asic.v}
vlog -sv -work work {D:/Thesis101/hardware/asic/rtl/input_sram_asic.v}
vlog -sv -work work {D:/Thesis101/hardware/asic/rtl/ecg_core_asic.v}
vlog -sv -work work {D:/Thesis101/hardware/asic/rtl/ecg_accelerator_top_asic.v}

# ── Reused technology-agnostic logic (hardware/RTL) ───────────────────────
vlog -sv -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/cp_block.v}
vlog -sv -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/cnn_controller.v}
vlog -sv -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/avalon_slave.v}
vlog -sv -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/gap_fc_argmax.v}
vlog -sv -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/cp_engine.v}

# ── ASIC testbench ────────────────────────────────────────────────────────
vlog -sv -work work {D:/Thesis101/hardware/asic/sim/tb_top_asic.v}

# Run from hardware/fpga/simulation/questa so $readmemh finds conv*_w.hex,
# golden/, ecg_sample*.hex (this .do is invoked with that as the cwd).
vsim -t 1ps -L rtl_work -L work -voptargs="+acc" tb_top_asic

run -all
