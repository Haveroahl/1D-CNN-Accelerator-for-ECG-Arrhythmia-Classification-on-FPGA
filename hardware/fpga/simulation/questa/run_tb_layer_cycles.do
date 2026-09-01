transcript on
if {[file exists rtl_work]} {
	vdel -lib rtl_work -all
}
vlib rtl_work
vmap work rtl_work

# RTL/ is now the ROM single-load build (weights baked via $readmemh, fixed
# Chapman topology, no bus/cfg ports). tb_top is compiled with +define+WEIGHT_ROM
# so its bus/config tests (TC08-TC10) are compiled out; the 21/21 bit-exact
# per-layer checks run against the ROM datapath.
vlog -sv -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/ping_pong_sram.v}
vlog -sv -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/input_sram.v}
vlog -sv -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/ecg_accelerator_top.v}
vlog -sv -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/ecg_core.v}
vlog -sv -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/cp_mac.v}
vlog -sv -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/cp_accumulate_rescale.v}
vlog -sv -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/cp_pool.v}
vlog -sv -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/cp_block.v}
vlog -sv -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/cnn_controller.v}
vlog -sv -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/avalon_slave.v}
vlog -sv -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/gap_unit.v}
vlog -sv -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/fc_unit.v}
vlog -sv -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/argmax_unit.v}
vlog -sv -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/gap_fc_argmax.v}
vlog -sv -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/cp_weight_store.v}
vlog -sv -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/cp_engine.v}

vlog -sv -work work +define+WEIGHT_ROM +incdir+D:/Thesis101/hardware/fpga/../testbench {D:/Thesis101/hardware/fpga/../testbench/tb_layer_cycles.v}

vsim -t 1ps -L altera_ver -L lpm_ver -L sgate_ver -L altera_mf_ver -L altera_lnsim_ver -L cyclonev_ver -L cyclonev_hssi_ver -L cyclonev_pcie_hip_ver -L rtl_work -L work -voptargs="+acc"  tb_layer_cycles

run -all
