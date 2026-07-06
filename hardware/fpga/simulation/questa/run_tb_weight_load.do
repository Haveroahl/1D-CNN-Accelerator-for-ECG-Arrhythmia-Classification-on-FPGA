transcript on
if {[file exists rtl_work]} {
	vdel -lib rtl_work -all
}
vlib rtl_work
vmap work rtl_work

# NO_WEIGHT_INIT: skip $readmemh in cp_engine/gap_fc_argmax so weights MUST be
# loaded via the Avalon bus (tb_weight_load drives the weight window).
set DEF +define+NO_WEIGHT_INIT

vlog -sv $DEF -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/ping_pong_sram.v}
vlog -sv $DEF -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/input_sram.v}
vlog -sv $DEF -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/ecg_accelerator_top.v}
vlog -sv $DEF -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/ecg_core.v}
vlog -sv $DEF -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/cp_mac.v}
vlog -sv $DEF -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/cp_accumulate_rescale.v}
vlog -sv $DEF -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/cp_pool.v}
vlog -sv $DEF -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/cp_block.v}
vlog -sv $DEF -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/cnn_controller.v}
vlog -sv $DEF -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/avalon_slave.v}
vlog -sv $DEF -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/gap_unit.v}
vlog -sv $DEF -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/fc_unit.v}
vlog -sv $DEF -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/argmax_unit.v}
vlog -sv $DEF -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/gap_fc_argmax.v}
vlog -sv $DEF -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/cp_weight_store.v}
vlog -sv $DEF -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/cp_engine.v}

vlog -sv $DEF -work work +incdir+D:/Thesis101/hardware/../hardware/testbench {D:/Thesis101/hardware/testbench/tb_weight_load.v}

vsim -t 1ps -L altera_ver -L lpm_ver -L sgate_ver -L altera_mf_ver -L altera_lnsim_ver -L cyclonev_ver -L cyclonev_hssi_ver -L cyclonev_pcie_hip_ver -L rtl_work -L work -voptargs="+acc"  tb_weight_load

run -all
