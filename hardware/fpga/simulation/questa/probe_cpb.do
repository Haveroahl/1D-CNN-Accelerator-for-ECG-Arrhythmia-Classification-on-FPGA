transcript on
if {[file exists rtl_work]} { vdel -lib rtl_work -all }
vlib rtl_work
vmap work rtl_work

set R D:/Thesis101/hardware/RTL
vlog -work work +incdir+$R $R/ping_pong_sram.v
vlog -work work +incdir+$R $R/input_sram.v
vlog -work work +incdir+$R $R/cp_weight_store.v
vlog -work work +incdir+$R $R/cp_mac.v
vlog -work work +incdir+$R $R/cp_accumulate_rescale.v
vlog -work work +incdir+$R $R/cp_pool.v
vlog -work work +incdir+$R $R/cp_block.v
vlog -work work +incdir+$R $R/cp_engine.v
vlog -work work +incdir+$R $R/cnn_controller.v
vlog -work work +incdir+$R $R/gap_unit.v
vlog -work work +incdir+$R $R/fc_unit.v
vlog -work work +incdir+$R $R/argmax_unit.v
vlog -work work +incdir+$R $R/gap_fc_argmax.v
vlog -work work +incdir+$R $R/avalon_slave.v
vlog -work work +incdir+$R $R/ecg_core.v
vlog -work work +incdir+$R $R/ecg_accelerator_top.v
vlog -work work +incdir+D:/Thesis101/hardware/testbench D:/Thesis101/hardware/testbench/tb_cpb_cycle_probe.v

vsim -t 1ps -L rtl_work -L work -voptargs="+acc" tb_cpb_cycle_probe
run -all
quit -f
