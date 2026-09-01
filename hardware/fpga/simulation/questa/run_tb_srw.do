transcript on
if {[file exists rtl_work]} { vdel -lib rtl_work -all }
vlib rtl_work
vmap work rtl_work

vlog -sv -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/cp_mac.v}
vlog -sv -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/cp_accumulate_rescale.v}
vlog -sv -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/cp_pool.v}
vlog -sv -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/cp_block.v}
vlog -sv -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/cp_weight_store.v}
vlog -sv -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/cp_engine.v}
vlog -sv -work work +incdir+D:/Thesis101/hardware/testbench {D:/Thesis101/hardware/testbench/tb_srw.v}

vsim -t 1ps -L rtl_work -L work -voptargs="+acc" tb_srw
run -all
