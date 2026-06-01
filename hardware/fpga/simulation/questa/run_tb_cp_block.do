transcript on
if {[file exists rtl_work]} {
	vdel -lib rtl_work -all
}
vlib rtl_work
vmap work rtl_work

vlog -sv -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/cp_block.v}
vlog -sv -work work +incdir+D:/Thesis101/hardware/../testbench {D:/Thesis101/hardware/testbench/tb_cp_block.v}

vsim -t 1ps -L rtl_work -L work -voptargs="+acc" tb_cp_block
run -all
