# wave_tb_cp_block_simple.do — GUI: vsim -gui -do wave_tb_cp_block_simple.do
transcript on
if {[file exists rtl_work]} { vdel -lib rtl_work -all }
vlib rtl_work
vmap work rtl_work

vlog -sv -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/cp_mac.v}
vlog -sv -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/cp_accumulate_rescale.v}
vlog -sv -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/cp_pool.v}
vlog -sv -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/cp_block.v}
vlog -sv -work work +incdir+D:/Thesis101/hardware/testbench {D:/Thesis101/hardware/testbench/tb_cp_block_simple.v}

vsim -t 1ps -L rtl_work -L work -voptargs="+acc" tb_cp_block_simple

add wave -divider {CLOCK}
add wave sim:/tb_cp_block_simple/clk
add wave sim:/tb_cp_block_simple/rst

add wave -divider {INPUT (5 tap x + weight)}
add wave -hex sim:/tb_cp_block_simple/x_in
add wave -hex sim:/tb_cp_block_simple/w
add wave sim:/tb_cp_block_simple/a_in
add wave sim:/tb_cp_block_simple/compute_en_in
add wave sim:/tb_cp_block_simple/nb

add wave -divider {MAC (S1-S4)}
add wave -decimal sim:/tb_cp_block_simple/dut/u_mac/tree_out

add wave -divider {ACC / RESCALE (S5-S8)}
add wave -decimal sim:/tb_cp_block_simple/dut/u_accres/acc
add wave sim:/tb_cp_block_simple/dut/relu_v
add wave -decimal sim:/tb_cp_block_simple/dut/u_accres/relu_out

add wave -divider {POOL (S9) -> OUTPUT}
add wave sim:/tb_cp_block_simple/dut/u_pool/pool_cnt
add wave sim:/tb_cp_block_simple/pool_write
add wave -decimal sim:/tb_cp_block_simple/pool_out

configure wave -timelineunits ns
echo "Loaded tb_cp_block_simple. Go: run -all ; roi: wave zoom full"
