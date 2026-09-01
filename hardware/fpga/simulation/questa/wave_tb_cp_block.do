# wave_tb_cp_block.do — compile + load + add waves cho unit test cp_block, KHONG run.
# GUI: vsim -gui -do wave_tb_cp_block.do   (roi bam Run / go: run -all)
transcript on
if {[file exists rtl_work]} { vdel -lib rtl_work -all }
vlib rtl_work
vmap work rtl_work

# cp_block split -> can 3 submodule truoc
vlog -sv -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/cp_mac.v}
vlog -sv -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/cp_accumulate_rescale.v}
vlog -sv -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/cp_pool.v}
vlog -sv -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/cp_block.v}
vlog -sv -work work +incdir+D:/Thesis101/hardware/testbench {D:/Thesis101/hardware/testbench/tb_cp_block.v}

vsim -t 1ps -L rtl_work -L work -voptargs="+acc" tb_cp_block

# ── Waves ──
add wave -divider {CLOCK / RESET}
add wave sim:/tb_cp_block/clk
add wave sim:/tb_cp_block/rst

add wave -divider {INPUTS (DUT drive)}
add wave -hex sim:/tb_cp_block/taps_in
add wave -hex sim:/tb_cp_block/w
add wave -decimal sim:/tb_cp_block/bias_in
add wave sim:/tb_cp_block/a_in
add wave sim:/tb_cp_block/in_ch
add wave sim:/tb_cp_block/compute_en_in
add wave sim:/tb_cp_block/nb
add wave sim:/tb_cp_block/relu_en
add wave sim:/tb_cp_block/pool_rst

add wave -divider {MAC (S1-S4)}
add wave -decimal sim:/tb_cp_block/dut/u_mac/tree_out

add wave -divider {ACCUMULATE / RESCALE (S5-S8)}
add wave -decimal sim:/tb_cp_block/dut/u_accres/acc
add wave sim:/tb_cp_block/dut/relu_v
add wave -decimal sim:/tb_cp_block/dut/u_accres/relu_out

add wave -divider {POOL (S9)}
add wave sim:/tb_cp_block/dut/u_pool/pool_cnt
add wave sim:/tb_cp_block/pool_write
add wave -decimal sim:/tb_cp_block/pool_out

add wave -divider {CHECK LATCH}
add wave sim:/tb_cp_block/pw_seen
add wave -decimal sim:/tb_cp_block/pw_value

configure wave -timelineunits ns
echo "=============================================================="
echo " Loaded tb_cp_block. Bam nut Run (hoac go: run -all)."
echo " Xong -> Wave > Zoom Full (hoac: wave zoom full)."
echo "=============================================================="
