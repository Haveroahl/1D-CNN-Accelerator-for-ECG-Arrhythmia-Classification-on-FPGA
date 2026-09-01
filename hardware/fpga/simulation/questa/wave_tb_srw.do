# wave_tb_srw.do — GUI: vsim -gui -do wave_tb_srw.do
# Xem SRW truot: srw_flat[0..4] (slot0=moi nhat, slot4=cu nhat) + mux_comb[0..4].
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

add wave -divider {CLOCK / CTRL}
add wave sim:/tb_srw/clk
add wave sim:/tb_srw/srw_rst
add wave sim:/tb_srw/shift_en
add wave sim:/tb_srw/compute_en
add wave -unsigned sim:/tb_srw/sram_rd_addr
add wave -decimal sim:/tb_srw/input_sram_dout

add wave -divider {SRW slots (slot0=moi nhat)}
add wave -decimal sim:/tb_srw/dut/srw_flat(0)
add wave -decimal sim:/tb_srw/dut/srw_flat(1)
add wave -decimal sim:/tb_srw/dut/srw_flat(2)
add wave -decimal sim:/tb_srw/dut/srw_flat(3)
add wave -decimal sim:/tb_srw/dut/srw_flat(4)

add wave -divider {MUX tap (oldest->newest, khop w[k])}
add wave -decimal sim:/tb_srw/dut/mux_comb(0)
add wave -decimal sim:/tb_srw/dut/mux_comb(1)
add wave -decimal sim:/tb_srw/dut/mux_comb(2)
add wave -decimal sim:/tb_srw/dut/mux_comb(3)
add wave -decimal sim:/tb_srw/dut/mux_comb(4)

add wave -divider {OUTPUT ch0}
add wave sim:/tb_srw/pong_we
add wave -decimal sim:/tb_srw/pong_din(7:0)

configure wave -timelineunits ns
echo "Loaded tb_srw. Go: run -all ; roi: wave zoom full"
