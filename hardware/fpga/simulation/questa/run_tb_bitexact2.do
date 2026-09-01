# run_tb_bitexact2.do — Georgia bit-exact (mac dinh SAMPLE=8) + add wave + run.
# GUI:  vsim -gui   roi Transcript: do run_tb_bitexact2.do   (song hien san)
# Batch: vsim -c -do "do run_tb_bitexact2.do; quit -f"
transcript on
if {[file exists rtl_work]} { vdel -lib rtl_work -all }
vlib rtl_work
vmap work rtl_work

# RTL/ = ban ROM single-load (weight ningba bake $readmemh). Georgia zero-shot
# tren cung weight nay -> khong reload. tb_bitexact2 mac dinh sample 8 (Georgia).
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
vlog -sv -work work +incdir+D:/Thesis101/hardware/testbench {D:/Thesis101/hardware/testbench/tb_bitexact2.v}

vsim -t 1ps -L altera_ver -L lpm_ver -L sgate_ver -L altera_mf_ver -L altera_lnsim_ver -L cyclonev_ver -L cyclonev_hssi_ver -L cyclonev_pcie_hip_ver -L rtl_work -L work -voptargs="+acc" tb_bitexact2

# ── Add waves (hien truoc khi run) ──
add wave -divider {CLOCK / RESET}
add wave sim:/tb_bitexact2/clk
add wave sim:/tb_bitexact2/rst

add wave -divider {AVALON BUS}
add wave -hex sim:/tb_bitexact2/avs_address
add wave      sim:/tb_bitexact2/avs_write
add wave -hex sim:/tb_bitexact2/avs_writedata

add wave -divider {ECG INPUT (signed) — Format>Analog(custom) Min -20 Max 20}
add wave -radix decimal sim:/tb_bitexact2/ecg_probe

add wave -divider {FSM CONTROLLER}
add wave sim:/tb_bitexact2/layer_state
add wave sim:/tb_bitexact2/fc_sub_state
add wave sim:/tb_bitexact2/u_top/u_core/busy
add wave sim:/tb_bitexact2/u_top/u_core/done
add wave sim:/tb_bitexact2/u_top/u_core/result

add wave -divider {CP-ENGINE (Conv/Pool)}
add wave      sim:/tb_bitexact2/u_top/u_core/cp_pool_write
add wave -hex sim:/tb_bitexact2/u_top/u_core/u_cpe/cp_pool_out

add wave -divider {GAP / FC}
add wave -hex sim:/tb_bitexact2/u_top/u_core/u_gfa/u_gap/gap_reg
add wave -hex sim:/tb_bitexact2/u_top/u_core/u_gfa/u_fc/fc_acc

add wave -divider {VERIFY FLAG}
add wave sim:/tb_bitexact2/verify_en

configure wave -timelineunits ns

run -all
wave zoom full
