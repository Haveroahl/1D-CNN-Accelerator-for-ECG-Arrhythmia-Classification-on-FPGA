# wave_tb_layer.do — compile + load + add waves cho integration test Conv1, KHONG run.
# GUI: vsim -gui -do wave_tb_layer.do   (roi bam Run / go: run -all)
transcript on
if {[file exists rtl_work]} { vdel -lib rtl_work -all }
vlib rtl_work
vmap work rtl_work

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
vlog -sv -work work +incdir+D:/Thesis101/hardware/testbench {D:/Thesis101/hardware/testbench/tb_layer.v}

vsim -t 1ps -L altera_ver -L lpm_ver -L sgate_ver -L altera_mf_ver -L altera_lnsim_ver -L cyclonev_ver -L cyclonev_hssi_ver -L cyclonev_pcie_hip_ver -L rtl_work -L work -voptargs="+acc" tb_layer

# ── Waves ──
add wave -divider {CLOCK / RESET}
add wave sim:/tb_layer/clk
add wave sim:/tb_layer/rst

add wave -divider {AVALON BUS (nap ECG)}
add wave -hex sim:/tb_layer/avs_address
add wave      sim:/tb_layer/avs_write
add wave -hex sim:/tb_layer/avs_writedata

add wave -divider {FSM CONTROLLER}
add wave sim:/tb_layer/layer_state
add wave sim:/tb_layer/bank_sel
add wave sim:/tb_layer/srw_rst
add wave sim:/tb_layer/compute_en

add wave -divider {CONV1 -> POOL}
add wave      sim:/tb_layer/pool_write
add wave -unsigned sim:/tb_layer/pong_addr
add wave -hex sim:/tb_layer/cp_en
add wave -hex sim:/tb_layer/cp_pong_we

configure wave -timelineunits ns
echo "=============================================================="
echo " Loaded tb_layer. Bam nut Run (hoac go: run -all)."
echo " Xong -> Wave > Zoom Full (hoac: wave zoom full)."
echo "=============================================================="
