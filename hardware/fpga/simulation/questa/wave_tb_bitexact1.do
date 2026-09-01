# wave_tb_bitexact1.do — compile + load + add waves, KHÔNG run.
# Dùng cho GUI: vsim -gui -do wave_tb_bitexact1.do
# Sau khi load xong, tự bấm nút Run (hoặc gõ `run -all`) để chạy và xem sóng.
transcript on
if {[file exists rtl_work]} { vdel -lib rtl_work -all }
vlib rtl_work
vmap work rtl_work

# ── Compile RTL (bản ROM single-load) ──
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
vlog -sv -work work +incdir+D:/Thesis101/hardware/testbench {D:/Thesis101/hardware/testbench/tb_bitexact1.v}

# ── Load (chưa run) ──
vsim -t 1ps -L altera_ver -L lpm_ver -L sgate_ver -L altera_mf_ver -L altera_lnsim_ver -L cyclonev_ver -L cyclonev_hssi_ver -L cyclonev_pcie_hip_ver -L rtl_work -L work -voptargs="+acc" tb_bitexact1

# ── Add waves (nhóm theo tầng) ──
add wave -divider {CLOCK / RESET}
add wave -position insertpoint sim:/tb_bitexact1/clk
add wave -position insertpoint sim:/tb_bitexact1/rst
add wave -position insertpoint sim:/tb_bitexact1/rst_n

add wave -divider {AVALON BUS}
add wave -hex sim:/tb_bitexact1/avs_address
add wave      sim:/tb_bitexact1/avs_write
add wave      sim:/tb_bitexact1/avs_read
add wave -hex sim:/tb_bitexact1/avs_writedata
add wave -hex sim:/tb_bitexact1/avs_readdata

add wave -divider {ECG INPUT (song, INT8 signed)}
# ecg_probe = gia tri ECG dang nap, tin hieu don [7:0] signed.
# De xem dang song: click phai -> Format -> Analog (custom), Min -20 Max 20.
add wave -radix decimal sim:/tb_bitexact1/ecg_probe

add wave -divider {FSM CONTROLLER}
add wave sim:/tb_bitexact1/layer_state
add wave sim:/tb_bitexact1/fc_sub_state
add wave sim:/tb_bitexact1/u_top/u_core/busy
add wave sim:/tb_bitexact1/u_top/u_core/done
add wave sim:/tb_bitexact1/u_top/u_core/result

add wave -divider {CP-ENGINE (Conv/Pool)}
add wave      sim:/tb_bitexact1/u_top/u_core/cp_pool_write
add wave -hex sim:/tb_bitexact1/u_top/u_core/u_cpe/cp_pool_out

add wave -divider {GAP / FC}
add wave -hex sim:/tb_bitexact1/u_top/u_core/u_gfa/u_gap/gap_reg
add wave -hex sim:/tb_bitexact1/u_top/u_core/u_gfa/u_fc/fc_acc

add wave -divider {VERIFY FLAG}
add wave sim:/tb_bitexact1/verify_en

# Zoom hợp lý; run tay bằng nút Run hoặc `run -all` trong Transcript.
configure wave -timelineunits ns
echo "=============================================================="
echo " Loaded. Bấm nút Run (hoặc gõ: run -all) để chạy mô phỏng."
echo " Sau khi chạy xong, bấm Wave > Zoom Full (hoặc gõ: wave zoom full)."
echo "=============================================================="
