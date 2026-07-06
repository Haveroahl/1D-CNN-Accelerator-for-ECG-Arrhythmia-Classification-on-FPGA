# Requires golden in topo_golden/<tag>/ — generate first with:
#   cd software/python && python gen_topo_golden.py \
#     --ecg ../../hardware/fpga/simulation/questa/ecg_sample0.hex \
#     --output_dir ../../hardware/fpga/simulation/questa/topo_golden
# Plus a topo_golden/chapman/ dir (after_gap.mem + logits_fc.mem copied from the
# real golden/sample0) for the harness self-check.
transcript on
if {[file exists rtl_work]} { vdel -lib rtl_work -all }
vlib rtl_work
vmap work rtl_work

# NO_WEIGHT_INIT: skip default Chapman $readmemh — tb_topo loads topo weights.
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

vlog -sv $DEF -work work +incdir+D:/Thesis101/hardware/testbench {D:/Thesis101/hardware/testbench/tb_topo.v}

vsim -t 1ps -L rtl_work -L work -voptargs="+acc" tb_topo
run -all
