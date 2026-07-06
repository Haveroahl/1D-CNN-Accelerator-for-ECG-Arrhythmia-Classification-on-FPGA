# Coverage run — tb_layer (integration)
transcript on
if {[file exists cov_work]} { vdel -lib cov_work -all }
vlib cov_work
vmap work cov_work

vlog -sv +cover=bcefsx -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/ping_pong_sram.v}
vlog -sv +cover=bcefsx -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/input_sram.v}
vlog -sv +cover=bcefsx -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/ecg_accelerator_top.v}
vlog -sv +cover=bcefsx -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/ecg_core.v}
vlog -sv +cover=bcefsx -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/cp_block.v}
vlog -sv +cover=bcefsx -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/cnn_controller.v}
vlog -sv +cover=bcefsx -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/avalon_slave.v}
vlog -sv +cover=bcefsx -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/gap_unit.v}
vlog -sv +cover=bcefsx -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/fc_unit.v}
vlog -sv +cover=bcefsx -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/argmax_unit.v}
vlog -sv +cover=bcefsx -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/gap_fc_argmax.v}
vlog -sv +cover=bcefsx -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/cp_engine.v}
vlog -sv -work work +incdir+D:/Thesis101/hardware/../testbench {D:/Thesis101/hardware/testbench/tb_layer.v}

# -onfinish stop: testbench calls $finish; keep sim alive so coverage save runs.
vsim -coverage -onfinish stop -t 1ps -L altera_ver -L lpm_ver -L sgate_ver -L altera_mf_ver -L altera_lnsim_ver -L cyclonev_ver -L cyclonev_hssi_ver -L cyclonev_pcie_hip_ver -L cov_work -L work -voptargs="+acc +cover=bcefsx" tb_layer
run -all
coverage save cov_layer.ucdb
quit -f
