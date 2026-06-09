# run_tb_gate.do — gate-level (post-synthesis) functional simulation.
#
# Drives the synthesized netlist ecg_accelerator_top.vo (Quartus EDA Netlist
# Writer output) with the black-box testbench tb_gate.v. Verifies the post-synth
# netlist produces the correct argmax class + deterministic latency for the 3
# reference samples.
#
# This is FUNCTIONAL gate-level sim (zero-delay). SDF timing back-annotation is
# not supported for Cyclone V under Quartus Prime Lite (EDA writer warning
# 10905) — timing is covered separately by STA. Hence no -sdftyp here.
#
# Prereq: run a Quartus compile + `quartus_eda --simulation --tool=questa_oem
#         --format=verilog ecg_accelerator_top` to (re)generate the .vo.
# Run from hardware/fpga/simulation/questa/ (CWD must hold ecg_sample*.hex,
# expected_results.hex).

transcript on
if {[file exists gate_work]} { vdel -lib gate_work -all }
vlib gate_work
vmap work gate_work

# Synthesized netlist (Quartus-mapped primitives; weights constant-folded in).
vlog -work work {D:/Thesis101/hardware/fpga/simulation/questa/ecg_accelerator_top.vo}

# Black-box testbench (external Avalon ports only — no hierarchical probes).
vlog -sv -work work +incdir+D:/Thesis101/hardware/testbench {D:/Thesis101/hardware/testbench/tb_gate.v}

# Cyclone V simulation libraries (same set used for RTL sim in cov_tb_top.do).
vsim -t 1ps -L altera_ver -L lpm_ver -L sgate_ver -L altera_mf_ver -L altera_lnsim_ver -L cyclonev_ver -L cyclonev_hssi_ver -L cyclonev_pcie_hip_ver -L gate_work -L work -voptargs="+acc" tb_gate

run -all
