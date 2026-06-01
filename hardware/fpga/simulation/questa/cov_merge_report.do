# Merge per-TB coverage by DESIGN UNIT (unions all instances of each module
# across testbenches), apply unreachable-default exclusions, emit reports.
#
# Why -du merge: each TB instantiates the design under a different top
# (/tb_top/u_top/..., /tb_fsm/u_top/..., /tb_cp_block/dut). A plain merge keeps
# them as separate instances, so a branch covered only in tb_fsm (e.g. the
# DONE_S re-start) shows "uncovered" under tb_top. Merging -du rolls every
# instance of a module into ONE figure = the true per-module coverage.
#
# Prereq: run cov_tb_cp_block.do, cov_tb_layer.do, cov_tb_fsm.do, cov_tb_top.do
#         first (each writes its .ucdb).  tb_top_probe excluded (same paths as tb_top).

vcover merge \
    -du cp_block -du cp_engine -du cnn_controller -du gap_fc_argmax \
    -du ping_pong_sram -du input_sram -du avalon_slave -du ecg_accelerator_top \
    cov_merged.ucdb \
    cov_cp_block.ucdb cov_layer.ucdb cov_fsm.ucdb cov_top.ucdb

# RAW (before exclusions, all metrics incl. toggle) — transparency.
puts "==== RAW (DU-merged, all metrics) ===="
vcover report -summary cov_merged.ucdb

# Apply unreachable-default exclusions, save, report FUNCTIONAL figure.
vsim -viewcov cov_merged.ucdb -do {
    do cov_exclude.do
    coverage save cov_merged_excl.ucdb
    quit -f
}

# FUNCTIONAL: branch/condition/expression/fsm/statement, toggle dropped
# (toggle is data-bus noise — values proven bit-exact by tb_top).
vcover report -details -code bcefs -output cov_report_functional.txt cov_merged_excl.ucdb
puts "==== FUNCTIONAL (DU-merged, excl. unreachable, -code bcefs) ===="
vcover report -summary -code bcefs cov_merged_excl.ucdb

quit -f
