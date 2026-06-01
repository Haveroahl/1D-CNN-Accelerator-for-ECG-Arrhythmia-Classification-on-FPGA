# Merge all per-TB coverage UCDBs and emit a combined report.
# Prereq: run cov_tb_cp_block.do, cov_tb_layer.do, cov_tb_top.do, cov_tb_fsm.do
#         first (each writes its .ucdb).
#
# tb_top_probe is intentionally excluded: it exercises the same RTL paths as
# tb_top (identical 7-stage bit-exact compare) and adds observability only, no
# new coverage bins.

vcover merge cov_merged.ucdb \
    cov_cp_block.ucdb cov_layer.ucdb cov_top.ucdb cov_fsm.ucdb

# Text summary by design unit (statement/branch/fsm/toggle/cond/expr).
vcover report -details -output cov_report.txt cov_merged.ucdb

# Per-instance totals to stdout.
vcover report -summary cov_merged.ucdb

quit -f
