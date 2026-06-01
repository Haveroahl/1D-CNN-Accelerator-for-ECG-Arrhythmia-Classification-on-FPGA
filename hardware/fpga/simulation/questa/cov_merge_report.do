# Merge all per-TB coverage UCDBs and emit reports.
# Prereq: run cov_tb_cp_block.do, cov_tb_layer.do, cov_tb_fsm.do, cov_tb_top.do
#         first (each writes its .ucdb).
#
# tb_top_probe is intentionally excluded from measurement: same RTL paths as
# tb_top (identical 7-stage bit-exact compare), no new bins.
#
# Reporting policy (see testplan §9 for rationale):
#   - FUNCTIONAL figure uses -code bcefs  (branch, condition, expression, fsm,
#     statement) and DROPS the toggle metric. Toggle coverage is not meaningful
#     for this design: ~54K toggle bins are weight-ROM / datapath data buses whose
#     bits are constant or data-dependent; their VALUES are proven bit-exact by
#     tb_top, which is the real correctness evidence. Including toggle would
#     understate functional coverage with noise.
#   - The raw all-metrics report is also emitted for full transparency.

vcover merge cov_merged.ucdb \
    cov_cp_block.ucdb cov_layer.ucdb cov_fsm.ucdb cov_top.ucdb

# (1) FUNCTIONAL coverage report (no toggle) — this is the figure to cite.
vcover report -details -code bcefs -output cov_report_functional.txt cov_merged.ucdb
puts "==== FUNCTIONAL coverage (no toggle, -code bcefs) ===="
vcover report -summary -code bcefs cov_merged.ucdb

# (2) RAW all-metrics report (incl. toggle) — for transparency.
vcover report -details -output cov_report_raw.txt cov_merged.ucdb
puts "==== RAW all-metrics (incl. toggle) ===="
vcover report -summary cov_merged.ucdb

quit -f
