# cov_exclude.do — exclude provably-unreachable defensive branches.
# Applied to the DU-merged coverage DB before the functional report.
#
# Each excluded line is a default / case-without-default "all-false" arm that
# CANNOT execute with valid inputs (the state/address variable is always within
# its defined set). Reason code EUR = "Excluded: UnReachable".
#
# These are NOT untested functional behavior — they are dead defensive code that
# only a fault-injection (forcing an illegal state) could reach.

# cnn_controller.v — FSM defensive defaults
coverage exclude -du cnn_controller -linerange 239 -reason EUR -comment {default: layer_state always in {LOAD,CONV1..4}}
coverage exclude -du cnn_controller -linerange 273 -reason EUR -comment {default: fc_sub_state always in {GAP,FC,FLUSH,ARGMAX,DONE}}
coverage exclude -du cnn_controller -linerange 288 -reason EUR -comment {default: illegal-state recovery, layer_state is 3-bit fully decoded}

# gap/fc/argmax split from gap_fc_argmax.v — defensive defaults
coverage exclude -du gap_unit     -linerange 93 -reason EUR -comment {GAP case default: fc_sub_state controlled by FSM}
coverage exclude -du argmax_unit  -linerange 45 -reason EUR -comment {argmax_step case all-false: 2-bit step always 0..3}

# avalon_slave.v — write case all-false (undefined write address)
coverage exclude -du avalon_slave   -linerange 46  -reason EUR -comment {write case all-false: undefined avs_address, no side effect}

# cp_engine.v L290: pong_we[oc] = cp_pool_write[oc] && cp_en[oc].
# FEC Row 3 (cp_en[oc]=0 while pool_write[oc]=1) is UNREACHABLE for oc=0..3:
# cp_en is only ever 0x00 (idle, no pool_write), 0x0F or 0xFF — bits [3:0] are
# ALWAYS 1 during compute. So channels 0-3 can never be disabled while writing.
# (Channels 4-7 DO get cp_en=0 in conv1/2 and their Row 3 is covered.)
coverage exclude -du cp_engine -linerange 290 -reason EUR -comment {pong_we ch0-3: cp_en[3:0] always 1 when computing (0x0F/0xFF)}
