# Coverage run — tb_cp_block (unit, M1 cp_block only)
transcript on
if {[file exists cov_work]} { vdel -lib cov_work -all }
vlib cov_work
vmap work cov_work

# +cover=bcefsx : branch, condition, expression, fsm, statement, toggle
vlog -sv +cover=bcefsx -work work +incdir+D:/Thesis101/hardware/RTL {D:/Thesis101/hardware/RTL/cp_block.v}
vlog -sv -work work +incdir+D:/Thesis101/hardware/../testbench {D:/Thesis101/hardware/testbench/tb_cp_block.v}

# -onfinish stop: the testbench calls $finish; without this, sim exits before
# coverage save runs. 'stop' halts on $finish so the save below executes.
vsim -coverage -onfinish stop -t 1ps -L cov_work -L work -voptargs="+acc +cover=bcefsx" tb_cp_block
run -all
coverage save cov_cp_block.ucdb
quit -f
