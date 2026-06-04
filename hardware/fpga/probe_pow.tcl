puts "=== set_power commands ==="
foreach c [lsort [info commands set_power*]] { puts $c }
puts "=== all *power_file* / *vcd* ==="
foreach c [lsort [info commands *power_file*]] { puts $c }
foreach c [lsort [info commands *vcd*]] { puts $c }
