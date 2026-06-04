project_open ecg_accelerator_top
# VCD top scope is tb_top; DUT instance is u_top -> map design top to that scope
set_power_file_assignment "simulation/questa/tb_top.vcd" \
    -vcd_instance_name "u_top" \
    -temp 25
project_close
