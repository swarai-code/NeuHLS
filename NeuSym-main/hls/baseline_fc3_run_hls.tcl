# baseline_fc3_run_hls.tcl
# Vitis HLS synthesis for baseline fc3 (128 → 1 dense layer)
# Comparison target for 1L symbolic experiments.

open_project hls_proj_baseline_fc3
set_top baseline_fc3
add_files {baseline_fc3.cpp}
open_solution "solution1"
set_part {xc7z020clg484-1}
create_clock -period 10 -name default
csynth_design
export_design -format ip_catalog
close_project
