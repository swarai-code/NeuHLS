# run_hls.tcl — placeholder Vitis HLS synthesis script for SR-2L-SRL
# Adjust paths and part number before running.

open_project hls_proj_SR-2L-SRL
set_top symbolic_head
add_files {/home/swarnalp/NeuSym_server_run1/outputs/hls/SR-2L-SRL_symbolic_head.cpp}
open_solution "solution1"
set_part {xc7z020clg484-1}   ;# change to target device
create_clock -period 10 -name default
csynth_design
export_design -format ip_catalog
close_project
