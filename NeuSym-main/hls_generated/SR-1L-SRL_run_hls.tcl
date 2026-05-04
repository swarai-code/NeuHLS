# hls/run_hls.tcl — SR-1L-SCE

open_project symbolic_head_proj
set_top symbolic_head
add_files {../hls_generated/SR-1L-SCE_symbolic_head.cpp}

open_solution "solution1"
set_part {xc7z020clg484-1}
create_clock -period 10 -name default

csynth_design
export_design -format ip_catalog

file copy -force \
    symbolic_head_proj/solution1/syn/report/symbolic_head_csynth.rpt \
    SR_1L_SCE_REPORT.rpt

close_project
