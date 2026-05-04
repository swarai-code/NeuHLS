# hls/run_hls.tcl
# Generic Vitis HLS synthesis script placeholder.
# After running generate_hls.py, copy a generated CPP from outputs/hls/
# and update the paths below.
#
# Usage (from Vitis HLS Tcl console or vitis_hls -f run_hls.tcl):
#   vitis_hls -f run_hls.tcl

# ── Project setup ──────────────────────────────────────────────────────────────
open_project symbolic_head_proj
set_top symbolic_head

# Adjust path to the generated CPP file
add_files {symbolic_head.cpp}

# ── Solution setup ─────────────────────────────────────────────────────────────
open_solution "solution1"

# Change to your target Xilinx device
set_part {xc7z020clg484-1}

# 100 MHz clock (10 ns period)
create_clock -period 10 -name default

# ── Synthesis ──────────────────────────────────────────────────────────────────
csynth_design

# ── Export ────────────────────────────────────────────────────────────────────
export_design -format ip_catalog

close_project
