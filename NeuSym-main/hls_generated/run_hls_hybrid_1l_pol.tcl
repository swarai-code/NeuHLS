open_project hybrid_mlp_1l_pol_proj

set_top hybrid_mlp_1l_pol

add_files hybrid_mlp_1l_pol.cpp
add_files hybrid_mlp_1l_pol.h

open_solution "solution1"

# For KR260/K26 SOM, use the exact part from your Vivado board/platform.
# If unsure, check Vivado project settings.
# Example placeholder:
set_part {xck26-sfvc784-2LV-c}

create_clock -period 10 -name default

csynth_design

# Optional, if you want IP export later:
# export_design -format ip_catalog

close_project
exit
