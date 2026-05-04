# baseline_fc2_fc3_run_hls.tcl
# Vitis HLS synthesis for baseline fc2+fc3 (512 -> 128 -> 1)
# Comparison target for 2L symbolic experiments.
#
# Note: baseline_fc2_fc3() takes W2/B2/W3/B3 as arguments.
# The testbench must supply these from the trained PyTorch checkpoint.
# To export weights:
#   import torch, numpy as np
#   ckpt = torch.load("outputs/checkpoints/baseline.pt", weights_only=True)
#   sd = ckpt["model_state_dict"]
#   np.savetxt("W2.csv", sd["fc2.weight"].numpy(), delimiter=",")
#   np.savetxt("B2.csv", sd["fc2.bias"].numpy(),   delimiter=",")
#   np.savetxt("W3.csv", sd["fc3.weight"].numpy(), delimiter=",")
#   print("B3:", sd["fc3.bias"].item())

open_project hls_proj_baseline_fc2_fc3
set_top baseline_fc2_fc3
add_files {baseline_fc2_fc3.cpp}
open_solution "solution1"
set_part {xc7z020clg484-1}
create_clock -period 10 -name default
csynth_design
export_design -format ip_catalog
close_project
