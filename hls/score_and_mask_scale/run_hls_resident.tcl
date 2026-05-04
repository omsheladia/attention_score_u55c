set root_dir [file normalize "."]
set build_dir "$root_dir/attention_score_u55c/hls/build"
set proj_dir "score_and_mask_scale_resident_prj"
set top_name "score_mask_scale_resident_u55c_kernel"

file mkdir $build_dir
cd $build_dir

open_project -reset $proj_dir
set_top $top_name

add_files "$root_dir/attention_score_u55c/hls/score_and_mask_scale/score_mask_scale_core_hls.cpp"
add_files "$root_dir/attention_score_u55c/hls/score_and_mask_scale/score_mask_scale_core_hls.hpp"
add_files -tb "$root_dir/attention_score_u55c/hls/score_and_mask_scale/tb_score_mask_scale.cpp"

open_solution -reset "sol1"
set_part {xcu55c-fsvh2892-2L-e}
create_clock -period 4

csim_design -argv "$root_dir/attention_score_u55c/sim/attention_score_tile"
csynth_design

exit
