set root_dir [file normalize "."]
set build_dir "$root_dir/attention_score_u55c/hls/build"
set proj_dir "v_weighted_sum_prj"
set top_name "v_weighted_sum_u55c_kernel"

file mkdir $build_dir
cd $build_dir

open_project -reset $proj_dir
set_top $top_name

add_files "$root_dir/attention_score_u55c/hls/v_weighted_sum/v_weighted_sum_core_hls.cpp"
add_files "$root_dir/attention_score_u55c/hls/v_weighted_sum/v_weighted_sum_core_hls.hpp"
add_files -tb "$root_dir/attention_score_u55c/hls/v_weighted_sum/tb_v_weighted_sum.cpp"

open_solution -reset "sol1"
set_part {xcu55c-fsvh2892-2L-e}
create_clock -period 4

csim_design
csynth_design

exit
