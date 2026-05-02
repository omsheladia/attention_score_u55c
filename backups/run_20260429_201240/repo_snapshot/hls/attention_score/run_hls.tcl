set root_dir [file normalize "."]
set build_dir "$root_dir/attention_score_u55c/hls/build"
set proj_dir "attention_score_prj"
set top_name "attention_score_u55c_kernel"

set vector_dir "$root_dir/attention_score_u55c/sim/attention_score_tile"

file mkdir $build_dir
cd $build_dir

open_project -reset $proj_dir
set_top $top_name

add_files "$root_dir/attention_score_u55c/hls/attention_score/attention_score_core_hls.cpp"
add_files "$root_dir/attention_score_u55c/hls/attention_score/attention_score_core_hls.hpp"
add_files -tb "$root_dir/attention_score_u55c/hls/attention_score/tb_attention_score.cpp"

open_solution -reset "sol1"
set_part {xcu55c-fsvh2892-2L-e}
create_clock -period 4

csim_design -argv $vector_dir
csynth_design

exit
