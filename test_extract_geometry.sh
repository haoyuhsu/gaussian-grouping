# object_name: apple, cookie, table

# select_obj_ids=(1 2 3)

python extract_meshes.py \
    --source_path ./output/teatime \
    --model_path ./output/teatime_apple \
    --custom_traj_name transforms_001 \
    --load_iteration 60000 \
    --selected_obj_ids 1
    # --select_obj_ids "${select_obj_ids[@]}"

# python extract_meshes.py \
#     --source_path ./output/teatime \
#     --model_path ./output/teatime_cookie \
#     --custom_traj_name transforms_001

# python extract_meshes.py \
#     --source_path ./output/teatime \
#     --model_path ./output/teatime_table \
#     --custom_traj_name transforms_001
