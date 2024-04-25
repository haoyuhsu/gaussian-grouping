##### extract instance 3DGS (id=0 is background) #####
python extract_meshes.py \
    --source_path ./output/garden_norm \
    --model_path ./output/garden_norm_vase_with_flowers \
    --custom_traj_name transforms_001 \
    --load_iteration 60000 \
    --selected_obj_ids 1
