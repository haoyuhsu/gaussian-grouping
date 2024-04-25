##### train 3DGS with segmentation from scratch #####
# python train.py \
#     -s ./output/teatime \
#     -r 1  \
#     -m ./output/teatime_table \
#     --config_file config/gaussian_dataset/train.json \
#     --custom_traj_name transforms_001 \
#     --object_name table \
#     --iterations 60000 \
#     --start_checkpoint ./output/teatime/chkpnt30000.pth

##### finetune pretrained vanilla 3DGS with segmentation #####
python train.py \
    -s ./output/garden_norm \
    -r 1  \
    -m ./output/garden_norm_vase_with_flowers \
    --config_file config/gaussian_dataset/train.json \
    --custom_traj_name transforms_001 \
    --object_name vase_with_flowers \
    --load_iteration 30000 \
    --iterations 60000 \
    --start_checkpoint ./output/garden_norm/chkpnt30000.pth \
    --custom_resolution 4

