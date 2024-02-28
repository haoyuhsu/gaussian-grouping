python train.py \
    -s ./output/teatime \
    -r 1  \
    -m ./output/teatime_table \
    --config_file config/gaussian_dataset/train.json \
    --custom_traj_name transforms_001 \
    --object_name table \
    --iterations 60000 \
    --start_checkpoint ./output/teatime/chkpnt30000.pth

python train.py \
    -s ./output/teatime \
    -r 1  \
    -m ./output/teatime_apple \
    --config_file config/gaussian_dataset/train.json \
    --custom_traj_name transforms_001 \
    --object_name apple \
    --iterations 60000 \
    --start_checkpoint ./output/teatime/chkpnt30000.pth

python train.py \
    -s ./output/teatime \
    -r 1  \
    -m ./output/teatime_cookie \
    --config_file config/gaussian_dataset/train.json \
    --custom_traj_name transforms_001 \
    --object_name cookie \
    --iterations 60000 \
    --start_checkpoint ./output/teatime/chkpnt30000.pth
