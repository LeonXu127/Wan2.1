CUDA_VISIBLE_DEVICES=2 python mt.py  \
    --task mt-1.3B \
    --ckpt_dir ~/Wan2.1-T2V-1.3B \
    --video_path motion_transfer/assets/moving_ellipse.mp4 \
    --save_folder motion_transfer/output \
    --reference_prompt "" \
    --prompt "A basketball rolling on the ground from left to right." \
    --size 832*480
