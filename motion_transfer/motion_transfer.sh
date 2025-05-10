CUDA_VISIBLE_DEVICES=7 python mt.py  \
    --task mt-1.3B \
    --ckpt_dir ~/Wan2.1-T2V-1.3B \
    --video_path motion_transfer/assets/moving_ellipse.mp4 \
    --save_folder motion_transfer/output \
    --reference_prompt "" \
    --prompt "A big rock in the desert." \
    --size 832*480
