CUDA_VISIBLE_DEVICES=1 python generate.py  \
    --task t2v-14B \
    --size 1280*720 \
    --ckpt_dir ~/Wan2.1-T2V-14B \
    --prompt "A basketball rolling on the ground from left to right."

# CUDA_VISIBLE_DEVICES=7 python generate.py  \
#     --task t2v-1.3B \
#     --size 832*480 \
#     --ckpt_dir ~/Wan2.1-T2V-1.3B \
#     --prompt "A basketball rolling on the ground from left to right."
