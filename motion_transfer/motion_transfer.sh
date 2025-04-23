CUDA_VISIBLE_DEVICES=7 python mt.py  \
    --task mt-1.3B \
    --video_path motion_transfer/assets/test1.mp4 \
    --size 832*480 \
    --ckpt_dir ~/Wan2.1-T2V-1.3B \
    --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
