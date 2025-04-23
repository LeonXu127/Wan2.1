# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
from datetime import datetime
import logging
import os
import sys
import warnings
import random

import torch
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES
from wan.configs.wan_mt_1_3B import mt_1_3B
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_video, str2bool


EXAMPLE_PROMPT = {
    "mt-1.3B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
}


def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    # The default sampling steps are 40 for image-to-video tasks and 50 for text-to-video tasks.
    if args.sample_steps is None:
        args.sample_steps = 40 if "i2v" in args.task else 50

    if args.sample_shift is None:
        args.sample_shift = 5.0
        if "i2v" in args.task and args.size in ["832*480", "480*832"]:
            args.sample_shift = 3.0

    # The default number of frames are 1 for text-to-image tasks and 81 for other tasks.
    if args.frame_num is None:
        args.frame_num = 1 if "t2i" in args.task else 81

    # T2I frame_num check
    if "t2i" in args.task:
        assert args.frame_num == 1, f"Unsupport frame_num {args.frame_num} for task {args.task}"

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    # Size check
    assert args.size in SUPPORTED_SIZES[
        args.
        task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="从视频和提示中提取motion特征"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="mt-1.3B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="832*480",
        choices=list(SIZE_CONFIGS.keys()),
        help="生成视频的大小（宽度*高度）"
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="从视频中采样的帧数。数字应该是4n+1"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="检查点目录的路径。"
    )
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="模型前向传播后是否将模型卸载到CPU，减少GPU内存使用。"
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="DiT中的ulysses并行大小。"
    )
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="DiT中的环注意力并行大小。"
    )
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="是否对T5使用FSDP。"
    )
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="是否将T5模型放在CPU上。"
    )
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="是否对DiT使用FSDP。"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="motion_transfer/reference",
        help="保存提取特征的目录。"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="一只猫在草地上奔跑",
        help="用于生成视频的提示。"
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="输入视频文件的路径。"
    )
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="用于采样的求解器。"
    )
    parser.add_argument(
        "--sample_steps", 
        type=int, 
        default=None, 
        help="采样步骤。"
    )
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="流匹配调度器的采样偏移因子。"
    )
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=5.0,
        help="无分类器引导尺度。"
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="生成视频的种子。"
    )

    args = parser.parse_args()

    _validate_args(args)

    return args


def _init_logging(rank):
    # 日志设置
    if rank == 0:
        # 设置格式
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1 or args.ring_size > 1
        ), f"context parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1 or args.ring_size > 1:
        assert args.ulysses_size * args.ring_size == world_size, f"The number of ulysses_size and ring_size should be equal to the world size."
        from xfuser.core.distributed import (initialize_model_parallel,
                                             init_distributed_environment)
        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size())

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )

    # if args.use_prompt_extend:
        # if args.prompt_extend_method == "dashscope":
        #     prompt_expander = DashScopePromptExpander(
        #         model_name=args.prompt_extend_model, is_vl="i2v" in args.task)
        # elif args.prompt_extend_method == "local_qwen":
        #     prompt_expander = QwenPromptExpander(
        #         model_name=args.prompt_extend_model,
        #         is_vl="i2v" in args.task,
        #         device=rank)
        # else:
        #     raise NotImplementedError(
        #         f"Unsupport prompt_extend_method: {args.prompt_extend_method}")

    cfg = mt_1_3B
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    if args.prompt is None:
        args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
    logging.info(f"Input prompt: {args.prompt}")
    # if args.use_prompt_extend:
        # logging.info("Extending prompt ...")
        # if rank == 0:
        #     prompt_output = prompt_expander(
        #         args.prompt,
        #         tar_lang=args.prompt_extend_target_lang,
        #         seed=args.base_seed)
        #     if prompt_output.status == False:
        #         logging.info(
        #             f"Extending prompt failed: {prompt_output.message}")
        #         logging.info("Falling back to original prompt.")
        #         input_prompt = args.prompt
        #     else:
        #         input_prompt = prompt_output.prompt
        #     input_prompt = [input_prompt]
        # else:
        #     input_prompt = [None]
        # if dist.is_initialized():
        #     dist.broadcast_object_list(input_prompt, src=0)
        # args.prompt = input_prompt[0]
        # logging.info(f"Extended prompt: {args.prompt}")

    # 创建WanMT
    logging.info("创建WanMT...")
    wan_mt = wan.WanMT(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
        t5_cpu=args.t5_cpu,
        video_name=args.video_path,
    )

    # 提取特征
    logging.info(f"开始提取特征，目标timestep: {cfg.extract_timestep}...")
    wan_mt.extract_features(
        args.prompt,
        args.video_path,
        size=SIZE_CONFIGS[args.size],
        frame_num=args.frame_num,
        shift=args.sample_shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.sample_steps,
        guide_scale=args.sample_guide_scale,
        seed=args.base_seed,
        offload_model=args.offload_model
    )

    if rank == 0:
        logging.info(f"特征提取完成，保存为: {os.path.join(wan_mt.save_path, wan_mt.feature_filename)}")
    input("Finished feature extraction...")

    logging.info(
        f"Generating video ...")
    video = wan_mt.generate(
        args.prompt,
        size=SIZE_CONFIGS[args.size],
        frame_num=args.frame_num,
        shift=args.sample_shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.sample_steps,
        guide_scale=args.sample_guide_scale,
        seed=args.base_seed,
        offload_model=args.offload_model
    )

    if rank == 0:
        if args.save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = args.prompt.replace(" ", "_").replace("/",
                                                                     "_")[:50]
            suffix = '.png' if "t2i" in args.task else '.mp4'
            args.save_file = f"{args.task}_{args.size.replace('*','x') if sys.platform=='win32' else args.size}_{args.ulysses_size}_{args.ring_size}_{formatted_prompt}_{formatted_time}" + suffix

        logging.info(f"Saving generated video to {args.save_file}")
        cache_video(
            tensor=video[None],
            save_file=args.save_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1))
    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
