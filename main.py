import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch

from config import parse_args
from utils.common import prepare_folders, set_seed, setup_logging


def bootstrap(args):
    args.log_file = os.path.join(args.logs_dir, f"{args.mode}_{args.session_stamp}.log")
    prepare_folders(args)
    setup_logging(args.log_file)
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # ------------------------------------------------------------------
    # GPU 性能开关：训练里所有 conv/matmul 的形状都是固定的（image_size、
    # batch_size、latent shape 全程不变），cuDNN benchmark 可以稳定挑到
    # 最优 kernel；TF32 在 A100/H100 上对 fp32 matmul 有显著收益，并且
    # autocast bf16 路径上不受影响。Stage-1/2 都受惠，正确性无变化。
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            # 老版本 torch 没有这个 API；退到 cudnn allow_tf32 即可。
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    print(f"Run directory: {args.run_dir}")
    print(f"Log file: {args.log_file}")
    print(f"Device: {device}")
    return device


def main():
    args = parse_args()
    device = bootstrap(args)

    if args.mode == "train_stage1":
        from engine.stage1 import train_stage1
        stage1_ckpt = train_stage1(args, device)
        print({"stage1_ckpt": stage1_ckpt})
        return

    if args.mode == "train_stage2":
        from engine.stage2 import train_stage2
        from engine.test_stage import test_stage
        stage2_summary = train_stage2(args, device)
        args.stage2_ckpt = stage2_summary.get("stage2_best_path") or stage2_summary.get("stage2_latest_path", "")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        test_stage(args, device)
        return

    if args.mode == "train_all":
        from engine.stage1 import train_stage1
        from engine.stage2 import train_stage2
        from engine.test_stage import test_stage
        stage1_ckpt = train_stage1(args, device)
        args.stage1_ckpt = stage1_ckpt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        stage2_summary = train_stage2(args, device)
        args.stage2_ckpt = stage2_summary.get("stage2_best_path") or stage2_summary.get("stage2_latest_path", "")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        test_stage(args, device)
        return

    if args.mode == "test":
        from engine.test_stage import test_stage
        test_stage(args, device)
        return

    if args.mode == "train_vqvae":
        from comparisons.vqvae_baseline import train_vqvae_baseline
        vqvae_ckpt = train_vqvae_baseline(args, device)
        print({"vqvae_ckpt": vqvae_ckpt})
        return

    if args.mode == "test_vqvae":
        from comparisons.vqvae_baseline import test_vqvae_baseline
        test_vqvae_baseline(args, device)
        return

    if args.mode == "train_test_vqvae":
        from comparisons.vqvae_baseline import train_vqvae_baseline, test_vqvae_baseline
        vqvae_ckpt = train_vqvae_baseline(args, device)
        args.vqvae_ckpt = vqvae_ckpt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        test_vqvae_baseline(args, device)
        return

    if args.mode == "train_cd3m":
        from comparisons.cd3m_baseline import train_cd3m_baseline
        cd3m_ckpt = train_cd3m_baseline(args, device)
        print({"cd3m_ckpt": cd3m_ckpt})
        return

    if args.mode == "test_cd3m":
        from comparisons.cd3m_baseline import test_cd3m_baseline
        test_cd3m_baseline(args, device)
        return

    if args.mode == "train_test_cd3m":
        from comparisons.cd3m_baseline import train_cd3m_baseline, test_cd3m_baseline
        cd3m_ckpt = train_cd3m_baseline(args, device)
        args.cd3m_ckpt = cd3m_ckpt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        test_cd3m_baseline(args, device)
        return

    if args.mode == "train_scdm":
        from comparisons.scdm_baseline import train_scdm_baseline
        scdm_ckpt = train_scdm_baseline(args, device)
        print({"scdm_ckpt": scdm_ckpt})
        return

    if args.mode == "test_scdm":
        from comparisons.scdm_baseline import test_scdm_baseline
        test_scdm_baseline(args, device)
        return

    if args.mode == "train_test_scdm":
        from comparisons.scdm_baseline import train_scdm_baseline, test_scdm_baseline
        scdm_ckpt = train_scdm_baseline(args, device)
        args.scdm_ckpt = scdm_ckpt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        test_scdm_baseline(args, device)
        return


    raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
