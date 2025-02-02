import torch
import os

def get_config(args):
    configuration = dict(
        SEED=1337,  # Random seed for reproducibility
        INPUT_SIZE=[112, 112],  # Input image size
        EMBEDDING_SIZE=512,  # Feature dimension
    )

    # ===== 🔥 Force Single-GPU Mode 🔥 ===== #
    if torch.cuda.is_available():
        configuration["GPU_ID"] = [0]  # Use only GPU 0
        configuration["DEVICE"] = torch.device("cuda:0")
        configuration["MULTI_GPU"] = False  # Disable multi-GPU
        print(f"✅ Running on GPU: {torch.cuda.get_device_name(0)}")
    else:
        configuration["GPU_ID"] = []  # No GPU detected
        configuration["DEVICE"] = torch.device("cpu")
        configuration["MULTI_GPU"] = False
        print("⚠️ No GPU detected! Running on CPU.")

    # Set training parameters
    configuration["NUM_EPOCH"] = args.epochs
    configuration["BATCH_SIZE"] = args.batch_size
    configuration["WORKERS"] = args.num_workers

    # ===== Dataset Selection ===== #
    dataset_paths = {
        "retina": "./Data/ms1m-retinaface-t1/",
        "casia": "./data/faces_webface_112x112/",
        "casia100": "./data/faces_webface_112x112_sub100_train_test/",
        "casia1000": "./data/faces_webface_112x112_sub1000/",
        "tsne": "./data/faces_Tsne_sub/",
        "imagenet100": "./data/imagenet100/",
    }

    if args.data_mode not in dataset_paths:
        raise Exception(f"❌ Unknown dataset mode: {args.data_mode}")

    configuration["DATA_ROOT"] = dataset_paths[args.data_mode]
    configuration["EVAL_PATH"] = "./eval/"

    # ===== Model Selection ===== #
    assert args.net in ["VIT", "VITs", "VIT_B16"], "❌ Invalid network type!"
    configuration["BACKBONE_NAME"] = args.net

    assert args.head in ["Softmax", "ArcFace", "CosFace", "SFaceLoss"], "❌ Invalid head type!"
    configuration["HEAD_NAME"] = args.head

    # ===== Resume Training Checkpoint ===== #
    configuration["BACKBONE_RESUME_ROOT"] = args.resume if args.resume else ""

    # ===== Model Save Path ===== #
    configuration["WORK_PATH"] = args.outdir
    os.makedirs(args.outdir, exist_ok=True)

    # ===== Transformer Depth ===== #
    configuration["NUM_LAYERS"] = args.vit_depth

    # ===== Forgetting & Regularization Parameters ===== #
    advanced_params = [
        "one_stage", "ewc", "MAS", "si", "online", "replay", "l2",
        "BND_pro", "few_shot", "grouping", "alpha_epoch", "per_forget_cls",
        "LIRF_T", "LIRF_alpha", "scrub_decay_epoch"
    ]

    for param in advanced_params:
        if hasattr(args, param):
            configuration[param] = getattr(args, param)

    # 🔥 **Fix KeyError for `ALPHA_EPOCH` 🔥**
    configuration["ALPHA_EPOCH"] = getattr(args, "alpha_epoch", 0)  # Default to 0 if missing

    # ===== Learning Rate Configuration ===== #
    configuration["lr_decay_rate"] = 0.1
    configuration["lr_decay_epochs"] = getattr(args, "scrub_decay_epoch", None)
    configuration["sgda_learning_rate"] = args.lr

    # ===== LoRA Configuration ===== #
    configuration["GROUP_POS"] = getattr(args, "lora_pos", "FFN")

    return configuration
