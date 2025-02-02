import torch
import torch.nn as nn
import sys
from vit_pytorch_face import ViT_face
from vit_pytorch_face import ViTs_face
from util.utils import get_val_data, perform_val
import sklearn
import cv2
import numpy as np
from image_iter import FaceDataset
import torch.utils.data as data
import argparse
import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import wandb


def main(args):
    print(args)

    # Check if CUDA (GPU) is available
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda:0")  # Use the single available GPU
        GPU_ID = [0]
        print("✅ Using GPU:", torch.cuda.get_device_name(0))
    else:
        DEVICE = torch.device("cpu")  # Fallback to CPU
        GPU_ID = "cpu"
        print("⚠️ GPU not found! Running on CPU.")

    # Define number of classes for CASIA-100
    NUM_CLASS = 100  

    # Initialize model
    if args.network == "VIT":
        model = ViT_face(
            image_size=112,
            patch_size=8,
            loss_type="CosFace",
            GPU_ID=GPU_ID,
            num_class=NUM_CLASS,
            dim=512,
            depth=args.depth,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1,
            lora_rank=args.lora_rank,
            lora_pos=args.lora_pos,
        )
    elif args.network == "VITs":
        model = ViTs_face(
            loss_type="CosFace",
            GPU_ID=GPU_ID,
            num_class=NUM_CLASS,
            image_size=112,
            patch_size=8,
            ac_patch_size=12,
            pad=4,
            dim=512,
            depth=args.depth,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1,
            lora_rank=args.lora_rank,
        )
    else:
        raise ValueError("❌ Invalid network type! Use 'VIT' or 'VITs'.")

    # Load pretrained model
    model_path = os.path.join(os.getcwd(), args.model)
    if os.path.exists(model_path):
        print(f"✅ Loading model from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE), strict=False)
    else:
        raise FileNotFoundError(f"❌ Model file not found: {model_path}")

    # Transformations
    data_transform = transforms.Compose([transforms.ToTensor()])

    # Load dataset
    test_dataset = datasets.ImageFolder(
        root="./data/faces_webface_112x112_sub100_train_test/test",
        transform=data_transform,
    )
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    # Move model to device
    model.to(DEVICE)
    model.eval()

    # Evaluate Model
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).long()
            outputs, _ = model(images, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print("\n🎯 Test Accuracy: {:.4f}%".format(accuracy))

    # Log accuracy to Weights & Biases (if enabled)
    wandb.log({"Test Accuracy": accuracy})

    # Per-class accuracy
    class_correct = [0.0] * NUM_CLASS
    class_total = [0.0] * NUM_CLASS
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).long()
            outputs, _ = model(images, labels)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print("\n🎯 Per-Class Accuracy:")
    for i in range(NUM_CLASS):
        if class_total[i] > 0:
            print(f"Class {i}: {100 * class_correct[i] / class_total[i]:.2f}%")

    # Save results
    with open("class_accuracy.txt", "w") as f:
        for i in range(NUM_CLASS):
            if class_total[i] > 0:
                f.write(f"Class {i}: {100 * class_correct[i] / class_total[i]:.2f}%\n")
    wandb.save("class_accuracy.txt")


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6/Backbone_VIT_Epoch_1185_Batch_45020_Time_2024-09-26-03-26_checkpoint.pth",
        help="Path to pretrained model",
    )
    parser.add_argument("--network", default="VIT", help="Network type (VIT or VITs)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lora_rank", type=int, default=0, help="LoRA rank")
    parser.add_argument(
        "--lora_pos",
        type=str,
        default="FFN",
        help="LoRA position (FFN or attention)",
    )
    parser.add_argument("--depth", type=int, default=6, help="Transformer depth")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument(
        "-w",
        "--workers_id",
        help="GPU ID (or 'cpu' for CPU mode)",
        default="0",
        type=str,
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
