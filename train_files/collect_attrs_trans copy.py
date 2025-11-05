import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from model_nesy import NeSyConceptLearner
import os
import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_set_cls(model, train_loader, val_loader, device, epochs=10, lr=0.001):
    """
    Train set transformer classifier using optimizer + scheduler setup from original code.
    Evaluate on validation set each epoch.
    """
    model.img2state_net.eval()
    model.set_cls.train()

    optimizer = torch.optim.Adam(
        [p for name, p in model.named_parameters() if p.requires_grad and 'set_cls' in name],
        lr=lr
    )
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    for epoch in range(epochs):
        running_loss, correct, total = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            imgs = (imgs - 0.5) * 2.0

            optimizer.zero_grad()
            outputs, _ = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_acc = correct / total * 100

        # Evaluate on validation set
        model.set_cls.eval()
        correct_val, total_val = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                imgs = (imgs - 0.5) * 2.0
                outputs, _ = model(imgs)
                preds = outputs.argmax(dim=1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_acc = correct_val / total_val * 100
        print(f"Epoch [{epoch+1}/{epochs}] TrainAcc: {train_acc:.2f}% | ValAcc: {val_acc:.2f}%")
        model.set_cls.train()


def match_slots(new_slots, ref_slots, cleaned_labels, threshold=0.5):
    """
    Robust: match slots using multiple groups + reject low similarity pairs.
    """
    group_idx = [
        i for i, lab in enumerate(cleaned_labels)
        if lab.startswith(('shape_', 'size_', 'material_',  'color_'))
    ]

    A = ref_slots[group_idx, :]
    B = new_slots[group_idx, :]

    cost = -np.dot(A.T, B)  # negative similarity

    row_ind, col_ind = linear_sum_assignment(cost)
    similarity = -cost[row_ind, col_ind]  # back to positive

    valid = similarity >= threshold

    # Only accept high-similarity matches
    row_ind = row_ind[valid]
    col_ind = col_ind[valid]

    # Allocate aligned matrix with same slots as ref_slots
    aligned = np.zeros_like(ref_slots)
    aligned[:, row_ind] = new_slots[:, col_ind]

    return aligned

def collect_attrs_trans(model, loader, device):
    """
    Collect attrs_trans outputs for all images in loader and return as numpy array.
    """
    all_attrs = []
    model.eval()

    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            _, attrs_trans = model(imgs)
            all_attrs.append(attrs_trans.cpu().numpy())

    all_attrs = np.concatenate(all_attrs, axis=0)  # [N, n_slots, n_attr]
    return all_attrs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', type=str, required=True, help='Comma-separated tasks e.g. t0,t1')
    parser.add_argument('--dataset_type', type=str, required=True, help='Dataset variant')
    parser.add_argument('--root_dir', type=str, required=True, help='Root dataset directory')
    parser.add_argument('--slot_attention_weights', type=str, required=True, help='Pretrained slot attention weights path')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tasks = args.tasks.split(',')

    for task in tasks:
        print(f"\n🔷 Processing Task: {task}")

        # Initialize model
        model = NeSyConceptLearner(
            n_classes=2,
            n_slots=10,
            n_iters=3,
            n_attr=18,
            n_set_heads=4,
            set_transf_hidden=128,
            category_ids=[3, 6, 8, 10, 17],
            device=device
        ).to(device)

        # Load pretrained slot attention weights only
        log = torch.load(args.slot_attention_weights, map_location=device)
        model.img2state_net.load_state_dict(log['weights'], strict=True)
        print("✅ Loaded pretrained Slot Attention weights.")

        # Dataset loaders
        transform = transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor()])

        dataset_path_train = os.path.join(args.root_dir, args.dataset_type, 'train', 'images', task)
        dataset_train = datasets.ImageFolder(dataset_path_train, transform=transform)
        loader_train = DataLoader(dataset_train, batch_size=64, shuffle=True)

        dataset_path_val = os.path.join(args.root_dir, args.dataset_type, 'val', 'images', task)
        dataset_val = datasets.ImageFolder(dataset_path_val, transform=transform)
        loader_val = DataLoader(dataset_val, batch_size=64, shuffle=False)

        dataset_path_test = os.path.join(args.root_dir, args.dataset_type, 'test', 'images', task)
        dataset_test = datasets.ImageFolder(dataset_path_test, transform=transform)
        loader_test = DataLoader(dataset_test, batch_size=64, shuffle=False)

        # ===== Train set transformer classifier =====
        print(f"🔷 Training Set Transformer classifier on {task}")
        train_set_cls(model, loader_train, loader_val, device, epochs=args.epochs, lr=args.lr)

        # ===== Save model checkpoint =====
        ckpt_path = os.path.join(args.root_dir, f"nesy_model_{task}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"✅ Saved model checkpoint to {ckpt_path}")

        # ===== Collect and save attrs_trans outputs for train, val, test only =====
        combined_attrs = collect_and_match_attrs_trans(model, splits_loaders, device, clean_attr_labels, match_slots)

        # 🔹 Save combined attrs_trans per task
        save_path = os.path.join(args.root_dir, f"attrs_trans_{task}_combined.npy")
        np.save(save_path, combined_attrs)
        print(f"✅ Saved combined attrs_trans outputs for {task} to {save_path}")
