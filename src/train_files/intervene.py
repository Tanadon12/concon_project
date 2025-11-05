import numpy as np
import torch
from model_nesy import NeSyConceptLearner
from scipy.optimize import linear_sum_assignment
import argparse
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 🔹 Match slots function
def match_slots(new_slots, ref_slots, cleaned_labels, threshold=0.5):
    group_idx = [i for i, lab in enumerate(cleaned_labels)
                 if lab.startswith(('shape_', 'size_', 'material_', 'color_'))]

    A = ref_slots[group_idx, :]
    B = new_slots[group_idx, :]

    cost = -np.dot(A.T, B)
    row_ind, col_ind = linear_sum_assignment(cost)
    similarity = -cost[row_ind, col_ind]

    valid = similarity >= threshold
    row_ind = row_ind[valid]
    col_ind = col_ind[valid]

    aligned = np.zeros_like(ref_slots)
    aligned[:, row_ind] = new_slots[:, col_ind]

    return aligned

# 🔹 Main script
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--attrs_t0', type=str, required=True)
    parser.add_argument('--attrs_t1', type=str, required=True)
    parser.add_argument('--slot_attention_weights', type=str, required=True)
    parser.add_argument('--dataset_path_t1', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'

    # 🔹 Load attrs_trans outputs
    attrs_t0 = np.load(args.attrs_t0)  # [N0, n_slots, n_attr]
    attrs_t1 = np.load(args.attrs_t1)  # [N1, n_slots, n_attr]

    clean_attr_labels = [
      'shape_sphere', 'shape_cube', 'shape_cylinder',
      'size_large', 'size_small',          
      'material_rubber', 'material_metal', 
      'color_cyan', 'color_blue', 'color_yellow', 'color_purple',
      'color_red', 'color_green', 'color_gray', 'color_brown'  
    ]

    # 🔹 Select first task0 sample as reference slots
    ref_slots_t0 = attrs_t0[0].T  # [n_attr, n_slots]

    # 🔹 Match and collect task0 slots
    matched_t0 = []
    for sample in attrs_t0:
        sample_T = sample.T  # [n_attr, n_slots]
        aligned = match_slots(sample_T, ref_slots_t0, clean_attr_labels, threshold=0.0)
        matched_t0.append(aligned.T)  # revert back to [n_slots, n_attr]

    matched_t0 = np.stack(matched_t0, axis=0)  # [N0, n_slots, n_attr]

    ref_slots_t1 = attrs_t1[0].T
    # 🔹 Match and collect task1 slots
    matched_t1 = []
    for sample in attrs_t1:
        sample_T = sample.T
        aligned = match_slots(sample_T, ref_slots_t1, clean_attr_labels, threshold=0.0)
        matched_t1.append(aligned.T)

    matched_t1 = np.stack(matched_t1, axis=0)  # [N1, n_slots, n_attr]

    # 🔹 Compute attribute means over samples and slots
    mean_t0 = matched_t0.mean(axis=(0,1))  # [n_attr]
    mean_t1 = matched_t1.mean(axis=(0,1))  # [n_attr]

    # 🔹 Compute attribute differences: task0 - task1
    diff = mean_t0 - mean_t1


    # 🔹 Plot differences for visualization
    diff_filtered = diff[3:]  # exclude pos_x, pos_y, pos_z
    df_diff = pd.DataFrame({'Attribute': clean_attr_labels, 'Difference': diff_filtered})
    df_diff_sorted = df_diff.reindex(df_diff['Difference'].abs().sort_values(ascending=False).index)

    print("🔬 Attribute differences (task0 - task1):")
    print(df_diff_sorted.to_string(index=False))

    # 🔹 Identify most different attribute
    confounder_attr = df_diff_sorted.iloc[0]['Attribute']
    confounder_index = clean_attr_labels.index(confounder_attr)
    print(f"\n✅ Selected confounder attribute: {confounder_attr} (index {confounder_index})")

    # 🔹 Load NeSy model with pretrained slot attention + t1 checkpoint
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

    log = torch.load(args.slot_attention_weights, map_location=device)
    model.img2state_net.load_state_dict(log['weights'], strict=True)

    ckpt = torch.load(args.model_t1_ckpt, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    print("✅ Loaded pretrained Slot Attention weights + t1 classifier checkpoint.")

    # 🔹 Dataset loader for task 0 (evaluation set)
    transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(args.dataset_path_t0, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 🔹 Evaluate with intervention
    correct_no_intervene = 0
    correct_intervene = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            cls, attrs_trans = model(imgs)

            # Baseline prediction
            preds = torch.argmax(cls, dim=1)
            correct_no_intervene += (preds == labels).sum().item()

            # 🔹 Intervene: zero out confounder attribute
            attrs_trans[:,:,confounder_index] = 0
            cls_intervene = model.set_cls(attrs_trans)
            preds_intervene = torch.argmax(cls_intervene, dim=1)
            correct_intervene += (preds_intervene == labels).sum().item()

            total += labels.size(0)

    # 🔹 Results
    baseline_acc = correct_no_intervene / total * 100
    intervene_acc = correct_intervene / total * 100
    drop = baseline_acc - intervene_acc

    print(f"\n🔷 Baseline accuracy on task0 (using t1 model): {baseline_acc:.2f}%")
    print(f"🔷 Post-intervention accuracy: {intervene_acc:.2f}%")
    print(f"🔻 Accuracy drop due to intervention on '{confounder_attr}': {drop:.2f}%")
    print("✅ Intervention evaluation completed.")