import argparse
import numpy as np
from PIL import Image
import torch
import os
import glob
from tqdm import tqdm
import shutil
from torchvision import transforms
from scipy.spatial.distance import cdist
import cv2

def load_dino_model(device):

    print("Loading DINOv2 model... (This may take a moment)")
    dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14', trust_repo=True).to(device)
    dino_model.eval()
    print("DINOv2 model loaded.")
    return dino_model

def get_dino_dense_features_and_coords(img_pil, model, device):

    transform = transforms.Compose([
        transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(518),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        features_dict = model.forward_features(img_tensor)
        patch_tokens = features_dict['x_norm_patchtokens'].squeeze(0).cpu().numpy()

    h_feat, w_feat = img_tensor.shape[2] // 14, img_tensor.shape[3] // 14
    y, x = np.mgrid[0:h_feat, 0:w_feat]
    coords = np.stack((x.ravel(), y.ravel()), axis=1) * 14 + 7

    return patch_tokens, coords

def calculate_ransac_inlier_ratio(defect_features, defect_coords, good_features, good_coords):

    distance_matrix = cdist(defect_features, good_features, 'cosine')
    matches_indices = distance_matrix.argmin(axis=1)

    src_pts = defect_coords.astype(np.float32)
    dst_pts = good_coords[matches_indices].astype(np.float32)

    if len(src_pts) < 4:
        return 0.0

    _, inlier_mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=14.0)

    if inlier_mask is None:
        return 0.0

    return np.sum(inlier_mask) / len(inlier_mask)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find best matches using DINOv2 dense features and RANSAC.")
    parser.add_argument("--data_root", default="./mvtec_anomaly_detection", help="Root directory.")
    parser.add_argument("--category", default=None, help="Category to process.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dino_model = load_dino_model(device)

    categories_to_process = [args.category] if args.category else [d for d in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, d))]

    for category in categories_to_process:
        print(f"\n{'='*20} Processing Category: {category} {'='*20}")
        category_path = os.path.join(args.data_root, category)
        best_rgb_dir = os.path.join(category_path, 'best')
        os.makedirs(best_rgb_dir, exist_ok=True)

        good_image_paths = glob.glob(os.path.join(category_path, 'train', 'good', '*.*'))

        defect_image_paths = [p for p in glob.glob(os.path.join(category_path, 'test', '*', '*.*')) if 'good' not in os.path.basename(os.path.dirname(p))]
        if not good_image_paths or not defect_image_paths: continue

        print("\n--- Pre-computing dense features for all 'good' images... ---")
        good_image_cache = {}
        for path in tqdm(good_image_paths, desc="Pre-computing good images"):
            img = Image.open(path).convert("RGB")
            features, coords = get_dino_dense_features_and_coords(img, dino_model, device)
            good_image_cache[path] = {'features': features, 'coords': coords}

        print("\n--- Finding best match for each defect image via RANSAC... ---")
        for defect_path in tqdm(defect_image_paths, desc="Matching defects"):
            try:
                defect_type = os.path.basename(os.path.dirname(defect_path))
                base_filename, _ = os.path.splitext(os.path.basename(defect_path))
                mask_path = os.path.join(category_path, 'ground_truth', defect_type, f"{base_filename}_mask.png")

                if not os.path.exists(mask_path): continue

                defect_img = Image.open(defect_path).convert("RGB")
                defect_features, defect_coords = get_dino_dense_features_and_coords(defect_img, dino_model, device)

                h_feat, w_feat = 518 // 14, 518 // 14
                mask_pil = Image.open(mask_path).convert('L').resize((w_feat, h_feat), Image.NEAREST)
                mask_flat = np.array(mask_pil).flatten() > 0

                non_defect_features = defect_features[~mask_flat]
                non_defect_coords = defect_coords[~mask_flat]

                if len(non_defect_features) < 10: continue

                best_score = -1.0
                best_match_path = None
                for good_path, good_data in good_image_cache.items():
                    good_features = good_data['features']
                    good_coords = good_data['coords']

                    score = calculate_ransac_inlier_ratio(non_defect_features, non_defect_coords, good_features, good_coords)

                    if score > best_score:
                        best_score = score
                        best_match_path = good_path

                if best_match_path:
                    output_subdir = os.path.join(best_rgb_dir, defect_type)
                    os.makedirs(output_subdir, exist_ok=True)
                    final_output_path = os.path.join(output_subdir, os.path.basename(defect_path))
                    shutil.copy(best_match_path, final_output_path)
            except Exception as e:
                print(f"Failed to process {defect_path}: {e}")

        print(f"  - Category '{category}' processing complete.")
    print("\n✅ All categories processed.")
