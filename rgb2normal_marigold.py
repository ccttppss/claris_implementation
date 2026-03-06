import argparse
import numpy as np
from scipy import ndimage
from PIL import Image
import torch
from torchvision.transforms.functional import to_pil_image
from diffusers import MarigoldNormalsPipeline
import os
import glob
from tqdm import tqdm

def generate_normal_map_advanced(
    input_path: str,
    output_path: str,
    pipe: MarigoldNormalsPipeline,
    device: str,

    processing_resolution: int = 768,                                       
    median_size: int = 0,
    contrast_threshold: float = 0.05,
    gaussian_sigma: float = 0.25
):

    try:
        img = Image.open(input_path).convert("RGB")
    except Exception as e:
        print(f"    - Failed to open image {input_path}: {e}")
        return

    output = pipe(
        img,
        output_type="pt",
        processing_resolution=processing_resolution
    )

    normals = output.prediction[0]
    normals_np = normals.cpu().numpy()

    if median_size > 0:
        filter_size = median_size if median_size % 2 != 0 else median_size + 1
        for i in range(normals_np.shape[0]):
            normals_np[i] = ndimage.median_filter(normals_np[i], size=filter_size)

    magnitude = np.sqrt(normals_np[0]**2 + normals_np[1]**2)
    mag_std = np.std(magnitude)

    if mag_std < contrast_threshold:
        print(f"  - Low contrast detected (std: {mag_std:.4f}). Applying contrast stretching.")
        for i in range(2):
            p_low, p_high = np.percentile(normals_np[i], [2, 98])
            if p_high > p_low:
                stretched = -1.0 + 2.0 * (normals_np[i] - p_low) / (p_high - p_low)
                normals_np[i] = np.clip(stretched, -1.0, 1.0)
        normals_np[2] = np.sqrt((1.0 - normals_np[0]**2 - normals_np[1]**2).clip(0))
    else:
        print(f"  - Sufficient contrast detected (std: {mag_std:.4f}). Skipping contrast stretching.")

    if gaussian_sigma > 0:
        normals_np = ndimage.gaussian_filter(normals_np, sigma=(0, gaussian_sigma, gaussian_sigma))

    normals = torch.from_numpy(normals_np).to(device)
    normals = normals * 0.5 + 0.5
    normals_byte = (normals * 255.0).clamp(0, 255).to(torch.uint8)

    normals_img = to_pil_image(normals_byte)
    normals_img = normals_img.resize((512, 512), Image.LANCZOS)
    normals_img.save(output_path)

def process_category(data_root: str, category_name: str, pipe: MarigoldNormalsPipeline, device: str):
    print(f"\nProcessing category: '{category_name}'")
    category_path = os.path.join(data_root, category_name)
    input_root_path = os.path.join(category_path, "best")
    output_root_path = os.path.join(category_path, "best_normal")

    if not os.path.isdir(input_root_path):
        print(f"  - Skipping: Input directory not found at '{input_root_path}'")
        return

    defect_folders = [f for f in os.listdir(input_root_path) if os.path.isdir(os.path.join(input_root_path, f))]

    if not defect_folders:
        print(f"  - Skipping: No defect subdirectories found in '{input_root_path}'")
        return

    for defect_name in tqdm(defect_folders, desc=f"  Defects in '{category_name}'"):
        current_input_dir = os.path.join(input_root_path, defect_name)
        current_output_dir = os.path.join(output_root_path, defect_name)
        os.makedirs(current_output_dir, exist_ok=True)

        image_files = glob.glob(os.path.join(current_input_dir, "*.png")) +                      glob.glob(os.path.join(current_input_dir, "*.jpg")) +                      glob.glob(os.path.join(current_input_dir, "*.jpeg"))

        if not image_files:
            continue

        for img_path in image_files:
            file_name = os.path.basename(img_path)
            output_file_path = os.path.join(current_output_dir, file_name)

            try:
                generate_normal_map_advanced(
                    img_path,
                    output_file_path,
                    pipe,
                    device,
                    processing_resolution=768,
                    median_size=0,
                    contrast_threshold=0.1,                        
                    gaussian_sigma=0
                )
            except Exception as e:
                print(f"    - Error processing {img_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate normal maps with high-resolution processing.")
    parser.add_argument("data_root", help="Root directory of the dataset.")
    parser.add_argument("category", nargs='?', default=None, help="Category to process. If not provided, all categories will be processed.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Using device: {device}, dtype: {torch_dtype}")

    try:
        pipe = MarigoldNormalsPipeline.from_pretrained(
            "prs-eth/marigold-normals-v1-1",
            torch_dtype=torch_dtype
        ).to(device)
        print("MarigoldNormalsPipeline loaded successfully.")
    except Exception as e:
        print(f"Failed to load MarigoldNormalsPipeline: {e}")
        exit()

    if args.category:
        categories_to_process = [args.category]
        if not os.path.isdir(os.path.join(args.data_root, args.category)):
             print(f"Error: Specified category '{args.category}' not found in '{args.data_root}'")
             exit()
    else:
        print("No category specified, processing all categories...")
        categories_to_process = [d for d in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, d))]

    for category in categories_to_process:
        process_category(args.data_root, category, pipe, device)

    print("\nAll requested normal map generations have been completed.")
