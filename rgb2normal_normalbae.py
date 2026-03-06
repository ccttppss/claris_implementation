import argparse
import torch
from PIL import Image
import numpy as np
import os
import glob
from tqdm import tqdm
from controlnet_aux import NormalBaeDetector                                

def generate_normal_map_aux(
    input_path: str,
    output_path: str,
    processor: NormalBaeDetector
):

    try:

        input_image = Image.open(input_path).convert("RGB")

        result_image = processor(input_image, detect_resolution=1024, image_resolution=1024)

        result_image.save(output_path)

    except Exception as e:
        print(f"    - Error processing {input_path} with controlnet_aux: {e}")

def process_category_aux(data_root: str, category_name: str, processor):

    print(f"\nProcessing category with controlnet_aux: '{category_name}'")
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
            generate_normal_map_aux(img_path, output_file_path, processor)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate normal maps using controlnet_aux and NormalBAE.")
    parser.add_argument("data_root", help="Root directory of the dataset.")
    parser.add_argument("category", nargs='?', default=None, help="Category to process. If not provided, all categories will be processed.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:

        processor = NormalBaeDetector.from_pretrained("lllyasviel/Annotators").to(device)
        print("NormalBaeDetector from controlnet_aux loaded successfully.")
    except Exception as e:
        print(f"Failed to load NormalBaeDetector: {e}")
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
        process_category_aux(args.data_root, category, processor)

    print("\nAll requested normal map generations have been completed.")
