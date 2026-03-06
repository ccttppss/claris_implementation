# CLARIS: Control-based Language-guided Realistic Imperfection Synthesis

A defect image generation framework using **ControlNet + LoRA + Textual Inversion** on top of Stable Diffusion Inpainting. Given a normal image and a mask, the model synthesizes a realistic defect in the masked region while preserving the surrounding context.

## Requirements

```bash
pip install -r requirements.txt
```

Tested with Python 3.10, CUDA 12.8, PyTorch 2.7.0.

## Dataset Setup

Download [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) and extract it directly inside this repository directory. The extracted folder must be named `mvtec_anomaly_detection`. The expected structure is:

```
claris/
└── mvtec_anomaly_detection/
    ├── bottle/
    │   ├── train/
    │   │   └── good/          # Candidate pool used by find_best.py to select best-matching normal images
    │   ├── test/
    │   │   ├── good/          # Normal images used as input to test.py (not seen during training)
    │   │   ├── broken_large/  # Defect images (used as training targets)
    │   │   └── ...
    │   └── ground_truth/
    │       ├── broken_large/  # Binary masks (filename: {image_stem}_mask.png)
    │       └── ...
    ├── cable/
    ├── capsule/
    └── ...                    # 15 categories total
```

> Do **not** rename the extracted folder. The scripts expect `./mvtec_anomaly_detection` by default.

## Step-by-Step Usage

Run the following steps in order. Each step's output is consumed by the next.

### Step 1: Find Best-Matching Normal Image

For each defect image in `test/`, finds the closest normal image from `train/good/` using **DINOv2 dense features + RANSAC** (matching is mask-aware: defect regions are excluded from comparison). The matched normal image is copied to `{category}/best/`.

```bash
# All categories
python find_best.py --data_root ./mvtec_anomaly_detection

# Single category
python find_best.py --data_root ./mvtec_anomaly_detection --category bottle
```

After this step, `mvtec_anomaly_detection/bottle/best/broken_large/000.png` will contain the best-matching normal image for `test/broken_large/000.png`.

### Step 2: Generate Normal Maps

Generates surface normal maps from the `best/` images using **Marigold** and saves them to `{category}/best_normal/`. These are used as ControlNet conditioning during training.

```bash
# All categories
python rgb2normal_marigold.py ./mvtec_anomaly_detection

# Single category
python rgb2normal_marigold.py ./mvtec_anomaly_detection bottle
```

> Alternatively, use `rgb2normal_normalbae.py` (**NormalBAE**) for a faster, simpler alternative.

After this step, `mvtec_anomaly_detection/bottle/best_normal/broken_large/000.png` will contain the corresponding normal map.

### Step 3: Train the Model

Trains a **LoRA-finetuned UNet** and **Textual Inversion** embeddings for each defect category. Each category is trained independently and saved under `checkpoints/{category}/`.

```bash
# Single category
python train.py --category bottle

# Example
python train.py \
  --pretrained_model_name_or_path runwayml/stable-diffusion-inpainting \
  --controlnet_model_name_or_path lllyasviel/control_v11p_sd15_normalbae \
  --dataset_root_dir ./mvtec_anomaly_detection \
  --output_dir ./checkpoints \
  --category bottle \
  --num_train_epochs 1000 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 3e-5 \
  --weight_decay 1e-4 \
  --lora_rank 16 \
  --mixed_precision fp16
```

Trained weights are saved to:
```
checkpoints/
└── bottle/
    ├── unet_lora/          # LoRA adapter weights
    └── learned_embeds.bin  # Textual Inversion embeddings
```

### Step 4: Generate Defect Images

Two modes are available depending on the use case.

#### Mode A — Expert Mode (`test.py`)

Manually provide the input image, mask, category, and prompt. This mode does not require a VLM and is faster.

```bash
# Example
python test.py \
  --lora_root_dir ./checkpoints \
  --category bottle \
  --input_image_path ./mvtec_anomaly_detection/bottle/test/good/000.png \
  --mask_image_path ./mvtec_anomaly_detection/bottle/ground_truth/broken_large/000_mask.png \
  --num_inference_steps 100 \
  --guidance_scale 10.0 \
  --controlnet_scale 0.5
```

If `--prompt` is not specified, it is auto-generated from the mask path and `info-map.json` (e.g., `"a photo of a bottle with <bottle-broken-large> defect"`). You can still pass `--prompt` manually to override.

#### Mode B — VLM Mode (`test_vlm.py`)

Provide only a normal image and a natural language prompt. **Gemma3** (`google/gemma-3n-e4b-it`) automatically identifies the object category, selects the appropriate defect token, and generates a binary mask based on the described location and size. A manually created mask can optionally be provided via `--mask_path`.

```bash
# Natural language only (mask auto-generated)
python test_vlm.py \
  --image_path ./mvtec_anomaly_detection/leather/test/good/000.png \
  --prompt "Make a medium glue in the center" \
  --lora_root_dir ./checkpoints

# With a manually specified mask
python test_vlm.py \
  --image_path ./mvtec_anomaly_detection/leather/test/good/000.png \
  --prompt "a glue defect" \
  --mask_path ./mvtec_anomaly_detection/leather/ground_truth/glue/000_mask.png \
  --lora_root_dir ./checkpoints
```

> **Note on Normal Map Extraction:** Both `test.py` and `test_vlm.py` use the lightweight **NormalBAE** for normal map estimation by default. If you prefer the heavier, higher-quality **Marigold** pipeline, simply append the `--use_marigold` flag to your command.

The output image is saved under `./generated_vlm/`.

> **Note:** `test_vlm.py` loads Gemma3 first, then releases VRAM before loading the diffusion pipeline. A GPU with at least 16GB VRAM is recommended.

## Project Structure

```
claris/
├── find_best.py            # Step 1: Find best-matching normal image per defect
├── rgb2normal_marigold.py  # Step 2: Normal map generation (Marigold)
├── rgb2normal_normalbae.py # Step 2 (alt): Normal map generation (NormalBAE)
├── train.py                # Step 3: Training (LoRA + Textual Inversion)
├── test.py                 # Step 4A: Defect generation (expert mode, manual mask & category)
├── test_vlm.py             # Step 4B: Defect generation (VLM mode, natural language interface)
├── dataset.py              # Dataset class used by train.py
├── info-map.json           # Defect type → placeholder token mapping
├── master_normals/         # Per-category reference normal maps (used by test.py / test_vlm.py)
└── requirements.txt
```