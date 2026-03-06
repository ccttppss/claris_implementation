import argparse
import torch
from diffusers import (
    StableDiffusionControlNetInpaintPipeline,
    ControlNetModel,
    UNet2DConditionModel
)
from diffusers import MarigoldNormalsPipeline
from peft import PeftModel
from PIL import Image
from pathlib import Path
import numpy as np
import torchvision.transforms.functional as TF
from torchvision.transforms import ToTensor

try:
    from controlnet_aux import NormalBaeDetector
except ImportError:
    NormalBaeDetector = None

def generate_normal_map_for_inference(
    img: Image.Image,
    estimator,
    use_marigold: bool,
    master_normal_path: Path,
    device: str,
    processing_resolution: int = 768,
) -> Image.Image:
    print("  - Step 1: Predicting normal map from input image...")
    if use_marigold:
        output = estimator(
            img,
            output_type="pt",
            processing_resolution=processing_resolution
        )
        generated_normals_tensor = output.prediction[0]
    else:
        generated_normals_tensor = ToTensor()(estimator(img, detect_resolution=1024, image_resolution=1024)) * 2.0 - 1.0

    print(f"  - Step 2: Loading master normal map '{master_normal_path.name}' to extract parameters...")
    if not master_normal_path.exists():
        raise FileNotFoundError(f"Master normal map not found at {master_normal_path}")

    master_img_pil = Image.open(master_normal_path).convert("RGB").resize(generated_normals_tensor.shape[-2:])
    master_tensor = ToTensor()(master_img_pil) * 2.0 - 1.0
    master_np = master_tensor.cpu().numpy()

    master_p_low_x, master_p_high_x = np.percentile(master_np[0], [2, 98])
    master_p_low_y, master_p_high_y = np.percentile(master_np[1], [2, 98])
    print(f"  - Extracted Parameters (X percentile): {master_p_low_x:.3f} ~ {master_p_high_x:.3f}")
    print(f"  - Extracted Parameters (Y percentile): {master_p_low_y:.3f} ~ {master_p_high_y:.3f}")

    print("  - Step 3: Applying extracted parameters to the new normal map...")
    processed_normals_np = generated_normals_tensor.cpu().numpy()

    if master_p_high_x > master_p_low_x:
        stretched_x = -1.0 + 2.0 * (processed_normals_np[0] - master_p_low_x) / (master_p_high_x - master_p_low_x)
        processed_normals_np[0] = np.clip(stretched_x, -1.0, 1.0)

    if master_p_high_y > master_p_low_y:
        stretched_y = -1.0 + 2.0 * (processed_normals_np[1] - master_p_low_y) / (master_p_high_y - master_p_low_y)
        processed_normals_np[1] = np.clip(stretched_y, -1.0, 1.0)

    processed_normals_np[2] = np.sqrt((1.0 - processed_normals_np[0]**2 - processed_normals_np[1]**2).clip(0))

    final_normals_tensor = torch.from_numpy(processed_normals_np).to(device)
    final_normals_tensor = final_normals_tensor * 0.5 + 0.5
    normals_byte = (final_normals_tensor * 255.0).clamp(0, 255).to(torch.uint8)

    normals_img = TF.to_pil_image(normals_byte)

    return normals_img

def load_textual_inversion_embeds(pipeline, learned_embeds_path: Path):
    if not learned_embeds_path.exists():
        print(f"Warning: Textual Inversion embeddings not found at {learned_embeds_path}. Skipping.")
        return
    try:
        learned_embeds = torch.load(learned_embeds_path, map_location="cpu")
    except Exception as e:
        print(f"Warning: Failed to load TI embeddings: {e}")
        return
    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder
    placeholder_tokens = list(learned_embeds.keys())
    tokenizer.add_tokens(placeholder_tokens)
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data
    for token, embed in learned_embeds.items():
        token_id = tokenizer.convert_tokens_to_ids(token)
        token_embeds[token_id] = embed
    print(f"Loaded {len(learned_embeds)} textual inversion embeddings.")

def main(args):
    import json
    base_dir = Path(args.lora_root_dir) / args.category
    unet_lora_path = base_dir / "unet_lora"
    ti_path = base_dir / "learned_embeds.bin"
    master_normal_path = Path("./master_normals") / f"{args.category}.png"

    if not unet_lora_path.is_dir():
        print(f"Error: LoRA directory not found: {unet_lora_path}")
        return

    if args.prompt is None:
        defect_type = Path(args.mask_image_path).parent.name
        key = f"{args.category}_{defect_type}"
        try:
            with open(args.info_map_path, 'r') as f:
                info_map = json.load(f)
            token = info_map[key]["placeholder_tokens"][0]
            args.prompt = f"a photo of a {args.category} with {token} defect"
            print(f"Auto-generated prompt: '{args.prompt}'")
        except (KeyError, FileNotFoundError) as e:
            print(f"Error: Could not auto-generate prompt ({e}). Please specify --prompt manually.")
            return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    try:
        if args.use_marigold:
            normal_estimator = MarigoldNormalsPipeline.from_pretrained(
                "prs-eth/marigold-normals-v1-1", torch_dtype=torch_dtype
            ).to(device)
            print("MarigoldNormalsPipeline loaded successfully.")
        else:
            if NormalBaeDetector is None:
                raise ImportError("controlnet_aux is not installed.")
            normal_estimator = NormalBaeDetector.from_pretrained("lllyasviel/Annotators").to(device)
            print("NormalBaeDetector loaded successfully.")
    except Exception as e:
        print(f"Failed to load normal estimator: {e}")
        return

    print("Loading base UNet model...")
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", torch_dtype=torch_dtype
    )
    print(f"Loading LoRA weights from: {unet_lora_path} using PEFT...")
    unet = PeftModel.from_pretrained(unet, unet_lora_path)
    print("Successfully loaded LoRA weights into UNet using PEFT.")
    print("Loading final pipeline with LoRA-applied UNet...")
    controlnet = ControlNetModel.from_pretrained(
        args.controlnet_model_name_or_path, torch_dtype=torch_dtype
    )
    pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=unet,
        controlnet=controlnet,
        torch_dtype=torch_dtype,
        safety_checker=None
    ).to(device)

    load_textual_inversion_embeds(pipeline, ti_path)

    try:
        init_image = Image.open(args.input_image_path).convert("RGB").resize((512, 512))
        mask_image = Image.open(args.mask_image_path).convert("L").resize((512, 512))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print("Generating Normal Map using reference-based post-processing...")
    control_image = generate_normal_map_for_inference(
        img=init_image,
        estimator=normal_estimator,
        use_marigold=args.use_marigold,
        master_normal_path=master_normal_path,
        device=device,
        processing_resolution=768
    )
    print("Successfully generated Normal Map.")

    debug_save_path = f"debug_normal_map_{args.category}_{Path(args.input_image_path).stem}.png"
    print(f"Saving debug normal map to: {debug_save_path}")
    control_image.save(debug_save_path)

    generator = torch.Generator(device=device).manual_seed(args.seed)

    print("Generating defect image...")
    result = pipeline(
        prompt=args.prompt,
        image=init_image,
        mask_image=mask_image,
        control_image=control_image,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.controlnet_scale,
        generator=generator
    ).images[0]

    filename_suffix = (
        f"cs{args.controlnet_scale}_"
        f"is{args.num_inference_steps}_"
        f"gs{args.guidance_scale}"
    )
    output_path = f"generated_{args.category}_{Path(args.input_image_path).stem}_{filename_suffix}.png"
    result.save(output_path)
    print(f"Saved result to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="runwayml/stable-diffusion-inpainting")
    parser.add_argument("--controlnet_model_name_or_path", type=str,
                        default="lllyasviel/control_v11p_sd15_normalbae")
    parser.add_argument("--lora_root_dir", type=str, required=True,
                        help="Root directory containing category subfolders with LoRA weights (e.g. ./checkpoints).")
    parser.add_argument("--category", type=str, required=True,
                        help="Category subfolder name under lora_root_dir.")
    parser.add_argument("--input_image_path", type=str, required=True,
                        help="Path to the clean input image.")
    parser.add_argument("--mask_image_path", type=str, required=True,
                        help="Path to the mask image for inpainting.")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Prompt with placeholder token. Auto-generated from mask path and info-map.json if not specified.")
    parser.add_argument("--info_map_path", type=str, default="./info-map.json",
                        help="Path to info-map.json (used for auto prompt generation).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--controlnet_scale", type=float, default=0.5)
    parser.add_argument("--num_inference_steps", type=int, default=100)
    parser.add_argument("--guidance_scale", type=float, default=10.0)
    parser.add_argument("--use_marigold", action="store_true", help="Use Marigold for normal map estimation (default uses NormalBae).")
    args = parser.parse_args()
    main(args)
