import argparse
import gc
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from diffusers import (ControlNetModel, MarigoldNormalsPipeline,
                       StableDiffusionControlNetInpaintPipeline,
                       UNet2DConditionModel)
from peft import PeftModel
from PIL import Image, ImageDraw
from torchvision.transforms import ToTensor
from transformers import pipeline

try:
    from controlnet_aux import NormalBaeDetector
except ImportError:
    print("Warning: controlnet_aux is not installed. `pip install controlnet_aux`")
    NormalBaeDetector = None


class DefectGeneratorVLM:
    def __init__(self, args):
        print("Initializing DefectGenerator for VLM...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.args = args

        with open(self.args.info_map_path, 'r') as f:
            self.info_map = json.load(f)

        try:
            self.available_categories = sorted([d.name for d in Path(self.args.lora_root_dir).iterdir() if d.is_dir()])
            if not self.available_categories:
                raise FileNotFoundError
            print(f"Found trained categories from LoRA directory: {self.available_categories}")
        except FileNotFoundError:
            raise RuntimeError(f"Could not find any trained category folders in '{self.args.lora_root_dir}'")

    def run_vlm_analysis(self):
        print("\n--- Step 1: Running VLM Analysis ---")
        vlm_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

        print(f"Loading VLM model: {self.args.vlm_model_id}...")
        vlm_pipe = pipeline(
            "image-text-to-text", model=self.args.vlm_model_id,
            device=self.device, torch_dtype=vlm_dtype
        )

        defect_context = "\n".join([f"- {k}: (related to '{' '.join(v['initializer_tokens'])}')" for k, v in self.info_map.items()])

        system_prompt = f"""You are an expert system for an image generation AI. Your task is to analyze a user's image and text prompt to generate a structured JSON output.

1.  **Analyze the image** and classify it into one of the following categories: {', '.join(self.available_categories)}.
2.  **Analyze the user's text prompt** to understand the desired defect.
3.  **Match the defect** to the most appropriate 'defect_key' from the list below.
4.  **Extract defect parameters** from the prompt:
    -   `area`: Return a LIST of found keywords. For "top left", return ["top", "left"]. For just "center", return ["center"]. Default to ["random"].
    -   `shape`: Choose from 'circle', 'ellipse', 'line' (for scratches), or 'random'.
    -   `size`: Choose from 'small', 'medium', 'large', or 'medium' if not specified.
5.  **Output ONLY a single valid JSON object** with the following structure: {{"category": "...", "defect_key": "...", "mask_params": {{"area": ["...", "..."], "shape": "...", "size": "..."}}}}

**Available Defect Keys:**
{defect_context}
"""
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "image", "image": Image.open(self.args.image_path)}, {"type": "text", "text": self.args.prompt}]}
        ]
        output = vlm_pipe(messages, max_new_tokens=200, do_sample=False)
        generated_json_str = output[0]["generated_text"][-1]["content"]
        print(f"VLM Raw Output:\n{generated_json_str}")

        vlm_output = None
        try:
            match = re.search(r'```json\s*([\s\S]*?)\s*```', generated_json_str)
            if match: generated_json_str = match.group(1)
            vlm_output = json.loads(generated_json_str)
        except json.JSONDecodeError:
            print("Error: VLM did not return valid JSON.")

        if vlm_output:
            display_output = vlm_output.copy()
            display_output.pop("mask_params", None)
            cat = display_output.get("category", "")
            d_key = display_output.get("defect_key", "")
            if cat and isinstance(d_key, str) and d_key.startswith(cat + "_"):
                display_output["defect_key"] = d_key[len(cat)+1:]
                
            print(f"\n[Final JSON Format Saved/Used internally]\n{json.dumps(display_output, indent=2)}")

        print("Releasing VLM from VRAM...")
        del vlm_pipe
        gc.collect()
        torch.cuda.empty_cache()

        return vlm_output

    def generate_image(self, vlm_output):
        if vlm_output is None:
            print("Halting generation due to VLM error.")
            return None

        print("\n--- Step 2: Generating Defect Image ---")
        category = vlm_output["category"]
        if category not in self.available_categories:
            print(f"Error: VLM returned category '{category}' which does not have a corresponding LoRA model.")
            return None

        defect_key = vlm_output["defect_key"]
        special_token = self.info_map[defect_key]["placeholder_tokens"][0]
        sd_prompt = f"a {category} with a {special_token} defect"

        print(f"Category: {category}")
        print(f"SD Prompt: '{sd_prompt}'")

        sd_dtype = torch.float16 if self.device == "cuda" else torch.float32

        if self.args.use_marigold:
            marigold_pipe = MarigoldNormalsPipeline.from_pretrained("prs-eth/marigold-normals-v1-1", torch_dtype=sd_dtype).to(self.device)
            normalbae_proc = None
        else:
            marigold_pipe = None
            normalbae_proc = NormalBaeDetector.from_pretrained("lllyasviel/Annotators").to(self.device) if NormalBaeDetector else None
        unet_lora_path = Path(self.args.lora_root_dir) / category / "unet_lora"
        ti_path = Path(self.args.lora_root_dir) / category / "learned_embeds.bin"
        unet = UNet2DConditionModel.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="unet", torch_dtype=sd_dtype)
        unet = PeftModel.from_pretrained(unet, str(unet_lora_path))
        controlnet = ControlNetModel.from_pretrained(self.args.controlnet_model_name_or_path, torch_dtype=sd_dtype)
        sd_pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            self.args.pretrained_model_name_or_path, unet=unet, controlnet=controlnet,
            torch_dtype=sd_dtype, safety_checker=None
        ).to(self.device)

        if ti_path.exists():
            learned_embeds = torch.load(ti_path, map_location=self.device)
            tokenizer, text_encoder = sd_pipe.tokenizer, sd_pipe.text_encoder
            tokenizer.add_tokens(list(learned_embeds.keys()))
            text_encoder.resize_token_embeddings(len(tokenizer))
            for token, embed in learned_embeds.items():
                text_encoder.get_input_embeddings().weight.data[tokenizer.convert_tokens_to_ids(token)] = embed

        source_image = Image.open(self.args.image_path).convert("RGB").resize((512, 512))
        control_image = _generate_normal_map_unified(source_image, category, marigold_pipe, normalbae_proc)

        if self.args.mask_path:
            mask_image = Image.open(self.args.mask_path).convert("L").resize((512, 512))
        else:
            mask_image = generate_mask_from_params(vlm_output)

        debug_dir = Path("./debug_inputs")
        debug_dir.mkdir(exist_ok=True)
        original_filename = Path(self.args.image_path).name
        source_image.save(debug_dir / f"debug_source_{original_filename}")
        control_image.save(debug_dir / "debug_control_normal_map.png")
        mask_image.save(debug_dir / "debug_mask.png")

        generator = torch.Generator(device=self.device).manual_seed(self.args.seed)

        result_image = sd_pipe(
            prompt=sd_prompt, image=source_image, mask_image=mask_image, control_image=control_image,
            num_inference_steps=self.args.num_inference_steps, guidance_scale=self.args.guidance_scale,
            controlnet_conditioning_scale=self.args.controlnet_scale, generator=generator,
            cross_attention_kwargs={"scale": self.args.lora_scale}
        ).images[0]

        print("\nGeneration Complete.")
        return result_image


def generate_mask_from_params(params: dict, size=(512, 512)) -> Image.Image:
    mask_details = params.get("mask_params", {})
    area_list = mask_details.get("area", ["random"])
    shape = mask_details.get("shape", "random")
    defect_size = mask_details.get("size", "medium")

    pad = 30
    img_w, img_h = size
    is_left = "left" in area_list
    is_right = "right" in area_list
    is_top = "top" in area_list
    is_bottom = "bottom" in area_list
    is_center = "center" in area_list

    if is_center:
        center_w, center_h = 256, 256
        x_range = ((img_w - center_w) // 2, (img_w + center_w) // 2)
        y_range = ((img_h - center_h) // 2, (img_h + center_h) // 2)
    elif is_top and is_left:
        corner_w, corner_h = 200, 200
        x_range = (pad, pad + corner_w); y_range = (pad, pad + corner_h)
    elif is_top and is_right:
        corner_w, corner_h = 200, 200
        x_range = (img_w - pad - corner_w, img_w - pad); y_range = (pad, pad + corner_h)
    elif is_bottom and is_left:
        corner_w, corner_h = 200, 200
        x_range = (pad, pad + corner_w); y_range = (img_h - pad - corner_h, img_h - pad)
    elif is_bottom and is_right:
        corner_w, corner_h = 200, 200
        x_range = (img_w - pad - corner_w, img_w - pad); y_range = (img_h - pad - corner_h, img_h - pad)
    elif is_left:
        edge_w = 140; x_range = (pad, pad + edge_w); y_range = (0, img_h)
    elif is_right:
        edge_w = 140; x_range = (img_w - pad - edge_w, img_w - pad); y_range = (0, img_h)
    elif is_top:
        edge_h = 140; x_range = (0, img_w); y_range = (pad, pad + edge_h)
    elif is_bottom:
        edge_h = 140; x_range = (0, img_w); y_range = (img_h - pad - edge_h, img_h - pad)
    else:
        x_range = (0, img_w); y_range = (0, img_h)

    size_map = {'small': (0.2, 0.4), 'medium': (0.4, 0.7), 'large': (0.7, 1.0)}
    min_ratio, max_ratio = size_map.get(defect_size, (0.4, 0.7))
    if shape == 'random': shape = random.choice(['circle', 'ellipse', 'line'])

    mask_image = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask_image)

    if shape == 'line':
        x1 = random.randint(*x_range); y1 = random.randint(*y_range)
        x2 = random.randint(*x_range); y2 = random.randint(*y_range)
        line_width = random.randint(3, 8)
        draw.line([(x1, y1), (x2, y2)], fill=255, width=line_width)
    else:
        center_x = random.randint(*x_range); center_y = random.randint(*y_range)
        max_rx = (x_range[1] - x_range[0]) / 2; max_ry = (y_range[1] - y_range[0]) / 2
        radius_x = int(max_rx * random.uniform(min_ratio, max_ratio))
        radius_y = int(max_ry * random.uniform(min_ratio, max_ratio))
        if shape == 'circle': radius = min(radius_x, radius_y); radius_x, radius_y = radius, radius
        x1 = center_x - radius_x; y1 = center_y - radius_y; x2 = center_x + radius_x; y2 = center_y + radius_y
        draw.ellipse([(x1, y1), (x2, y2)], fill=255)

    return mask_image


def _generate_normal_map_unified(source_img, category, marigold_pipe, normalbae_proc):
    if normalbae_proc:
        raw_normals_tensor = ToTensor()(normalbae_proc(source_img, detect_resolution=1024, image_resolution=1024)) * 2.0 - 1.0
    elif marigold_pipe:
        raw_normals_tensor = marigold_pipe(source_img, output_type="pt", processing_resolution=768).prediction[0]
    else:
        raise ValueError("No normal map estimator is available (controlnet_aux may not be installed).")
    master_normal_path = Path("./master_normals") / f"{category}.png"
    if not master_normal_path.exists():
        final_normals_tensor = raw_normals_tensor
    else:
        master_tensor = ToTensor()(Image.open(master_normal_path).convert("RGB").resize(raw_normals_tensor.shape[-2:])) * 2.0 - 1.0
        master_p_low_x, master_p_high_x = np.percentile(master_tensor.cpu().numpy()[0], [2, 98])
        master_p_low_y, master_p_high_y = np.percentile(master_tensor.cpu().numpy()[1], [2, 98])
        processed_np = raw_normals_tensor.cpu().numpy()
        if master_p_high_x > master_p_low_x:
            processed_np[0] = np.clip(-1.0 + 2.0 * (processed_np[0] - master_p_low_x) / (master_p_high_x - master_p_low_x), -1.0, 1.0)
        if master_p_high_y > master_p_low_y:
            processed_np[1] = np.clip(-1.0 + 2.0 * (processed_np[1] - master_p_low_y) / (master_p_high_y - master_p_low_y), -1.0, 1.0)
        processed_np[2] = np.sqrt((1.0 - processed_np[0]**2 - processed_np[1]**2).clip(0))
        final_normals_tensor = torch.from_numpy(processed_np)
    return TF.to_pil_image((final_normals_tensor * 0.5 + 0.5).cpu().clamp(0, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a defect image using a real VLM.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the normal source image.")
    parser.add_argument("--prompt", type=str, required=True, help="Natural language prompt describing the defect.")
    parser.add_argument("--mask_path", type=str, default=None, help="(Optional) Path to a specific mask image.")
    parser.add_argument("--output_dir", type=str, default='./generated_vlm', help="Directory to save the generated image.")
    parser.add_argument("--info_map_path", type=str, default='./info-map.json', help="Path to the info-map.json file.")
    parser.add_argument("--lora_root_dir", type=str, default='./checkpoints', help="Root directory for LoRA weights.")
    parser.add_argument("--vlm_model_id", type=str, default="google/gemma-3n-e4b-it", help="Hugging Face model ID for the VLM.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="runwayml/stable-diffusion-inpainting")
    parser.add_argument("--controlnet_model_name_or_path", type=str, default="lllyasviel/control_v11p_sd15_normalbae")
    parser.add_argument("--seed", type=int, default=random.randint(0, 2**32 - 1))
    parser.add_argument("--controlnet_scale", type=float, default=0.5)
    parser.add_argument("--num_inference_steps", type=int, default=100)
    parser.add_argument("--guidance_scale", type=float, default=10.0)
    parser.add_argument("--lora_scale", type=float, default=0.7, help="Strength of the LoRA weights' influence (0.0 to 1.0).")
    parser.add_argument("--use_marigold", action="store_true", help="Use Marigold for normal map estimation (default uses NormalBae).")
    args = parser.parse_args()

    generator = DefectGeneratorVLM(args)
    vlm_analysis_result = generator.run_vlm_analysis()
    generated_image = generator.generate_image(vlm_analysis_result)

    if generated_image:
        os.makedirs(args.output_dir, exist_ok=True)
        prompt_filename = re.sub(r'[\s<>:"/\\|?*]+', '_', args.prompt)
        output_filename = f"{Path(args.image_path).stem}_{prompt_filename[:50]}_{args.seed}.png"
        output_path = os.path.join(args.output_dir, output_filename)
        generated_image.save(output_path)
        print(f"\nImage saved to: {output_path}")
