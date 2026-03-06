import argparse
import math
import os
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    UNet2DConditionModel
)
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, CLIPTextModel
from peft import LoraConfig, get_peft_model
import lpips
from dataset import MVTecADDatasetAugmented

def setup_textual_inversion(tokenizer, text_encoder, info_map):

    tokens = [t for v in info_map.values() for t in v["placeholder_tokens"]]
    tokens = sorted(set(tokens))
    tokenizer.add_tokens(tokens)
    text_encoder.resize_token_embeddings(len(tokenizer))

    with torch.no_grad():
        original_embeddings = text_encoder.get_input_embeddings().weight.data.clone()

    for defect in info_map.values():
        init_ids = tokenizer.convert_tokens_to_ids(defect["initializer_tokens"])
        avg_emb = original_embeddings[init_ids].mean(0)

        for placeholder in defect["placeholder_tokens"]:
            pid = tokenizer.convert_tokens_to_ids(placeholder)
            with torch.no_grad():
                text_encoder.get_input_embeddings().weight.data[pid] = avg_emb

    return list(text_encoder.get_input_embeddings().parameters())

def train_single_category(args, category, output_dir, info_map):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb",
        project_dir=output_dir
    )
    set_seed(args.seed)
    if accelerator.is_main_process:
        accelerator.init_trackers(f"defect_generation_{category}", config=vars(args))

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )

    ti_parameters = setup_textual_inversion(tokenizer, text_encoder, info_map)

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    controlnet = ControlNetModel.from_pretrained(
        args.controlnet_model_name_or_path
    )

    vae.requires_grad_(False)
    controlnet.requires_grad_(False)
    for module in [
        text_encoder.text_model.encoder,
        text_encoder.text_model.final_layer_norm,
        text_encoder.text_model.embeddings.position_embedding
    ]:
        module.requires_grad_(False)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, lora_config)

    optim_params = list(filter(lambda p: p.requires_grad, unet.parameters())) + ti_parameters
    optimizer = torch.optim.AdamW(optim_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    train_dataset = MVTecADDatasetAugmented(
        args.dataset_root_dir, tokenizer, info_map,
        args.resolution, category, 'train'
    )
    val_dataset = MVTecADDatasetAugmented(
        args.dataset_root_dir, tokenizer, info_map,
        args.resolution, category, 'val'
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.train_batch_size,
        shuffle=True, num_workers=args.dataloader_num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.train_batch_size,
        shuffle=False, num_workers=args.dataloader_num_workers, pin_memory=True
    )

    lpips_fn = lpips.LPIPS(net='vgg').to(accelerator.device).eval()

    total_steps = math.ceil(
        len(train_loader) / args.gradient_accumulation_steps
    ) * args.num_train_epochs
    lr_scheduler = get_scheduler(
        args.lr_scheduler, optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=total_steps
    )

    unet, optimizer, train_loader, val_loader, lr_scheduler, text_encoder =        accelerator.prepare(
            unet, optimizer, train_loader, val_loader, lr_scheduler, text_encoder
        )
    vae.to(accelerator.device)
    controlnet.to(accelerator.device)

    best_val_loss = float('inf')
    global_step = 0

    for epoch in range(args.num_train_epochs):
        unet.train()
        text_encoder.train()
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")
        for batch in train_bar:
            with accelerator.accumulate(unet):
                with torch.no_grad():
                    target_latents = vae.encode(
                        batch['pixel_values_target'].to(vae.dtype)
                    ).latent_dist.mode() * vae.config.scaling_factor

                    input_latents = vae.encode(
                        batch['pixel_values_input'].to(vae.dtype)
                    ).latent_dist.mode() * vae.config.scaling_factor

                noise = torch.randn_like(target_latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (target_latents.shape[0],), device=target_latents.device
                ).long()
                noisy_latents = noise_scheduler.add_noise(target_latents, noise, timesteps)

                bs = target_latents.shape[0]
                do_uncond = torch.rand(bs, device=target_latents.device) < 0.1
                input_ids = batch['input_ids'].clone()
                input_ids[do_uncond] = tokenizer.pad_token_id
                encoder_states = text_encoder(input_ids)[0]

                with torch.no_grad():
                    downs, mid = controlnet(
                        noisy_latents.to(controlnet.dtype),
                        timesteps,
                        encoder_states.to(controlnet.dtype),
                        controlnet_cond=batch['control_image'].to(controlnet.dtype),
                        return_dict=False
                    )

                mask = F.interpolate(batch['pixel_values_mask'], size=target_latents.shape[-2:])
                context_latents = input_latents * (1 - mask)
                model_input = torch.cat([noisy_latents, mask, context_latents], dim=1)

                pred_noise = unet(
                    model_input,
                    timesteps,
                    encoder_states,
                    down_block_additional_residuals=[d.to(unet.dtype) for d in downs],
                    mid_block_additional_residual=mid.to(unet.dtype)
                ).sample

                masked_pred = pred_noise * mask
                masked_noise = noise * mask
                loss_l2 = F.mse_loss(
                    masked_pred.float(), 
                    masked_noise.float(), 
                    reduction='sum'
                ) / (mask.sum() + 1e-8)

                alphas = noise_scheduler.alphas_cumprod.to(timesteps.device)
                sqrt_alpha = alphas[timesteps].sqrt().view(-1,1,1,1)
                sqrt_1_alpha = (1 - alphas[timesteps]).sqrt().view(-1,1,1,1)
                pred_original = (
                    (noisy_latents.float() - sqrt_1_alpha * pred_noise.float()) / sqrt_alpha
                ).clamp(-4,4)
                decoded = vae.decode(pred_original / vae.config.scaling_factor).sample
                loss_lpips = lpips_fn(
                    decoded.clamp(-1,1),
                    batch['pixel_values_target'].clamp(-1,1)
                ).mean()

                total_loss = args.w_l2 * loss_l2 + args.w_lpips * loss_lpips
                accelerator.backward(total_loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            accelerator.log({
                'train_loss': total_loss.item(), 'l2': loss_l2.item(), 'lpips': loss_lpips.item()
            }, step=global_step)
            train_bar.set_postfix(train_loss=total_loss.item())
        train_bar.close()

        unet.eval()
        text_encoder.eval()
        val_loss_sum = 0.0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1} Validation")
        with torch.no_grad():
            for batch in val_bar:
                target_latents = vae.encode(
                    batch['pixel_values_target'].to(vae.dtype)
                ).latent_dist.mode() * vae.config.scaling_factor

                input_latents = vae.encode(
                    batch['pixel_values_input'].to(vae.dtype)
                ).latent_dist.mode() * vae.config.scaling_factor

                noise = torch.randn_like(target_latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (target_latents.shape[0],), device=target_latents.device
                ).long()
                noisy_latents = noise_scheduler.add_noise(target_latents, noise, timesteps)
                encoder_states = text_encoder(batch['input_ids'])[0]

                downs, mid = controlnet(
                    noisy_latents.to(controlnet.dtype),
                    timesteps,
                    encoder_states.to(controlnet.dtype),
                    controlnet_cond=batch['control_image'].to(controlnet.dtype),
                    return_dict=False
                )

                mask = F.interpolate(batch['pixel_values_mask'], size=target_latents.shape[-2:])
                context_latents = input_latents * (1 - mask)
                model_input = torch.cat([noisy_latents, mask, context_latents], dim=1)

                pred_noise = unet(
                    model_input,
                    timesteps,
                    encoder_states,
                    down_block_additional_residuals=[d.to(unet.dtype) for d in downs],
                    mid_block_additional_residual=mid.to(unet.dtype)
                ).sample

                masked_pred = pred_noise * mask
                masked_noise = noise * mask
                loss_l2 = F.mse_loss(
                    masked_pred.float(), 
                    masked_noise.float(), 
                    reduction='sum'
                ) / (mask.sum() + 1e-8)
                alphas = noise_scheduler.alphas_cumprod.to(timesteps.device)
                sqrt_alpha = alphas[timesteps].sqrt().view(-1,1,1,1)
                sqrt_1_alpha = (1 - alphas[timesteps]).sqrt().view(-1,1,1,1)
                pred_original = ((noisy_latents.float() - sqrt_1_alpha * pred_noise.float()) / sqrt_alpha).clamp(-4,4)
                decoded = vae.decode(pred_original / vae.config.scaling_factor).sample
                loss_lpips = lpips_fn(
                    decoded.clamp(-1,1), batch['pixel_values_target'].clamp(-1,1)
                ).mean()

                val_loss = args.w_l2 * loss_l2 + args.w_lpips * loss_lpips
                val_loss_sum += val_loss.item()
                val_bar.set_postfix(val_loss=val_loss.item())

        avg_val_loss = val_loss_sum / len(val_loader)
        accelerator.log({'val_loss': avg_val_loss}, step=global_step)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            accelerator.print(f"New best model at epoch {epoch+1} with val_loss {best_val_loss:.4f}, saving...")
            if accelerator.is_main_process:
                accelerator.unwrap_model(unet).save_pretrained(
                    os.path.join(output_dir, 'unet_lora'),
                    safe_serialization=True
                )

                te_unwrapped = accelerator.unwrap_model(text_encoder)
                learned_embeds = {}

                for defect in info_map.values():
                    for placeholder in defect['placeholder_tokens']:
                        pid = tokenizer.convert_tokens_to_ids(placeholder)
                        if pid != tokenizer.unk_token_id:
                            learned_embeds[placeholder] = te_unwrapped.get_input_embeddings().weight.data[pid].cpu()
                torch.save(learned_embeds, os.path.join(output_dir, 'learned_embeds.bin'))

    accelerator.end_training()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_model_name_or_path", default="runwayml/stable-diffusion-inpainting")
    parser.add_argument("--controlnet_model_name_or_path", default="lllyasviel/control_v11p_sd15_normalbae")
    parser.add_argument("--output_dir", default="./checkpoints")
    parser.add_argument("--dataset_root_dir", default="./mvtec_anomaly_detection")
    parser.add_argument("--info_map_path", default="./info-map.json")
    parser.add_argument("--category", default=None)
    parser.add_argument("--w_l2", type=float, default=1.0)
    parser.add_argument("--w_lpips", type=float, default=0.5)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=1000)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", choices=["no","fp16","bf16"], default="fp16")
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    args = parser.parse_args()

    with open(args.info_map_path) as f:
        info_map = json.load(f)
    root = Path(args.dataset_root_dir)
    categories = [args.category] if args.category else [d.name for d in root.iterdir() if d.is_dir()]

    for category in categories:
        print(f"--- Starting training for category: {category} ---")
        category_output_dir = os.path.join(args.output_dir, category)
        os.makedirs(category_output_dir, exist_ok=True)

        category_info_map = {k: v for k, v in info_map.items() if k.startswith(category)}

        train_single_category(args, category, category_output_dir, category_info_map)
