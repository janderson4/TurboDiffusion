# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import math
import torch
from einops import rearrange, repeat
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.v2 as T
import numpy as np

from imaginaire.utils.io import save_image_or_video
from imaginaire.utils import log

from rcm.datasets.utils import VIDEO_RES_SIZE_INFO
from rcm.utils.umt5 import clear_umt5_memory, get_umt5_embedding
from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface

from modify_model import tensor_kwargs, create_model

torch._dynamo.config.suppress_errors = True

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TurboDiffusion script for Wan2.1 T2V with Fixed Start/End Frames")
    parser.add_argument("--dit_path", type=str, required=True, help="Path to the DiT model checkpoint")
    parser.add_argument("--start_image_path", type=str, required=True, help="Path to the starting reference image")
    parser.add_argument("--end_image_path", type=str, required=True, help="Path to the ending reference image")
    parser.add_argument("--model", choices=["Wan2.1-1.3B", "Wan2.1-14B"], default="Wan2.1-1.3B")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--num_steps", type=int, choices=[1, 2, 3, 4], default=4)
    parser.add_argument("--sigma_max", type=float, default=80)
    parser.add_argument("--vae_path", type=str, default="checkpoints/Wan2.1_VAE.pth")
    parser.add_argument("--text_encoder_path", type=str, default="checkpoints/models_t5_umt5-xxl-enc-bf16.pth")
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--resolution", default="480p", type=str)
    parser.add_argument("--aspect_ratio", default="16:9", type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_path", type=str, default="output/fixed_ends_video.mp4")
    parser.add_argument("--attention_type", choices=["sla", "sagesla", "original"], default="sagesla")
    parser.add_argument("--sla_topk", type=float, default=0.1)
    parser.add_argument("--quant_linear", action="store_true")
    parser.add_argument("--default_norm", action="store_true")
    return parser.parse_args()

def preprocess_image(path, w, h, device):
    img = Image.open(path).convert("RGB")
    transform = T.Compose([
        T.ToImage(),
        T.Resize(size=(h, w), antialias=True),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return transform(img).unsqueeze(0).to(device=device)

if __name__ == "__main__":
    args = parse_arguments()

    # 1. Text Embedding
    log.info(f"Computing embedding for prompt: {args.prompt}")
    with torch.no_grad():
        text_emb = get_umt5_embedding(checkpoint_path=args.text_encoder_path, prompts=args.prompt).to(**tensor_kwargs)
    clear_umt5_memory()

    # 2. Model & Tokenizer Setup
    log.info(f"Loading DiT model from {args.dit_path}")
    net = create_model(dit_path=args.dit_path, args=args).cpu()
    tokenizer = Wan2pt1VAEInterface(vae_pth=args.vae_path)
    
    w, h = VIDEO_RES_SIZE_INFO[args.resolution][args.aspect_ratio]
    lat_t = tokenizer.get_latent_num_frames(args.num_frames)
    
    # 3. Reference Image Processing
    log.info(f"Preprocessing start and end images to {w}x{h}...")
    start_img_t = preprocess_image(args.start_image_path, w, h, tensor_kwargs["device"])
    end_img_t = preprocess_image(args.end_image_path, w, h, tensor_kwargs["device"])
    
    with torch.no_grad():
        # We encode images individually as single-frame videos to get the latents
        # Wan VAE expects (B, C, T, H, W)
        z_start = tokenizer.encode(start_img_t.unsqueeze(2)) # -> B, C, 1, H_lat, W_lat
        z_end = tokenizer.encode(end_img_t.unsqueeze(2))     # -> B, C, 1, H_lat, W_lat
    
    # 4. Latent Initialization
    generator = torch.Generator(device=tensor_kwargs["device"]).manual_seed(args.seed)
    state_shape = [tokenizer.latent_ch, lat_t, h // tokenizer.spatial_compression_factor, w // tokenizer.spatial_compression_factor]
    
    init_noise = torch.randn(args.num_samples, *state_shape, dtype=torch.float32, device=tensor_kwargs["device"], generator=generator)

    # 5. Timestep Setup (rCM / Rectified Flow)
    mid_t = [1.5, 1.4, 1.0][: args.num_steps - 1]
    t_steps = torch.tensor([math.atan(args.sigma_max), *mid_t, 0], dtype=torch.float64, device=init_noise.device)
    t_steps = torch.sin(t_steps) / (torch.cos(t_steps) + torch.sin(t_steps)) # Noise (1.0) to Clean (0.0)

    # 6. Sampling Loop with Frame Fixing
    x = init_noise.to(torch.float64) * t_steps[0]
    ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
    condition = {"crossattn_emb": repeat(text_emb.to(**tensor_kwargs), "b l d -> (k b) l d", k=args.num_samples)}
    
    net.cuda()
    for i, (t_cur, t_next) in enumerate(tqdm(list(zip(t_steps[:-1], t_steps[1:])), desc="Sampling", total=len(t_steps)-1)):
        # --- FIXING LOGIC ---
        # Map clean latents to the current noise level: x_t = (1-t)*z + t*epsilon
        # t_cur = 1.0 is noise, t_cur = 0.0 is clean
        with torch.no_grad():
            x[:, :, 0:1, :, :] = (1 - t_cur) * z_start + t_cur * init_noise[:, :, 0:1, :, :]
            x[:, :, -1:, :, :] = (1 - t_cur) * z_end + t_cur * init_noise[:, :, -1:, :, :]
        
        with torch.no_grad():
            v_pred = net(x_B_C_T_H_W=x.to(**tensor_kwargs), 
                         timesteps_B_T=(t_cur.float() * ones * 1000).to(**tensor_kwargs), 
                         **condition).to(torch.float64)
            
            # SDE update step
            x = (1 - t_next) * (x - t_cur * v_pred) + t_next * torch.randn(*x.shape, dtype=torch.float32, device=x.device, generator=generator)

    # Final latent replacement to ensure perfect alignment
    x[:, :, 0:1, :, :] = z_start
    x[:, :, -1:, :, :] = z_end
    
    samples = x.float()
    net.cpu()
    torch.cuda.empty_cache()

    # 7. Decode and Save
    with torch.no_grad():
        video = tokenizer.decode(samples)
    
    video_out = (1.0 + video.float().cpu().unsqueeze(0).clamp(-1, 1)) / 2.0
    save_image_or_video(rearrange(video_out, "n b c t h w -> c t (n h) (b w)"), args.save_path, fps=16)
    log.success(f"Video saved to {args.save_path}")
