import argparse
import datetime
import logging
import inspect
import math
import os
import random
import gc
import copy

from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms as T
import diffusers
import transformers

from torchvision import transforms
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from mymodels.unet_3d_condition_it import UNet3DConditionModel

from mydiffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

from diffusers import DPMSolverMultistepScheduler, DDPMScheduler, TextToVideoSDPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, export_to_video
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention_processor import AttnProcessor2_0, Attention
from diffusers.models.attention import BasicTransformerBlock
from transformers import AutoProcessor, CLIPVisionModel, CLIPVisionModelWithProjection

from transformers import CLIPTextModel, CLIPTokenizer
from transformers.models.clip.modeling_clip import CLIPEncoder
from utils.dataset import VideoJsonDataset, SingleVideoDataset, \
    ImageDataset, VideoFolderDataset, CachedDataset
from einops import rearrange, repeat
from utils.lora_handler import LoraHandler, LORA_VERSIONS


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, q, k, v):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        attn = F.softmax(attn, dim=-1)

        output = torch.matmul(attn, v)
        return output, attn
    
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head=16, d_model=1024, d_k=64, d_v=64):

        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        q, attn = self.attention(q, k, v)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.fc(q)
        q += residual
        q = self.layer_norm(q)
        return q
    
class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in=1024, d_hid=512):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x += residual

        x = self.layer_norm(x)

        return x

class ResidualMLP(nn.Module):
    def __init__(self):
        super(ResidualMLP, self).__init__()
        self.linear1 = nn.Linear(1280, 1024)
        self.linear2 = nn.Linear(1024, 1024)

    def forward(self, x):

        out1 = self.linear1(x) 
        out2 = self.linear2(out1)

        return out2 + out1

class Crossattn(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, ):
        super().__init__()
        self.cros_attn = MultiHeadAttention()
        self.pos_ffn = PositionwiseFeedForward()


    def forward(self, img1, img2):

        dec_output = self.cros_attn(img1, img2, img2)
        dec_output = self.pos_ffn(dec_output)
        return dec_output


class FeatureExtractor(torch.nn.Module):
    """
    FeatureExtractor for FPN
    """

    def __init__(
        self,
        pretrained_model_path='./Text-To-Video-Finetuning/mymodels/model_scope_diffusers', num_channels=[128, 256, 512, 1024]
    ):
        super().__init__()
        self.noise_scheduler = DDPMScheduler.from_config(pretrained_model_path, subfolder="scheduler")
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
        self.dim_tokens_enc = [512, 960, 1600, 1920]
        self.layer_dims = num_channels
        self.unet = UNet3DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
        self.cross_attn = Crossattn()

        self.act_1_postprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=self.dim_tokens_enc[0],
                out_channels=self.layer_dims[0] * 2,
                kernel_size=1, stride=1, padding=0,
            ),
            nn.Conv2d(
                in_channels=self.layer_dims[0] * 2,
                out_channels=self.layer_dims[0],
                kernel_size=1, stride=1, padding=0,
            )
        )

        self.act_2_postprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=self.dim_tokens_enc[1],
                out_channels=self.layer_dims[1] * 2,
                kernel_size=1, stride=1, padding=0,
            ),
            nn.Conv2d(
                in_channels=self.layer_dims[1] * 2,
                out_channels=self.layer_dims[1],
                kernel_size=1, stride=1, padding=0,
            )
        )

        self.act_3_postprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=self.dim_tokens_enc[2],
                out_channels=self.layer_dims[2] * 2,
                kernel_size=1, stride=1, padding=0,
            ),
            nn.Conv2d(
                in_channels=self.layer_dims[2] * 2,
                out_channels=self.layer_dims[2],
                kernel_size=1, stride=1, padding=0,
            )
        )

        self.act_4_postprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=self.dim_tokens_enc[3],
                out_channels=self.layer_dims[3],
                kernel_size=1, stride=1, padding=0,
            ),
            nn.Conv2d(
                in_channels=self.layer_dims[3],
                out_channels=self.layer_dims[3],
                kernel_size=1, stride=1, padding=0,
            )
        )

        self.act_postprocess = nn.ModuleList([
            self.act_1_postprocess,
            self.act_2_postprocess,
            self.act_3_postprocess,
            self.act_4_postprocess
        ])


        self.cv_model = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.projection = ResidualMLP()
        self.noise_pred = nn.Conv2d(4, 4, 3, padding=1)
        self.noise_fc = nn.Sequential(
            nn.Linear(4, 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(4, 4, bias=False),
            nn.Sigmoid()
        )
        self.cv_processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.freeze_models([self.vae, self.text_encoder, self.unet, self.cv_model])
        self.unet.eval()
        self.text_encoder.eval()
        self.cv_model.eval()


    def get_prompt_ids(self, prompt):
        prompt_ids = self.tokenizer(
                prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
        ).input_ids

        return prompt_ids

    def freeze_models(self, models_to_freeze):
        for model in models_to_freeze:
            if model is not None: model.requires_grad_(False) 

    def tensor_to_vae_latent(self, t):
        video_length = t.shape[1]

        t = rearrange(t, "b f c h w -> (b f) c h w")
        latents, rt_feature = self.vae.encode(t).latent_dist
        latents = latents.sample()
        # latents = vae.encode(t).latent_dist.sample()
        latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
        ft1_all = []
        ft2_all = []
        for eft in rt_feature:
            ft1_all.append(eft[0])
            ft2_all.append(eft[1])
        ft1_all = torch.cat(ft1_all)
        ft2_all = torch.cat(ft2_all)
        latents = latents * 0.18215
        return latents, [ft1_all, ft2_all]
    
    def sample_noise(self, latents, noise_strength, use_offset_noise=False):
        b ,c, f, *_ = latents.shape
        noise_latents = torch.randn_like(latents, device=latents.device)
        offset_noise = None

        if use_offset_noise:
            offset_noise = torch.randn(b, c, f, 1, 1, device=latents.device)
            noise_latents = noise_latents + noise_strength * offset_noise
        return noise_latents

    def forward(self, batch):
        pixel_values = batch["pixel_values"]
        original_pixel_values = ((pixel_values + 1) * 127.5).to(dtype=torch.int)

        video_length = pixel_values.shape[1]

        original_pixel_values = rearrange(original_pixel_values, "b f c h w -> (b f) c h w")
        cv_inputs = self.cv_processor(images=original_pixel_values, return_tensors="pt")
        cv_inputs = {k: v.cuda() for k, v in cv_inputs.items()}

        image_tokens = self.cv_model(**cv_inputs).last_hidden_state

        token_ids = self.get_prompt_ids(batch["captions"])
        encoder_hidden_states = self.text_encoder(token_ids.cuda())[0]
        encoder_hidden_states_rp = encoder_hidden_states.repeat_interleave(repeats=video_length, dim=0)
        image_tokens = self.projection(image_tokens)
        encoder_hidden_states_c = self.cross_attn(encoder_hidden_states_rp, image_tokens)
        
        latents, ret_ft_e = self.tensor_to_vae_latent(pixel_values)

        latents_for_noise = rearrange(latents, "b c f h w -> (b f) c h w")

        latents_for_noise = self.noise_pred(latents_for_noise)

        latents_for_noise = rearrange(latents_for_noise, "b c h w -> b h w c")

        latents_for_noise = self.noise_fc(latents_for_noise)

        noise_pre = rearrange(latents_for_noise, "(b f) h w c -> b c f h w ", f=video_length)


        # Get video length
        video_length = latents.shape[2]
  
        # Sample noise that we'll add to the latents
        # noise = self.sample_noise(latents, 0.1)
        bsz = latents.shape[0]
        noise = noise_pre 
        noise_mean = torch.mean(noise)
        noise_std = torch.std(noise)

        noise = (noise - noise_mean) / (noise_std + 1e-5) 

        # Sample a random timestep for each video
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        timesteps = torch.zeros_like(timesteps) 
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
            
        # # Encode text embeddings
        # token_ids = batch['prompt_ids']

        # # Assume extra batch dimnesion.
        # if len(token_ids.shape) > 2:
        #     token_ids = token_ids[0]
            
        # encoder_hidden_states = self.text_encoder(token_ids)[0]
        model_pred, ret_ft_unet = self.unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states_c).sample

        final_ret_ft = []
        for ei in ret_ft_e:
            final_ret_ft.append(ei)
        for ei in ret_ft_unet:
            final_ret_ft.append(ei)
        
        my_decout_f = []
        my_decout_f.append(final_ret_ft[1])
        final_ret_ft = final_ret_ft[2:]
        for i in range(len(final_ret_ft)//2):
            try:
                my_decout_f.append(torch.cat([final_ret_ft[i], final_ret_ft[len(final_ret_ft) - i -1]], dim=1))
            except:
                a = final_ret_ft[i]
                b = final_ret_ft[len(final_ret_ft) - i -1]
                a = a[:,:,0:b.size(2),0:b.size(3)]
                my_decout_f.append(torch.cat([a,b], dim=1))
        layers = my_decout_f
        my_decout = [self.act_postprocess[idx](l) for idx, l in enumerate(layers)]
        
        return  my_decout


