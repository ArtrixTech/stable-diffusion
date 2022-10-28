import argparse
import os
import re
from ctypes.wintypes import HACCEL
from tkinter import W
import torch
import numpy as np
from random import randint
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
from ldm.util import instantiate_from_config
from optimizedSD.optimUtils import split_weighted_subprompts, logger
from transformers import logging
# from samplers import CompVisDenoiser
logging.set_verbosity_error()

default_infer_config = "optimizedSD/v1-inference.yaml"
DEFAULT_CKPT = "models/ldm/stable-diffusion-v1/model.ckpt"


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    return sd


class OptimizedStableDiffusion:

    class InferOption:

        def __init__(self, prompt, ddim_steps=30, ddim_eta=0.0, n_iter=1, scale=7.5):

            self.prompt = prompt
            self.ddim_steps = ddim_steps
            self.ddim_eta = ddim_eta
            self.n_iter = n_iter
            self.scale = scale
            self.seed = randint(0, 1000000)

        def next_seed(self):
            self.seed += 1
            return self.seed

    def __init__(self, W=512, H=512, C=4, f=8, batch_size=1, device='cuda', ckpt=DEFAULT_CKPT, unet_bs=1, turbo=True, precision='autocast', format='png', sampler='euler_a'):

        self.W = W
        self.H = H
        self.C = C
        self.f = f
        self.batch_size = batch_size
        self.device = device
        self.ckpt = ckpt
        self.unet_bs = unet_bs
        self.turbo = turbo
        self.precision = precision
        self.format = format
        self.sampler = sampler

        # Init the Random Environment
        seed_everything(randint(0, 1000000))

        sd = load_model_from_config(f"{self.ckpt}")
        li, lo = [], []
        for key, value in sd.items():
            sp = key.split(".")
            if (sp[0]) == "model":
                if "input_blocks" in sp:
                    li.append(key)
                elif "middle_block" in sp:
                    li.append(key)
                elif "time_embed" in sp:
                    li.append(key)
                else:
                    lo.append(key)
        for key in li:
            sd["model1." + key[6:]] = sd.pop(key)
        for key in lo:
            sd["model2." + key[6:]] = sd.pop(key)

        config = OmegaConf.load(f"{default_infer_config}")

        self.model = instantiate_from_config(config.modelUNet)
        _, _ = self.model.load_state_dict(sd, strict=False)
        self.model.eval()
        self.model.unet_bs = self.unet_bs
        self.model.cdevice = self.device
        self.model.turbo = self.turbo

        self.modelCS = instantiate_from_config(config.modelCondStage)
        _, _ = self.modelCS.load_state_dict(sd, strict=False)
        self.modelCS.eval()
        self.modelCS.cond_stage_model.device = self.device

        self.modelFS = instantiate_from_config(config.modelFirstStage)
        _, _ = self.modelFS.load_state_dict(sd, strict=False)
        self.modelFS.eval()
        del sd

        if self.device != "cpu" and self.precision == "autocast":
            self.model.half()
            self.modelCS.half()

        self.start_code = None

        if self.precision == "autocast" and self.device != "cpu":
            self.precision_scope = autocast
        else:
            self.precision_scope = nullcontext

    def infer(self, infer_option: InferOption):

        assert infer_option.prompt is not None
        prompt = infer_option.prompt
        print(f"Using prompt: {prompt}")
        data = [self.batch_size * [prompt]]

        return_img_list=[]

        with torch.no_grad():

            all_samples = list()
            for n in trange(infer_option.n_iter, desc="Sampling"):
                for prompts in tqdm(data, desc="data"):

                    with self.precision_scope("cuda"):
                        self.modelCS.to(self.device)
                        uc = None
                        if infer_option.scale != 1.0:
                            uc = self.modelCS.get_learned_conditioning(
                                self.batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)

                        subprompts, weights = split_weighted_subprompts(
                            prompts[0])
                        if len(subprompts) > 1:
                            c = torch.zeros_like(uc)
                            totalWeight = sum(weights)
                            # normalize each "sub prompt" and add it
                            for i in range(len(subprompts)):
                                weight = weights[i]
                                # if not skip_normalize:
                                weight = weight / totalWeight
                                c = torch.add(c, self.modelCS.get_learned_conditioning(
                                    subprompts[i]), alpha=weight)
                        else:
                            c = self.modelCS.get_learned_conditioning(prompts)

                        shape = [self.batch_size, self.C,
                                 self.H // self.f, self.W // self.f]

                        if self.device != "cpu":
                            mem = torch.cuda.memory_allocated() / 1e6
                            self.modelCS.to("cpu")
                            while torch.cuda.memory_allocated() / 1e6 >= mem:
                                time.sleep(1)

                        self.modelFS.to(self.device)
                        self.modelFS.half()

                        samples_ddim = self.model.sample(
                            S=infer_option.ddim_steps,
                            conditioning=c,
                            seed=infer_option.seed,
                            shape=shape,
                            verbose=False,
                            unconditional_guidance_scale=infer_option.scale,
                            unconditional_conditioning=uc,
                            eta=infer_option.ddim_eta,
                            x_T=self.start_code,
                            sampler=self.sampler,
                            img_callback=img_cb,
                            optimized_sd_object=self
                        )

                        #self.modelFS.to(self.device)

                        print(samples_ddim.shape)
                        print("saving images")
                        for i in range(self.batch_size):

                            x_samples_ddim = self.modelFS.decode_first_stage(
                                samples_ddim[i].unsqueeze(0))
                            x_sample = torch.clamp(
                                (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            x_sample = 255.0 * \
                                rearrange(
                                    x_sample[0].cpu().numpy(), "c h w -> h w c")
                            
                            return_img_list.append(Image.fromarray(x_sample.astype(np.uint8)))

                            infer_option.next_seed()

                        if self.device != "cpu":
                            mem = torch.cuda.memory_allocated() / 1e6
                            self.modelFS.to("cpu")
                            while torch.cuda.memory_allocated() / 1e6 >= mem:
                                time.sleep(1)
                        del samples_ddim
                        print("memory_final = ",
                              torch.cuda.memory_allocated() / 1e6)
        torch.cuda.empty_cache()
        return return_img_list


def img_cb(optimized_sd_obj,x_pred, i):

    #=modelFS.to('cuda')
    x_samples_ddim = optimized_sd_obj.modelFS.decode_first_stage(x_pred)
    x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
    x_sample = 255.0 * rearrange(x_sample[0].cpu().numpy(), "c h w -> h w c")
    im = Image.fromarray(x_sample.astype(np.uint8))
    #plt.imshow(np.asarray(im))
    #plt.show()
    # im.show()


if __name__ == "__main__":
    infer_option = OptimizedStableDiffusion.InferOption("domestic cat walking on street with tails lifted, detailed, 8k wallpaper, realistic, cute, good shape")
    infer_option.n_iter = 2

    import matplotlib.pyplot as plt
    import numpy as np

    sd=OptimizedStableDiffusion()
    result=sd.infer(infer_option)

    fig = plt.figure(figsize=(2, 1))

    i=0
    for im in result:
        fig.add_subplot(2, 1, i+1)
        plt.imshow(np.asarray(im))
        
        i+=1
    plt.show()