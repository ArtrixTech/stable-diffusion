from op_sd_pack import OptimizedStableDiffusion
from swin_ir_pack import SwinIR
import matplotlib.pyplot as plt
import numpy as np

sd = OptimizedStableDiffusion(W=704,H=448)
sd_infer_option = OptimizedStableDiffusion.InferOption(
    "infinite hyperbolic intricate maze, futuristic eco warehouse made out of dead vines, glass mezzanine level, lots of windows, wood pallets, designed by Aesop, forest house surrounded by massive willow trees and vines, white exterior facade, in full frame, , exterior view, twisted house, 3d printed canopy, clay, earth architecture, cavelike interiors, convoluted spaces, hyper realistic, photorealism, octane render, unreal engine, 4k")
sd_infer_option.n_iter = 2

"""
swin = SwinIR(
    model_path='model_zoo/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth',
    task='lightweight_sr',
    preload_model=False)
"""
swin = SwinIR(
    model_path='model_zoo/swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth',
    task='real_sr',
    preload_model=False)

sd_result = sd.infer(sd_infer_option)

fig = plt.figure(figsize=(2, 2))
i = 0

for im in sd_result:

    swin_result = swin.infer(im)
    #swin_result.show()

    fig.add_subplot(2, 2, i+1)
    plt.imshow(np.asarray(im))
    i += 1

    fig.add_subplot(2, 2, i+1)
    plt.imshow(np.asarray(swin_result))
    i += 1
    

plt.show()

