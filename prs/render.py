import os
import copy
import shutil
import numpy as np

import torch
from torch import Tensor
from torch import autocast_mode, autocast

from pytorch_lightning import seed_everything
from einops import rearrange, repeat
from k_diffusion.external import CompVisDenoiser
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from contextlib import contextmanager, nullcontext
from tqdm import tqdm, trange

from prs.utils import thats_numberwang, slerp
from prs.denoiser import KCFGDenoiser, CFGDenoiser
from prs.image import load_img, save_image, metadata
from prs.randomizer import dynamic_value, randomize_prompt
from prs.settings import save_settings
from prs.prompt import split_weighted_subprompts

from k_diffusion.sampling import (
    sample_lms,
    sample_dpm_2,
    sample_dpm_2_ancestral,
    sample_euler,
    sample_euler_ancestral,
    sample_heun,
    get_sigmas_karras,
    append_zero,
    sample_dpmpp_2m,
    sample_dpmpp_2s_ancestral,
)

# samplers from the Karras et al paper
KARRAS_SAMPLERS = {
    "k_heun",
    "k_euler",
    "k_dpm_2",
    "k_dpmpp_2m_ka",
    "k_dpmpp_2s_ancestral_ka",
}
NON_KARRAS_K_DIFF_SAMPLERS = {
    "k_lms",
    "k_dpm_2_ancestral",
    "k_euler_ancestral",
    "k_dpmpp_2s_ancestral",
    "k_dpmpp_2m",
    "k_dpm_fast",
    "k_dpm_adaptive",
}
K_DIFF_SAMPLERS = {*KARRAS_SAMPLERS, *NON_KARRAS_K_DIFF_SAMPLERS}
NOT_K_DIFF_SAMPLERS = {"ddim", "plms"}
VALID_SAMPLERS = {*K_DIFF_SAMPLERS, *NOT_K_DIFF_SAMPLERS}


def do_run(device, model, opt):
    print(f"Starting render!")
    seed_everything(opt.seed)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = 1

    # prompt = opt.prompt
    data = [batch_size * [opt.prompt]]
    # data = opt.prompt

    # grid is a leftover from stable, but we use it to give our output file a unique name
    grid_count = thats_numberwang(outpath, opt.batch_name)

    progress_image = "progress.jpg" if opt.filetype == ".jpg" else "progress.png"

    if opt.improve_composition == True:
        opt.n_iter = (
            1  # TODO: allow multiple iterations when doing improved composition
        )

    if opt.method in K_DIFF_SAMPLERS:
        model_k_wrapped = CompVisDenoiser(model, quantize=True)
        model_k_guidance = KCFGDenoiser(model_k_wrapped)
    elif opt.method in NOT_K_DIFF_SAMPLERS:
        if opt.method == "plms":
            sampler = PLMSSampler(model, device)
        else:
            sampler = DDIMSampler(model, device)

    def img_to_latent(width, height, path: str, opt) -> Tensor:
        assert os.path.isfile(path)
        if device.type == "cuda":
            image = load_img(width, height, path, opt).to(device).half()
        else:
            image = load_img(width, height, path, opt).to(device)
        image = repeat(image, "1 ... -> b ...", b=batch_size)
        latent: Tensor = model.get_first_stage_encoding(
            model.encode_first_stage(image)
        )  # move to latent space
        return latent

    render_left_to_do = True
    compositional_init = None
    target_w = opt.W
    target_h = opt.H

    opt.ddim_steps = (
        int(dynamic_value(opt.ddim_steps))
        if type(opt.ddim_steps) == str
        else opt.ddim_steps
    )

    while render_left_to_do:
        if compositional_init != None:
            opt.init_image = compositional_init
            opt.W = target_w
            opt.H = target_h
            # opt.strength = 0.65 # might make this a separate variable later
        elif opt.improve_composition == True:
            opt.W = 512
            opt.H = 512
        if opt.init_image is not None:
            init_latent = img_to_latent(opt.W, opt.H, opt.init_image, opt)
            assert (
                0.0 <= opt.strength <= 1.0
            ), "can only work with strength in [0.0, 1.0]"
            t_enc = int(opt.strength * opt.ddim_steps)
        else:
            init_latent = None

        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]

        precision_scope = autocast if opt.precision == "autocast" else nullcontext
        # apple silicon support
        if device.type == "mps":
            precision_scope = nullcontext

        rand_size = [batch_size, *shape]
        og_start_code = (
            torch.randn(rand_size, device="cpu").to(device)
            if device.type == "mps"
            else torch.randn(rand_size, device=device)
        )
        start_code = og_start_code

        should_cancel = False
        # __matmul__, addbmm, addmm, addmv, addr, baddbmm, bmm, chain_matmul, multi_dot, conv1d, conv2d, conv3d, conv_transpose1d, conv_transpose2d, conv_transpose3d, GRUCell, linear, LSTMCell, matmul, mm, mv, prelu, RNNCell
        with torch.no_grad(), precision_scope("cuda"), model.ema_scope():
            for n in trange(opt.n_iter, desc="Sampling"):
                for prompts in tqdm(data, desc="data"):
                    if should_cancel:
                        continue

                    uc = None
                    # process dynamic values
                    scale = (
                        float(dynamic_value(opt.scale))
                        if type(opt.scale) == str
                        else opt.scale
                    )
                    ddim_eta = (
                        float(dynamic_value(opt.ddim_eta))
                        if type(opt.ddim_eta) == str
                        else opt.ddim_eta
                    )

                    if scale != 1.0:
                        uc = model.get_learned_conditioning(batch_size * [""])

                    # process the prompt for randomizers and dynamic values
                    # (don't do this after creating a compositional init, so we can keep the same prompt)
                    if compositional_init == None:
                        newprompts = []
                        for prompt in prompts:
                            if (
                                not ("skip_randomize" in vars(opt))
                                or opt.skip_randomize != False
                            ):
                                prompt = randomize_prompt(prompt)
                            prompt = dynamic_value(prompt)
                            newprompts.append(prompt)
                        prompts = newprompts

                    print(f"\nPrompt for this image:\n   {prompts}\n")
                    # split the prompt if it has : for weighting
                    normalize_prompt_weights = True
                    weighted_subprompts = split_weighted_subprompts(
                        prompts[0], normalize_prompt_weights
                    )

                    # save a settings file for this image
                    if opt.save_settings == True and opt.improve_composition == False:
                        used = copy.deepcopy(opt)
                        used.scale = scale
                        used.ddim_eta = ddim_eta
                        save_settings(prompts[0], grid_count, used)

                    # sub-prompt weighting used if more than 1
                    if len(weighted_subprompts) > 1:
                        c = torch.zeros_like(
                            uc
                        )  # i dont know if this is correct.. but it works
                        for i in range(0, len(weighted_subprompts)):
                            if weighted_subprompts[i][1] < 0:
                                uc = torch.zeros_like(uc)
                                break
                        for i in range(0, len(weighted_subprompts)):
                            tensor = model.get_learned_conditioning(
                                weighted_subprompts[i][0]
                            )
                            if weighted_subprompts[i][1] > 0:
                                c = torch.add(
                                    c, tensor, alpha=weighted_subprompts[i][1]
                                )
                            else:
                                uc = torch.add(
                                    uc, tensor, alpha=-weighted_subprompts[i][1]
                                )
                    else:  # just behave like usual
                        c = model.get_learned_conditioning(prompts)

                    if opt.variance != 0.0:
                        # add a little extra random noise to get varying output with same seed
                        base_x = og_start_code  # torch.randn(rand_size, device=device) * sigmas[0]
                        torch.manual_seed(opt.variance_seed + n)
                        target_x = (
                            torch.randn(rand_size, device="cpu").to(device)
                            if device.type == "mps"
                            else torch.randn(rand_size, device=device)
                        )
                        start_code = slerp(
                            device,
                            max(0.0, min(1.0, opt.variance)),
                            base_x,
                            target_x,
                        )

                    karras_noise = False

                    if opt.method in NOT_K_DIFF_SAMPLERS:
                        if init_latent is None or isinstance(sampler, PLMSSampler):
                            samples_ddim, _ = sampler.sample(
                                S=opt.ddim_steps,
                                conditioning=c,
                                batch_size=batch_size,
                                shape=shape,
                                verbose=False,
                                unconditional_guidance_scale=scale,
                                unconditional_conditioning=uc,
                                eta=ddim_eta,
                                x_T=start_code,
                            )
                            sigmas = None
                        else:
                            # encode (scaled latent)
                            z_enc = sampler.stochastic_encode(
                                init_latent,
                                torch.tensor([t_enc] * batch_size).to(device),
                            )
                            # decode it
                            samples = sampler.decode(
                                z_enc,
                                c,
                                t_enc,
                                unconditional_guidance_scale=scale,
                                unconditional_conditioning=uc,
                            )

                    else:
                        if opt.method == "k_dpm_2":
                            sampling_fn = sample_dpm_2
                            # karras_noise = True
                        elif opt.method == "k_dpm_2_ancestral":
                            sampling_fn = sample_dpm_2_ancestral
                        elif opt.method == "k_dpmpp_2m":
                            sampling_fn = sample_dpmpp_2m
                        elif opt.method == "k_dpmpp_2m_ka":
                            sampling_fn = sample_dpmpp_2m
                            karras_noise = True
                        elif opt.method == "k_dpmpp_2s_ancestral":
                            sampling_fn = sample_dpmpp_2s_ancestral
                        elif opt.method == "k_dpmpp_2s_ancestral_ka":
                            sampling_fn = sample_dpmpp_2s_ancestral
                            karras_noise = True
                        elif opt.method == "k_heun":
                            sampling_fn = sample_heun
                            karras_noise = True
                        elif opt.method == "k_euler":
                            sampling_fn = sample_euler
                            karras_noise = True
                        elif opt.method == "k_euler_ancestral":
                            sampling_fn = sample_euler_ancestral
                        else:
                            sampling_fn = sample_lms

                        noise_schedule_sampler_args = {}

                        if karras_noise:
                            end_karras_ramp_early = False  # this is only needed for really low step counts, not going to bother with it right now

                            def get_premature_sigma_min(
                                steps: int,
                                sigma_max: float,
                                sigma_min_nominal: float,
                                rho: float,
                            ) -> float:
                                min_inv_rho = sigma_min_nominal ** (1 / rho)
                                max_inv_rho = sigma_max ** (1 / rho)
                                ramp = (steps - 2) * 1 / (steps - 1)
                                sigma_min = (
                                    max_inv_rho + ramp * (min_inv_rho - max_inv_rho)
                                ) ** rho
                                return sigma_min

                            rho = 7.0
                            sigma_max = model_k_wrapped.sigmas[-1].item()
                            sigma_min_nominal = model_k_wrapped.sigmas[0].item()
                            premature_sigma_min = get_premature_sigma_min(
                                steps=opt.ddim_steps + 1,
                                sigma_max=sigma_max,
                                sigma_min_nominal=sigma_min_nominal,
                                rho=rho,
                            )
                            sigmas = get_sigmas_karras(
                                n=opt.ddim_steps,
                                sigma_min=premature_sigma_min
                                if end_karras_ramp_early
                                else sigma_min_nominal,
                                sigma_max=sigma_max,
                                rho=rho,
                                device=device,
                            )

                        else:
                            sigmas = model_k_wrapped.get_sigmas(opt.ddim_steps)

                        if init_latent is not None:
                            sigmas = sigmas[len(sigmas) - t_enc - 1 :]

                        x = start_code * sigmas[0]  # for GPU draw
                        if init_latent is not None:
                            x = init_latent + x

                        extra_args = {
                            "conditions": (c,),
                            "uncond": uc,
                            "cond_scale": scale,
                        }
                        samples = sampling_fn(
                            model_k_guidance,
                            x,
                            sigmas,
                            extra_args=extra_args,
                            **noise_schedule_sampler_args,
                        )

                    x_samples = model.decode_first_stage(samples)
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                    meta = metadata(prompts, scale, ddim_eta, opt)

                    for x_sample in x_samples:
                        x_sample = 255.0 * rearrange(
                            x_sample.cpu().numpy(), "c h w -> h w c"
                        )
                        output_filename = os.path.join(
                            outpath,
                            f"{opt.batch_name}{opt.device_id}-{grid_count:04}{opt.filetype}",
                        )

                        output_image = save_image(
                            x_sample.astype(np.uint8), progress_image, meta, opt
                        )

                        if (
                            opt.improve_composition == False
                        ):  # this is our actual output, so save it accordingly
                            shutil.copy2(progress_image, output_filename)
                            print(f'\nOutput saved as "{output_filename}"\n')
                            render_left_to_do = False
                            grid_count += 1
                        else:  # otherwise, we use this output as an init for another run
                            compositional_init = progress_image
                            opt.improve_composition = False
                            opt.prompt = prompts[0]
                            data = [
                                batch_size * [opt.prompt]
                            ]  # make sure we use the enhanced prompt instead of the original
                            print(
                                "\nImprove Composition enabled! Re-rendering at the desired size."
                            )
                        # TODO: allow current output_image to feed into a new run. Might want
                        # to make modifications to the image when we are making it. Like
                        # panning/zooming or 3d transforms

                        if opt.zoom_in and opt.zoom_in_amount > 0:
                            print("zooming in disabled")
                            # see zoom_in_WIP
                        else:
                            output_image.close()

    return output_filename
