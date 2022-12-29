import argparse, os
import torch
from omegaconf import OmegaConf
import time
from prs.render import do_run
from prs.gobig import do_gobig
from prs.settings import Settings
from prs.model_loader import load_model_from_config, load_vae, load_embedding

from torch.types import (
    _device,
)

from types import SimpleNamespace
import json5

try:
    # this silences the annoying "Some weights of the model checkpoint were not used when initializing..." message at start.
    from transformers import logging

    logging.set_verbosity_error()
except:
    pass

# # samplers from the Karras et al paper
# KARRAS_SAMPLERS = {
#     "k_heun",
#     "k_euler",
#     "k_dpm_2",
#     "k_dpmpp_2m_ka",
#     "k_dpmpp_2s_ancestral_ka",
# }
# NON_KARRAS_K_DIFF_SAMPLERS = {
#     "k_lms",
#     "k_dpm_2_ancestral",
#     "k_euler_ancestral",
#     "k_dpmpp_2s_ancestral",
#     "k_dpmpp_2m",
#     "k_dpm_fast",
#     "k_dpm_adaptive",
# }
# K_DIFF_SAMPLERS = {*KARRAS_SAMPLERS, *NON_KARRAS_K_DIFF_SAMPLERS}
# NOT_K_DIFF_SAMPLERS = {"ddim", "plms"}
# VALID_SAMPLERS = {*K_DIFF_SAMPLERS, *NOT_K_DIFF_SAMPLERS}


def parse_args():
    my_parser = argparse.ArgumentParser(
        prog="ProgRock-Stable",
        description="Generate images from text prompts, based on Stable Diffusion.",
    )
    my_parser.add_argument(
        "-s",
        "--settings",
        action="append",
        required=False,
        default=["settings.json"],
        help="A settings JSON file to use, best to put in quotes. Multiples are allowed and layered in order.",
    )
    my_parser.add_argument(
        "-o",
        "--output",
        action="store",
        required=False,
        help="What output directory to use within images_out",
    )
    my_parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        action="store",
        required=False,
        help="Override the prompt",
    )
    my_parser.add_argument(
        "-c",
        "--cpu",
        type=int,
        nargs="?",
        action="store",
        required=False,
        default=False,
        const=0,
        help="Force use of CPU instead of GPU, and how many threads to run",
    )
    my_parser.add_argument(
        "-n",
        "--n_batches",
        type=int,
        action="store",
        required=False,
        help="How many batches of images to generate",
    )
    my_parser.add_argument(
        "-i",
        "--n_iter",
        type=int,
        action="store",
        required=False,
        help="How many images to generate within a batch",
    )
    my_parser.add_argument(
        "--seed",
        type=int,
        action="store",
        required=False,
        help="Specify the numeric seed to be used",
    )
    my_parser.add_argument(
        "-f",
        "--from_file",
        action="store",
        required=False,
        help="A text file with prompts (one per line)",
    )
    my_parser.add_argument(
        "--gobig",
        action="store_true",
        required=False,
        help="After generation, the image is split into sections and re-rendered, to double the size.",
    )
    my_parser.add_argument(
        "--gobig_init",
        action="store",
        required=False,
        help="An image to use to kick off GO BIG mode, skipping the initial render.",
    )
    my_parser.add_argument(
        "--gobig_scale",
        action="store",
        type=int,
        default=2,
        required=False,
        help="What scale to multiply your original image by. 2 is a good value. 3 is insane. Anything more and I wish you luck.",
    )
    my_parser.add_argument(
        "--gobig_prescaled",
        action="store_true",
        required=False,
        help="Add this option if you have already upscaled the image you want to gobig on. The image and its resolution will be used.",
    )
    my_parser.add_argument(
        "--device",
        action="store",
        default="cuda:0",
        required=False,
        help="The device to use for pytorch.",
    )
    my_parser.add_argument(
        "--interactive",
        action="store_true",
        required=False,
        help="Advanced option for bots and such. Wait for a job file, render it, then wait some more.",
    )
    my_parser.add_argument(
        "--improve",
        action="store_true",
        required=False,
        help="Improve quality on larger images by first rendering a compositional 512x512 image.",
    )

    return my_parser.parse_args()


def validate_methods(settings: Settings):
    valid_methods = [
        "k_lms",
        "k_dpm_2_ancestral",
        "k_dpm_2",
        "k_heun",
        "k_dpmpp_2s_ancestral",
        "k_dpmpp_2m",
        "k_dpmpp_2s_ancestral_ka",
        "k_dpmpp_2m_ka",
        "k_euler_ancestral",
        "k_euler",
        "ddim",
    ]
    if any(settings.method in s for s in valid_methods):
        print(f"Using {settings.method} sampling method.")
    else:
        print(f"Method {settings.method} is not available. The valid choices are:")
        print(valid_methods)
        print()
        print(f"Falling back k_lms")
        settings.method = "k_lms"


def load_init_settings(cl_args, settings):
    for setting_arg in cl_args.settings:
        try:
            with open(setting_arg, "r", encoding="utf-8") as json_file:
                print(f"Parsing {setting_arg}")
                settings_file = json5.load(json_file)
                settings.apply_settings_file(setting_arg, settings_file)
        except Exception as e:
            print("Failed to open or parse " + setting_arg + " - Check formatting.")
            print(e)
            quit()

    # override settings from files with anything coming in from the command line
    if cl_args.prompt:
        settings.prompt = cl_args.prompt

    if cl_args.output:
        settings.batch_name = cl_args.output

    if cl_args.n_batches:
        settings.n_batches = cl_args.n_batches

    if cl_args.n_iter:
        settings.n_iter = cl_args.n_iter

    if cl_args.from_file:
        settings.from_file = cl_args.from_file

    if cl_args.seed:
        settings.seed = cl_args.seed

    if cl_args.gobig:
        settings.gobig = cl_args.gobig

    if cl_args.gobig_init:
        settings.gobig_init = cl_args.gobig_init

    if cl_args.gobig_prescaled:
        settings.gobig_prescaled = cl_args.gobig_prescaled

    return settings


def find_device(cl_args, settings):
    # setup the device
    device_id = ""  # leave this blank unless it's a cuda device
    if torch.cuda.is_available() and "cuda" in cl_args.device:
        device = torch.device(f"{cl_args.device}")
        device_id = (
            ("_" + cl_args.device.rsplit(":", 1)[1])
            if "0" not in cl_args.device
            else ""
        )
    elif ("mps" in cl_args.device) or (torch.backends.mps.is_available()):
        device = torch.device("mps")
        settings.method = (
            "ddim"  # k_diffusion currently not working on anything other than cuda
        )
    else:
        # fallback to CPU if we don't recognize the device name given
        device = torch.device("cpu")
        cores = os.cpu_count()
        if cores == None:
            raise ValueError("could not find the cpu cores")

        torch.set_num_threads(cores)
        settings.method = (
            "ddim"  # k_diffusion currently not working on anything other than cuda
        )

    return device_id


from typing import List


def load_prompts(settings: Settings) -> List[str]:
    if settings.from_file is not None:
        with open(settings.from_file, "r", encoding="utf-8") as f:
            prompts = f.read().splitlines()
    else:
        if settings.prompt:
            prompts = [settings.prompt]
        else:
            prompts = []

    return prompts


def load_job(job_json: str, settings: Settings) -> bool:
    try:
        # wait a small amount of time for the file to save completely.
        # Otherwise invalid JSON
        time.sleep(0.3)
        with open(job_json, "r", encoding="utf-8") as json_file:
            settings_file = json5.load(json_file)
            settings.apply_settings_file(job_json, settings_file)
            prompts = []
            prompts.append(settings.prompt)

        print("\nJob finished! And so we wait...\n")
        os.remove(job_json)
        return True
    except Exception as e:
        print("Failed to open or parse " + job_json + " - Check formatting.")
        print(e)
        os.remove(job_json)


import copy


def interactive(device: _device, settings: Settings):
    # Interactive mode waits for a job json, runs it, then goes back to waiting
    job_json = ("job_" + str(device) + ".json").replace(":", "_")
    print(f"\nInteractive Mode On! Waiting for {job_json}")

    job_ready = False
    while job_ready == False:
        if os.path.exists(job_json):
            print(f"Job file found! Processing.")
            job_ready = load_job(job_json, copy.deepcopy(settings))
        else:
            time.sleep(0.5)


def load_to_device(model: torch.nn.Module, device: _device):
    # load the model to the device
    # if "cuda" in str(device):
        # torch.set_default_tensor_type(torch.HalfTensor)
        # model = model.half()  # half-precision mode for gpus, saves vram, good good
    model = model.to(device)


def cooldown(prompts: List[str], settings: Settings, i: int):
    if settings.cool_down > 0 and (
        (i < (settings.n_batches - 1)) or p < (len(prompts) - 1)
    ):
        print(f"Pausing {settings.cool_down} seconds to give your poor GPU a rest...")
        time.sleep(settings.cool_down)


def batch(
    prompt: str,
    model: torch.nn.Module,
    device: _device,
    config,
    quality,
    filetype,
    outdir,
    settings: Settings,
    i: int,
):
    opt = settings_to_batch_opt(
        prompt, outdir, device, quality, filetype, i, config, settings
    )
    # render the image(s)!
    try:
        do_run(device, model, opt)
        return False
    except OSError as err:
        print("\nError failed to do something with the OS. %s\n" % err)
        return False
    except KeyboardInterrupt:
        print("\nJob cancelled! And so we wait...\n")
        return False


def to_device_id(device):
    return f"%s-%s" % (device.type, device.index)


def main():
    print("\nPROG ROCK STABLE")
    print("----------------")
    print("")

    cl_args = parse_args()

    # Load the JSON config files
    settings = Settings()

    load_init_settings(cl_args, settings)
    validate_methods(settings)

    # setup the model
    ckpt = settings.checkpoint  # "./models/sd-v1-3-full-ema.ckpt"
    inf_config = "./configs/stable-diffusion/v1-inference.yaml"
    print(f"Loading the model and checkpoint ({ckpt})...")
    config = OmegaConf.load(f"{inf_config}")
    model = load_model_from_config(config, f"{ckpt}", verbose=False)

    if model == None:
        raise ValueError("could not load the model: %s", ckpt)

    # device_id = "cuda" if torch.cuda.is_available() else ""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device_id = to_device_id(device) if torch.cuda.is_available() else ""

    # VAE
    load_vae(model, device, "/mnt/900/models/vae-ft-mse-840000-ema-pruned.ckpt", "cpu")
    # load_vae(model, device, "/mnt/900/models/vae/analog-diffusion-vae.bin", "cpu")

    load_embedding(
        model, device, "/mnt/900/builds/stablediffusion/embeddings/Style-Winter.pt"
    )

    print("Pytorch is using device:", device_id)

    if "cuda" in str(device):
        model.cuda()

    # inference mode
    model.eval()

    load_to_device(model, device)

    there_is_work_to_do = True
    while there_is_work_to_do:
        interactive(device, settings)

        # outdir = (f'{settings.out_path}/{settings.batch_name}')
        outdir = os.path.join(settings.out_path, settings.batch_name)
        print(f"Saving output to {outdir}")
        filetype = ".jpg" if settings.use_jpg == True else ".png"
        quality = 97 if settings.use_jpg else 100

        prompts = load_prompts(settings)

        for p in range(len(prompts)):
            for i in range(settings.n_batches):
                batch(
                    prompts[p],
                    model,
                    device,
                    config,
                    quality,
                    filetype,
                    outdir,
                    settings,
                    i,
                )
                cooldown(prompts, settings, i)
            if not settings.frozen_seed:
                settings.seed = (
                    settings.seed + 1
                    if isinstance(settings.seed, int)
                    else settings.seed
                )
        if cl_args.interactive == False:
            # only doing one render, so we stop after this
            there_is_work_to_do = False


if __name__ == "__main__":
    main()
