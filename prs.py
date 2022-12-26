import argparse, os
import torch
from omegaconf import OmegaConf
import time
from prs.render import do_run
from prs.gobig import do_gobig
from prs.settings import Settings
from prs.model_loader import load_model_from_config

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



def main():
    print("\nPROG ROCK STABLE")
    print("----------------")
    print("")

    cl_args = parse_args()

    # Load the JSON config files
    settings = Settings()
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

    # setup the model
    ckpt = settings.checkpoint  # "./models/sd-v1-3-full-ema.ckpt"
    inf_config = "./configs/stable-diffusion/v1-inference.yaml"
    print(f"Loading the model and checkpoint ({ckpt})...")
    config = OmegaConf.load(f"{inf_config}")
    model = load_model_from_config(config, f"{ckpt}", verbose=False)

    if model == None:
        raise ValueError("could not load the model: %s", ckpt)

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

    starting_settings = (
        settings  # save our initial setup so we can get back to it if needed
    )

    print("Pytorch is using device:", device)

    if "cuda" in str(device):
        model.cuda()
    model.eval()

    # load the model to the device
    if "cuda" in str(device):
        torch.set_default_tensor_type(torch.HalfTensor)
        model = model.half()  # half-precision mode for gpus, saves vram, good good
    model = model.to(device)

    there_is_work_to_do = True
    while there_is_work_to_do:
        if cl_args.interactive:
            # Interactive mode waits for a job json, runs it, then goes back to waiting
            job_json = ("job_" + cl_args.device + ".json").replace(":", "_")
            print(f"\nInteractive Mode On! Waiting for {job_json}")
            job_ready = False
            while job_ready == False:
                if os.path.exists(job_json):
                    print(f"Job file found! Processing.")
                    settings = starting_settings
                    try:
                        with open(job_json, "r", encoding="utf-8") as json_file:
                            settings_file = json5.load(json_file)
                            settings.apply_settings_file(job_json, settings_file)
                            prompts = []
                            prompts.append(settings.prompt)
                        job_ready = True
                    except Exception as e:
                        print(
                            "Failed to open or parse "
                            + job_json
                            + " - Check formatting."
                        )
                        print(e)
                        os.remove(job_json)
                else:
                    time.sleep(0.5)

        # outdir = (f'{settings.out_path}/{settings.batch_name}')
        outdir = os.path.join(settings.out_path, settings.batch_name)
        print(f"Saving output to {outdir}")
        filetype = ".jpg" if settings.use_jpg == True else ".png"
        quality = 97 if settings.use_jpg else 100

        prompts = []
        if settings.from_file is not None:
            with open(settings.from_file, "r", encoding="utf-8") as f:
                prompts = f.read().splitlines()
        else:
            prompts.append(settings.prompt)

        for p in range(len(prompts)):
            for i in range(settings.n_batches):
                # pack up our settings into a simple namespace for the renderer
                opt = {
                    "prompt": prompts[p],
                    "checkpoint": settings.checkpoint,
                    "batch_name": settings.batch_name,
                    "outdir": outdir,
                    "ddim_steps": settings.steps,
                    "ddim_eta": settings.eta,
                    "n_iter": settings.n_iter,
                    "W": settings.width,
                    "H": settings.height,
                    "C": 4,
                    "f": 8,
                    "scale": settings.scale,
                    "dyn": settings.dyn,
                    "seed": settings.seed + i,
                    "variance": settings.variance,
                    "variance_seed": settings.seed + i + 1,
                    "precision": "autocast",
                    # Zoom in
                    "zoom_in": settings.zoom_in,
                    "zoom_in_amount": settings.zoom_in_amount,
                    "zoom_in_x": settings.zoom_in_x,
                    "zoom_in_y": settings.zoom_in_y,
                    "zoom_in_strength": settings.zoom_in_strength,
                    "zoom_in_depth": settings.zoom_in_depth,
                    "zoom_in_steps": settings.zoom_in_steps,
                    "init_image": settings.init_image,
                    "strength": 1.0 - settings.init_strength,
                    "resize_method": settings.resize_method,
                    "gobig": settings.gobig,
                    "gobig_init": settings.gobig_init,
                    "gobig_scale": settings.gobig_scale,
                    "gobig_prescaled": settings.gobig_prescaled,
                    "gobig_maximize": settings.gobig_maximize,
                    "gobig_overlap": settings.gobig_overlap,
                    "gobig_keep_slices": settings.gobig_keep_slices,
                    "esrgan_model": settings.esrgan_model,
                    "gobig_cgs": settings.gobig_cgs,
                    "augment_prompt": settings.augment_prompt,
                    "config": config,
                    "filetype": filetype,
                    "hide_metadata": settings.hide_metadata,
                    "quality": quality,
                    "device_id": device_id,
                    "method": settings.method,
                    "save_settings": settings.save_settings,
                    "improve_composition": settings.improve_composition,
                    "skip_randomize": True,
                }
                opt = SimpleNamespace(**opt)

                # render the image(s)!
                if settings.gobig_init == None:
                    # either just a regular render, or a regular render that will next go_big
                    try:
                        gobig_init = do_run(device, model, opt)
                    except OSError as err:
                        print("\nError failed to do something with the OS. %s\n" % err)
                        continue
                    except KeyboardInterrupt:
                        print("\nJob cancelled! And so we wait...\n")
                        continue
                else:
                    gobig_init = settings.gobig_init
                if settings.gobig:
                    do_gobig(gobig_init, device, model, opt)
                if settings.cool_down > 0 and (
                    (i < (settings.n_batches - 1)) or p < (len(prompts) - 1)
                ):
                    print(
                        f"Pausing {settings.cool_down} seconds to give your poor GPU a rest..."
                    )
                    time.sleep(settings.cool_down)
            if not settings.frozen_seed:
                settings.seed = settings.seed + 1
        if cl_args.interactive == False:
            # only doing one render, so we stop after this
            there_is_work_to_do = False
        else:
            print("\nJob finished! And so we wait...\n")
            os.remove(job_json)


if __name__ == "__main__":
    main()
