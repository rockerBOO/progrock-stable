from PIL import Image
from PIL.PngImagePlugin import PngInfo
import numpy as np
import subprocess
import torch


def get_resampling_mode():
    try:
        from PIL import __version__, Image

        major_ver = int(__version__.split(".")[0])
        if major_ver >= 9:
            return Image.Resampling.LANCZOS
        else:
            return Image.LANCZOS
    except Exception as ex:
        return 1  # 'Lanczos' irrespective of version.


def load_img(w, h, path, opt):
    image = Image.open(path).convert("RGB")
    xw, xh = image.size
    if xw != w or xh != h:
        if opt.resize_method == "realesrgan":
            image = esrgan_resize(image, opt.device_id, opt.esrgan_model)
        image = image.resize((w, h), get_resampling_mode())
        image.convert("RGB")
        print(
            f"Warning: Init image size ({xw}x{xh}) differs from target size ({w}x{h})."
        )
        print(
            f"         It will be resized (if using improved composition mode, this is expected)"
        )
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def esrgan_resize(input, id, esrgan_model="realesrgan-x4plus"):
    input.save(f"_esrgan_orig{id}.png")
    input.close()
    try:
        subprocess.run(
            [
                "realesrgan-ncnn-vulkan",
                "-n",
                esrgan_model,
                "-i",
                "_esrgan_orig.png",
                "-o",
                "_esrgan_.png",
            ],
            stdout=subprocess.PIPE,
        ).stdout.decode("utf-8")
        output = Image.open("_esrgan_.png").convert("RGB")
        return output
    except Exception as e:
        print(
            "ESRGAN resize failed. Make sure realesrgan-ncnn-vulkan is in your path (or in this directory)"
        )
        print(e)
        quit()


def metadata(prompts, scale, ddim_eta, opt):
    metadata = PngInfo()
    if opt.hide_metadata == False:
        metadata.add_text("prompt", str(prompts))
        metadata.add_text("seed", str(opt.seed))
        metadata.add_text("steps", str(opt.ddim_steps))
        metadata.add_text("scale", str(scale))
        metadata.add_text("ETA", str(ddim_eta))
        metadata.add_text("method", str(opt.method))
        metadata.add_text("init_image", str(opt.init_image))
        metadata.add_text("variance", str(opt.variance))

    return metadata


def save_image(array, filename, metadata, opt):
    output_image = Image.fromarray(array)
    output_image.save(
        filename,
        pnginfo=metadata,
        quality=opt.quality,
    )

    return output_image
