from prs.utils import is_json_key_present
import random
from json import dump

class Settings:
    prompt = "A druid in his shop, selling potions and trinkets, fantasy painting by raphael lacoste and craig mullins"
    batch_name = "default"
    out_path = "./out"
    n_batches = 1
    steps = 50
    eta = 0.0
    n_iter = 1
    width = 512
    height = 512
    scale = 5.0
    dyn = None
    from_file = None
    seed = "random"
    variance = 0.0
    frozen_seed = False
    zoom_in = False
    zoom_in_amount = 1
    zoom_in_depth = 2
    zoom_in_x = False
    zoom_in_y = False
    zoom_in_strength = 0.5
    zoom_in_steps = False
    init_image = None
    init_strength = 0.5
    resize_method = "basic"
    gobig = False
    gobig_init = None
    gobig_prescaled = False
    gobig_maximize = True
    gobig_overlap = 64
    gobig_realesrgan = False
    gobig_keep_slices = False
    gobig_cgs = None
    augment_prompt = None
    esrgan_model = "realesrgan-x4plus"
    cool_down = 0.0
    checkpoint = "./models/sd-v1-4.ckpt"
    use_jpg = False
    hide_metadata = False
    method = "k_lms"
    save_settings = False
    improve_composition = False

    def apply_settings_file(self, filename, settings_file):
        print(f"Applying settings file: {filename}")
        if is_json_key_present(settings_file, "prompt"):
            self.prompt = settings_file["prompt"]
        if is_json_key_present(settings_file, "batch_name"):
            self.batch_name = settings_file["batch_name"]
        if is_json_key_present(settings_file, "out_path"):
            self.out_path = settings_file["out_path"]
        if is_json_key_present(settings_file, "n_batches"):
            self.n_batches = settings_file["n_batches"]
        if is_json_key_present(settings_file, "steps"):
            self.steps = settings_file["steps"]
        if is_json_key_present(settings_file, "eta"):
            self.eta = settings_file["eta"]
        if is_json_key_present(settings_file, "n_iter"):
            self.n_iter = settings_file["n_iter"]
        if is_json_key_present(settings_file, "width"):
            self.width = settings_file["width"]
        if is_json_key_present(settings_file, "height"):
            self.height = settings_file["height"]
        if is_json_key_present(settings_file, "scale"):
            self.scale = settings_file["scale"]
        if is_json_key_present(settings_file, "dyn"):
            self.dyn = settings_file["dyn"]
        if is_json_key_present(settings_file, "from_file"):
            self.from_file = settings_file["from_file"]
        if is_json_key_present(settings_file, "seed"):
            self.seed = settings_file["seed"]
            if self.seed == "random":
                self.seed = random.randint(1, 10000000)
        if is_json_key_present(settings_file, "variance"):
            self.variance = settings_file["variance"]
        if is_json_key_present(settings_file, "frozen_seed"):
            self.frozen_seed = settings_file["frozen_seed"]
        if is_json_key_present(settings_file, "init_strength"):
            self.init_strength = settings_file["init_strength"]
        if is_json_key_present(settings_file, "zoom_in"):
            zoom_in = settings_file["zoom_in"]
            if is_json_key_present(zoom_in, "depth"):
                self.zoom_in_depth = zoom_in["depth"]
            if is_json_key_present(zoom_in, "x"):
                self.zoom_in_x = zoom_in["x"]
            if is_json_key_present(zoom_in, "y"):
                self.zoom_in_y = zoom_in["y"]
            if is_json_key_present(zoom_in, "strength"):
                self.zoom_in_strength = zoom_in["strength"]
            if is_json_key_present(zoom_in, "steps"):
                self.zoom_in_steps = zoom_in["steps"]
            if is_json_key_present(zoom_in, "amount"):
                self.zoom_in_amount = zoom_in["amount"]
                self.zoom_in = True
        if is_json_key_present(settings_file, "init_image"):
            self.init_image = settings_file["init_image"]
        if is_json_key_present(settings_file, "resize_method"):
            self.resize_method = settings_file["resize_method"]
        if is_json_key_present(settings_file, "gobig_realesrgan"):
            print(
                '\nThe "gobig_realesrgan" setting is deprecated, use "resize_method" instead.\n'
            )
        if is_json_key_present(settings_file, "gobig"):
            self.gobig = settings_file["gobig"]
        if is_json_key_present(settings_file, "gobig_init"):
            self.gobig_init = settings_file["gobig_init"]
        if is_json_key_present(settings_file, "gobig_scale"):
            self.gobig_scale = settings_file["gobig_scale"]
        if is_json_key_present(settings_file, "gobig_prescaled"):
            self.gobig_prescaled = settings_file["gobig_prescaled"]
        if is_json_key_present(settings_file, "gobig_maximize"):
            self.gobig_maximize = settings_file["gobig_maximize"]
        if is_json_key_present(settings_file, "gobig_overlap"):
            self.gobig_overlap = settings_file["gobig_overlap"]
        if is_json_key_present(settings_file, "esrgan_model"):
            self.esrgan_model = settings_file["esrgan_model"]
        if is_json_key_present(settings_file, "gobig_cgs"):
            self.gobig_cgs = settings_file["gobig_cgs"]
        if is_json_key_present(settings_file, "augment_prompt"):
            self.augment_prompt = settings_file["augment_prompt"]
        if is_json_key_present(settings_file, "gobig_keep_slices"):
            self.gobig_keep_slices = settings_file["gobig_keep_slices"]
        if is_json_key_present(settings_file, "cool_down"):
            self.cool_down = settings_file["cool_down"]
        if is_json_key_present(settings_file, "checkpoint"):
            self.checkpoint = settings_file["checkpoint"]
        if is_json_key_present(settings_file, "use_jpg"):
            self.use_jpg = settings_file["use_jpg"]
        if is_json_key_present(settings_file, "hide_metadata"):
            self.hide_metadata = settings_file["hide_metadata"]
        if is_json_key_present(settings_file, "method"):
            self.method = settings_file["method"]
        if is_json_key_present(settings_file, "save_settings"):
            self.save_settings = settings_file["save_settings"]
        if is_json_key_present(settings_file, "improve_composition"):
            self.improve_composition = settings_file["improve_composition"]


def save_settings(prompt, filenum, options):
    setting_list = {
        "prompt": prompt,
        "checkpoint": options.checkpoint,
        "batch_name": options.batch_name,
        "steps": options.ddim_steps,
        "eta": options.ddim_eta,
        "n_iter": options.n_iter,
        "width": options.W,
        "height": options.H,
        "scale": options.scale,
        "dyn": options.dyn,
        "seed": options.seed,
        "variance": options.variance,
        "init_image": options.init_image,
        "zoom_in": options.zoom_in,
        "zoom_in_amount": options.zoom_in_amount,
        "zoom_in_depth": options.zoom_in_depth,
        "zoom_in_y": options.zoom_in_y,
        "zoom_in_x": options.zoom_in_x,
        "zoom_in_y": options.zoom_in_y,
        "zoom_in_steps": options.zoom_in_steps,
        "init_strength": 1.0 - options.strength,
        "resize_method": options.resize_method,
        "gobig": options.gobig,
        "gobig_init": options.gobig_init,
        "gobig_scale": options.gobig_scale,
        "gobig_prescaled": options.gobig_prescaled,
        "gobig_maximize": options.gobig_maximize,
        "gobig_overlap": options.gobig_overlap,
        "gobig_keep_slices": options.gobig_keep_slices,
        "esrgan_model": options.esrgan_model,
        "gobig_cgs": options.gobig_cgs,
        "augment_prompt": options.augment_prompt,
        "use_jpg": "true" if options.filetype == ".jpg" else "false",
        "hide_metadata": options.hide_metadata,
        "method": options.method,
        "improve_composition": options.improve_composition,
    }
    with open(
        f"{options.outdir}/{options.batch_name}-{filenum:04}.json",
        "w+",
        encoding="utf-8",
    ) as f:
        dump(setting_list, f, ensure_ascii=False, indent=4)

