from prs.image import get_resampling_mode

def zoom_in_WIP(opt, model, device, output_image, metadata):
    print("zooming in!")
    new_opt = copy.deepcopy(opt)
    next_run_image = "next-run.png"

    img = apply_zoom_at(
        img=output_image,
        zoom=new_opt.zoom_in_depth,
        x=new_opt.zoom_in_x,
        y=new_opt.zoom_in_y,
    )

    print("\nZooms left: %s" % opt.zoom_in)

    # create a new unique value
    bump_grid_count = thats_numberwang(opt.outpath, opt.batch_name)
    output_filename = os.path.join(
        outpath,
        f"{new_opt.batch_name}{new_opt.device_id}-{bump_grid_count:04}{new_opt.filetype}",
    )
    img.save(
        next_run_image,
        pnginfo=metadata,
        quality=97,
    )

    # if new_opt.save_settings:
    #     used = copy.deepcopy(new_opt)
    #     used.scale = scale
    #     used.ddim_eta = ddim_eta
    #     save_settings(used, prompts[0], bump_grid_count)

    # Prepare for next run
    # TODO: feeding back in the image causes degraded images to be made
    new_opt.init_image = f"./{next_run_image}"

    # Decrement our zoom_in value
    new_opt.zoom_in_amount = new_opt.zoom_in_amount - 1

    # Reset batching and iterations
    new_opt.n_batches = 1
    new_opt.n_iter = 1

    if new_opt.zoom_in_steps:
        new_opt.ddim_steps = new_opt.zoom_in_steps

    # Apply zoom strength to new init strength
    if new_opt.zoom_in_strength:
        new_opt.strength = 1 - opt.zoom_in_strength

    img.close()
    output_image.close()
    print("\nRunning again with adjustments using previously created image")
    print(new_opt)
    do_run(device, model, new_opt)


def apply_zoom_at(img, zoom, x, y):
    w, h = img.size
    zoom2 = zoom * 2
    img = img.crop((x - w / zoom2, y - h / zoom2, x + w / zoom2, y + h / zoom2))
    return img.resize((w, h), get_resampling_mode())
