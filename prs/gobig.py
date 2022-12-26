
def do_gobig(gobig_init, device, model, opt):
    overlap = opt.gobig_overlap
    original_prompt = opt.prompt
    outpath = opt.outdir
    # get our render size for each slice, and our target size
    input_image = Image.open(gobig_init).convert("RGBA")
    if opt.gobig_prescaled == False:
        opt.W, opt.H = input_image.size
        target_W = opt.W * opt.gobig_scale
        target_H = opt.H * opt.gobig_scale
        if opt.resize_method == "realesrgan":
            input_image = esrgan_resize(input_image, opt.device_id, opt.esrgan_model)
        target_image = input_image.resize(
            (target_W, target_H), get_resampling_mode()
        )  # esrgan resizes 4x by default, so this brings us in line with our actual scale target
    else:
        # target_W, target_H = input_image.size
        target_image = input_image
    # slice up the image into a grid
    slices, target_image = grid_slice(
        target_image, overlap, (opt.W, opt.H), opt.gobig_maximize
    )
    # now we trigger a do_run for each slice
    betterslices = []
    slice_image = f"slice{opt.device_id}.png"
    for count, chunk_w_coords in enumerate(slices):
        chunk, coord_x, coord_y = chunk_w_coords
        chunk.save(slice_image)
        chunk.close()
        opt.init_image = slice_image
        opt.save_settings = (
            False  # we don't need to keep settings for each slice, just the main image.
        )
        opt.n_iter = 1  # no point doing multiple iterations since only one will be used
        opt.improve_composition = (
            False  # don't want to do stretching and yet another init image during gobig
        )
        opt.seed = opt.seed + 1
        opt.scale = opt.gobig_cgs if opt.gobig_cgs != None else opt.scale
        if opt.augment_prompt != None:
            # now augment the prompt
            opt.prompt = opt.augment_prompt + " " + original_prompt
        result = do_run(device, model, opt)
        resultslice = Image.open(result).convert("RGBA")
        betterslices.append((resultslice.copy(), coord_x, coord_y))
        resultslice.close()
        if opt.gobig_keep_slices == False:
            os.remove(result)
    # create an alpha channel for compositing the slices
    alpha = Image.new("L", (opt.W, opt.H), color=0xFF)
    alpha_gradient = ImageDraw.Draw(alpha)
    a = 0
    i = 0
    a_overlap = int(
        overlap / 2
    )  # we want the alpha gradient to be half the size of the overlap, otherwise we always see some of the original background underneath
    shape = ((opt.W, opt.H), (0, 0))
    while i < overlap:
        alpha_gradient.rectangle(shape, fill=a)
        a += int(255 / a_overlap)
        a = 255 if a > 255 else a
        i += 1
        shape = ((opt.W - i, opt.H - i), (i, i))
    alpha_gradient.rectangle(
        shape, fill=255
    )  # one last one to make sure the non-overlap section is fully used.
    mask = Image.new("RGBA", (opt.W, opt.H), color=0)
    mask.putalpha(alpha)
    # now composite the slices together
    finished_slices = []
    for betterslice, x, y in betterslices:
        finished_slice = addalpha(betterslice, mask)
        finished_slices.append((finished_slice, x, y))
    final_output = grid_merge(target_image, finished_slices)
    # name the file in a way that hopefully doesn't break things
    print(f"result is {result}")
    result = result.replace(".png", "")
    result_split = result.rsplit("-", 1)
    result_split[0] = result_split[0] + "_gobig-"
    result = result_split[0] + result_split[1]
    print(f"Gobig output saved as {result}{opt.filetype}")
    final_output.save(f"{result}{opt.filetype}", quality=opt.quality)
    final_output.close()
    input_image.close()


def addalpha(im, mask):
    imr, img, imb, ima = im.split()
    mmr, mmg, mmb, mma = mask.split()
    im = Image.merge(
        "RGBA", [imr, img, imb, mma]
    )  # we want the RGB from the original, but the transparency from the mask
    return im


# Alternative method composites a grid of images at the positions provided
def grid_merge(source, slices):
    source.convert("RGBA")
    for slice, posx, posy in slices:  # go in reverse to get proper stacking
        source.alpha_composite(slice, (posx, posy))
    return source


def grid_coords(target, original, overlap, maxed):
    # generate a list of coordinate tuples for our sections, in order of how they'll be rendered
    # target should be the size for the gobig result, original is the size of each chunk being rendered
    target_x, target_y = target
    original_x, original_y = original
    do_calc = True
    while do_calc:
        print(f"Target size is {target_x} x {target_y}")
        center = []
        center_x = int(target_x / 2)
        center_y = int(target_y / 2)
        x = center_x - int(original_x / 2)
        y = center_y - int(original_y / 2)
        center.append((x, y))  # center chunk
        uy = y  # up
        uy_list = []
        dy = y  # down
        dy_list = []
        lx = x  # left
        lx_list = []
        rx = x  # right
        rx_list = []
        while uy > 0:  # center row vertical up
            uy = uy - original_y + overlap
            uy_list.append((lx, uy))
        while (dy + original_y) <= target_y:  # center row vertical down
            dy = dy + original_y - overlap
            dy_list.append((rx, dy))
        while lx > 0:
            lx = lx - original_x + overlap
            lx_list.append((lx, y))
            uy = y
            while uy > 0:
                uy = uy - original_y + overlap
                uy_list.append((lx, uy))
            dy = y
            while (dy + original_y) <= target_y:
                dy = dy + original_y - overlap
                dy_list.append((lx, dy))
        while (rx + original_x) <= target_x:
            rx = rx + original_x - overlap
            rx_list.append((rx, y))
            uy = y
            while uy > 0:
                uy = uy - original_y + overlap
                uy_list.append((rx, uy))
            dy = y
            while (dy + original_y) <= target_y:
                dy = dy + original_y - overlap
                dy_list.append((rx, dy))
        if maxed:
            # calculate a new size that will fill the canvas, which will be optionally used in grid_slice and go_big
            last_coordx, last_coordy = dy_list[-1:][0]
            render_edgey = (
                last_coordy + original_y
            )  # outer bottom edge of the render canvas
            render_edgex = (
                last_coordx + original_x
            )  # outer side edge of the render canvas
            render_edgex += (
                render_edgex - target_x
            )  # we have to extend the "negative" side as well, so we do it twice
            render_edgey += render_edgey - target_y
            scalarx = render_edgex / target_x
            scalary = render_edgey / target_y
            if scalarx <= scalary:
                target_x = int(target_x * scalarx)
                target_y = int(target_y * scalarx)
            else:
                target_x = int(target_x * scalary)
                target_y = int(target_y * scalary)
            maxed = False
        else:
            do_calc = False
    # now put all the chunks into one master list of coordinates (essentially reverse of how we calculated them so that the central slices will be on top)
    result = []
    for coords in dy_list[::-1]:
        result.append(coords)
    for coords in uy_list[::-1]:
        result.append(coords)
    for coords in rx_list[::-1]:
        result.append(coords)
    for coords in lx_list[::-1]:
        result.append(coords)
    result.append(center[0])
    return result, (target_x, target_y)


# Chop our source into a grid of images that each equal the size of the original render
def grid_slice(source, overlap, og_size, maxed=False):
    width, height = og_size  # size of the slices to be rendered
    coordinates, new_size = grid_coords(source.size, og_size, overlap, maxed)
    if source.size != new_size:
        source = source.resize(new_size, get_resampling_mode())
    slices = []
    for coordinate in coordinates:
        x, y = coordinate
        slices.append(((source.crop((x, y, x + width, y + height))), x, y))
    global slices_todo
    slices_todo = len(slices) - 1
    return slices, source
