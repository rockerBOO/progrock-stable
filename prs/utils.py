import os 
import torch
import numpy as np

# Simple check to see if a key is present in the settings file
def is_json_key_present(json, key, subkey="none"):
    try:
        if subkey != "none":
            buf = json[key][subkey]
        else:
            buf = json[key]
    except KeyError:
        return False
    if type(buf) == type(None):
        return False
    return True


def thats_numberwang(dir, wildcard):
    # get the highest numbered file in the out directory, and add 1. So simple.
    files = os.listdir(dir)
    filenums = []
    filenum = 0
    for file in files:
        if wildcard in file:
            start = file.rfind("-")
            end = file.rfind(".")
            try:
                filenum = file[start + 1 : end]
                filenum = int(filenum)
            except:
                print(f'Improperly named file "{file}" in output directory')
                print(f"Please make sure output filenames use the name-1234.png format")
                quit()
            filenums.append(filenum)
    if not filenums:
        numberwang = 0
    else:
        numberwang = max(filenums) + 1
    return numberwang

def slerp(device, t, v0:torch.Tensor, v1:torch.Tensor, DOT_THRESHOLD=0.9995):
    v0 = v0.detach().cpu().numpy()
    v1 = v1.detach().cpu().numpy()
    
    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    v2 = torch.from_numpy(v2).to(device)

    return v2
