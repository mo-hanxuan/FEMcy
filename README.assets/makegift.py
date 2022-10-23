from moviepy.editor import ImageSequenceClip
import functools
import os; os.system("")

def compare(name0: str, name1: str):
    name0, name1 = name0.split("_"), name1.split("_")
    time0, newtonLoop0, relaxLoop0 = name0[-4], name0[-3], name0[-2]
    time1, newtonLoop1, relaxLoop1 = name1[-4], name1[-3], name1[-2]
    if time0 < time1:
        return -1
    elif time0 > time1:
        return 1
    else:
        if newtonLoop0 < newtonLoop1:
            return -1
        elif newtonLoop0 > newtonLoop1:
            return 1
        else:
            if relaxLoop0 < relaxLoop1:
                return -1
            elif relaxLoop0 > relaxLoop1:
                return 1
            else:
                return 0


path = input("\033[35;1m please give the folder path where your pictures stored at. path = \033[0m")
path += "/"

img_name = input("\033[35;1m img name (no prefix) = \033[0m")
img_names = []
for path_, dir_list, file_list in os.walk(path):
    for file in file_list:
        if img_name == file[:len(img_name)] and file[-4:] == ".png":
            img_names.append(path + file)

showNewtonStep = True
if not showNewtonStep:
    i = 0
    while i < len(img_names):
        if "time" not in img_names[i]:
            del img_names[i]
        else:
            i += 1
    ### sort by time, e.g., if file names are timeXX.png, we sort them by XX
    img_names = sorted(img_names, key=lambda x: x.split("time")[-1].split("_")[0])

    fps = 6  # frame per second
    print("len(img_names) =", len(img_names), ", fps =", fps)

    clip = ImageSequenceClip(img_names, fps=fps, durations=8.*len(img_names)/fps)

    gif_name = input("\033[32;1m please give a name for the gif file (no prefix): \033[0m")
    clip.write_gif(path + gif_name + ".gif")
else:
    i = 0
    while i < len(img_names):
        if "time" in img_names[i]:
            del img_names[i]
        else:
            i += 1
    ### sort by time, e.g., if file names are timeXX.png, we sort them by XX
    img_names = sorted(img_names, key=functools.cmp_to_key(compare))

    fps = 30  # frame per second
    print("len(img_names) =", len(img_names), ", fps =", fps)

    clip = ImageSequenceClip(img_names, fps=fps, durations=8.*len(img_names)/fps)

    gif_name = input("\033[32;1m please give a name for the gif file (no prefix): \033[0m")
    clip.write_gif(path + gif_name + "_newtonSteps.gif")