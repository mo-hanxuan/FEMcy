from genericpath import exists
from moviepy.editor import ImageSequenceClip
import os; os.system("")

path = input("\033[35;1m please give the folder path where your pictures stored at. path = \033[0m")
path += "/"

img_name = input("\033[35;1m img name (no prefix) = \033[0m")
img_names = []
for path_, dir_list, file_list in os.walk(path):
    for file in file_list:
        if img_name == file[:len(img_name)] and file[-4:] == ".png":
            img_names.append(path + file)
### sort by time, e.g., if file names are time_XX.png, we sort them by XX
img_names = sorted(img_names, key=lambda x: x.split("(")[-1].split(")")[0].split("_"))

fps = 6  # frame per second
print("len(img_names) =", len(img_names), ", fps =", fps)

clip = ImageSequenceClip(img_names, fps=fps, durations=8.*len(img_names)/fps)

gif_name = input("\033[32;1m please give a name for the gif file (no prefix): \033[0m")
clip.write_gif(path + gif_name + ".gif")