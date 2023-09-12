import os
import json
import einops
import imageio
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.utils import make_grid, save_image

def get_test_label(args, test_file):
    label_dict = json.load(open(os.path.join(args.dataset_dir, "objects.json")))
    labels = json.load(open(os.path.join(args.dataset_dir, test_file + ".json")))

    newLabels = []
    for i in range(len(labels)):
        onehot_label = torch.zeros(24, dtype=torch.float32)
        for j in range(len(labels[i])):
            onehot_label[label_dict[labels[i][j]]] = 1 
        newLabels.append(onehot_label)

    return newLabels

def evaluate(model, scheduler, epoch, args, device, test_file):
    test_label = torch.stack(get_test_label(args, test_file)).to(device)
    num_samples = len(test_label)

    x = torch.randn(num_samples, 3, args.sample_size, args.sample_size).to(device)
    prog_list = [x[1]]
    for i, t in enumerate(scheduler.timesteps):
        with torch.no_grad():
            noise_residual = model(x, t, test_label).sample

        x = scheduler.step(noise_residual, t, x).prev_sample
        if i % 111 == 0:
            prog_list.append(x[1])  
    prog_img = torch.stack(prog_list).to(device)
    prog_img = (prog_img / 2 + 0.5).clamp(0, 1)
    image = (x / 2 + 0.5).clamp(0, 1)


    save_image(make_grid(image, nrow=8), "{}/{}_{}.png".format(args.figure_dir, test_file, epoch))
    save_image(make_grid(prog_img), "{}/{}_{}_{}.png".format(args.figure_dir, test_file, epoch, "progressive"))
    return x, test_label

