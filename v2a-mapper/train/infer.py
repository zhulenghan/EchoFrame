import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
from tqdm import tqdm, trange
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import numpy as np
import librosa
import torch
import laion_clap
# from extraction.vgg_sound import *
# 在aws notebook环境需要加上这个
import sys
notebook_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(notebook_dir, "..")))

from audioldm import image_to_audio, build_model, clap_to_audio
import torch
import torchaudio
from transformers import AutoProcessor, AutoModel
import soundfile as sf 
from tqdm import trange




from models import *
from vgg_dataset import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "audioldm-s-full-v2"
audioldm=build_model(model_name=MODEL_NAME)


def image2audio(file_name, images, duration=10, guidance_scale=2.5, random_seed=42, n_candidates=3):
    waveform = clap_to_audio(
        latent_diffusion=audioldm,
        clap_feat=images,
        seed=random_seed,
        duration=duration,
        guidance_scale=guidance_scale,
        n_candidate_gen_per_text=int(n_candidates),
    )  # [bs, 1, samples]

    for i, wave in enumerate(waveform):
        # filename = f"{file_name}.wav"
        filename = file_name
        sf.write(filename, wave[0], 16000, 'PCM_16') 

    return waveform


def generate_audio(model,dataset, num_gen = 10,
                    video_path = "/mnt/new_volume/vgg_sound/scratch/shared/beegfs/hchen/train_data/VGGSound_final/video/",
                    output_path = "/home/ubuntu/project/v2a-mapper/train/test_inf/test_vgg/gen_outputs/",
                    output_ori_path = "/home/ubuntu/project/v2a-mapper/train/test_inf/test_vgg/ground_truth/"):
    
    test_loader = DataLoader(
    dataset, 
    batch_size=1, 
    shuffle=False, 
    num_workers=2,
    pin_memory=True
)   
    video_names = set([x[:-4] + '.mp4' for x in os.listdir(output_path)])
    print(video_names)
    
    
    cnt = 0
    for clip_feat, clap_feat, id, caption in tqdm(test_loader):
        if id[0] in video_names:
            continue
        model.eval()
        clip_feat = clip_feat.to(device)  # (64, 512)
        clap_feat = clap_feat.squeeze(0).to(device)  # (512)

        gen_clap_feat = model(clip_feat).to(device=device)  # (1, 512)

        image2audio(
            file_name= output_path + id[0][:-4] + ".wav",
            images=gen_clap_feat.to(device=device),
            duration=10,
            guidance_scale=3,
            random_seed=42,
            n_candidates=5
        )
        # !ffmpeg -i {video_path + id[0]} -vn -acodec pcm_s16le -ar 44100 -ac 2 /home/ubuntu/project/v2a-mapper/train/test_inf/test_vgg/test_outputs/{id[0][:-4]}.wav  
        import subprocess

        video_path_new = video_path + id[0]
        output_dir = output_ori_path
        conda_env = "mp"

        cmd = f"conda run -n {conda_env} python /home/ubuntu/project/v2a-mapper/train/audio_utls.py --video {video_path_new} --output {output_dir}"
        subprocess.run(cmd, shell=True)


        cnt += 1
        if cnt >= num_gen:
            break

def dummy_generate_audio(model,dataset, num_gen = 10,
                    video_path = "/mnt/new_volume/vgg_sound/scratch/shared/beegfs/hchen/train_data/VGGSound_final/video/",
                    output_path = "/home/ubuntu/project/v2a-mapper/train/test_inf/test_vgg/gen_outputs/",
                    output_ori_path = "/home/ubuntu/project/v2a-mapper/train/test_inf/test_vgg/ori_videos/"):
    
    test_loader = DataLoader(
    dataset, 
    batch_size=1, 
    shuffle=False, 
    num_workers=2,
    pin_memory=True
)   
    video_names = set([x[:-4] + '.mp4' for x in os.listdir(output_path)])
    print(video_names)
    
    
    cnt = 0
    for clip_feat, clap_feat, id, caption in tqdm(test_loader):
        if id[0] in video_names:
            continue
        # model.eval()
        # clip_feat = clip_feat.to(device)  # (64, 512)
        # clap_feat = clap_feat.squeeze(0).to(device)  # (512)

        # gen_clap_feat = model(clip_feat).to(device=device)  # (1, 512)

        # image2audio(
        #     file_name= output_path + id[0][:-4] + ".wav",
        #     images=gen_clap_feat.to(device=device),
        #     duration=10,
        #     guidance_scale=3,
        #     random_seed=42,
        #     n_candidates=5
        # )
        # !ffmpeg -i {video_path + id[0]} -vn -acodec pcm_s16le -ar 44100 -ac 2 /home/ubuntu/project/v2a-mapper/train/test_inf/test_vgg/test_outputs/{id[0][:-4]}.wav  
        import subprocess

        video_path_new = video_path + id[0]
        output_dir = output_ori_path
        conda_env = "mp"

        cmd = f"cp {video_path_new} {output_dir}"

        #cmd = f"conda run -n {conda_env} python /home/ubuntu/project/v2a-mapper/train/audio_utls.py --video {video_path_new} --output {output_dir}"
        subprocess.run(cmd, shell=True)


        cnt += 1
        if cnt >= num_gen:
            break