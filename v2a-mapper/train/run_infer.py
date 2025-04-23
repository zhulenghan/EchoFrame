# %%
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





# %%

from models import *
from vgg_dataset import *
from tqdm import trange
from infer import *

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
vgg_video_path = "/mnt/new_volume/vgg_sound/scratch/shared/beegfs/hchen/train_data/VGGSound_final/video/"
gen_output_path = "/home/ubuntu/project/v2a-mapper/train/test_inf/test_vgg/gen_outputs/"
output_ori_path = "/home/ubuntu/project/v2a-mapper/train/test_inf/test_vgg/ground_truth/"
zip_output_path = "/home/ubuntu/project/v2a-mapper/train/test_inf/test_vgg/output_audio.zip"



# %%
# model = V2AMapperMLPImproved(input_dim=512, hidden_dim=512, output_dim=512).to(device)
model = V2AMapperMLPImproved(input_dim=512, hidden_dim=2048, output_dim=512).to(device)
model = V2AMapperMLPImprovednoresidule(input_dim=512, hidden_dim=2048, output_dim=512).to(device)
# model = V2AMapperBiLSTM(input_dim=512, hidden_dim=1024, output_dim=512, lstm_hidden=256, lstm_layers=1,dropout=0.3).to(device)

# %%
# ckpt_path = "/home/ubuntu/best_model.pth"

ckpt_path = "/home/ubuntu/project/v2a-mapper/train/ckpts-lstm/best_model.pth"
ckpt_path = "/home/ubuntu/project/v2a-mapper/train/ckpts_mlp_res/best_model_mlp_res.pth"
ckpt_path = "/home/ubuntu/project/v2a-mapper/train/ckpts_mlp_wr/best_model_mlp_wores.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# model = V2AMapperMLP()
model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
model = model.to(device)

# %%

class LargeVideoTestDataset(Dataset):
    def __init__(self, data_dir, subset_ratio = 0.2, transform=None):
        """
        root_dir: 保存所有 .pth 文件的目录，每个文件对应一个 sample。
        transform: 如果需要对数据做预处理，可在这里传入。
        """
        super().__init__()
        # 仅收集当前目录下所有的 pth 文件列表
        file_list = []

        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".pth"):
                    file_list.append(os.path.join(root, file))

        # 仅使用前 20% 的数据
        num_samples = int(len(file_list) * subset_ratio)

        self.file_paths = sorted(file_list)[:num_samples]
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # 在这里按需读取，而不是一次性加载全部
        pth_path = self.file_paths[idx]
        sample_data = torch.load(pth_path)  
        clip_feat = sample_data['clip_features']  # (64, 512)
        clap_feat = sample_data['clap_features']  # (1, 512)
        id = sample_data['id']
        caption = sample_data['caption']

        if self.transform:
            clip_feat, clap_feat = self.transform((clip_feat, clap_feat))

        return clip_feat, clap_feat, id, caption

# %%
root = "/mnt/new_volume2/vgg_sound_emb"
partition = "test"
data_dir = f"{root}/{partition}"
# data_dir ="/mnt/new_volume2/ai_video_emb/"
vgg_sound = LargeVideoTestDataset(data_dir, subset_ratio = 1)

# %%
generate_audio(model,vgg_sound, num_gen = 100,
                    video_path = vgg_video_path,
                    output_path = gen_output_path,
                    output_ori_path = output_ori_path)

# %%
# dummy_generate_audio(model,vgg_sound, num_gen = 100,
#                     video_path = vgg_video_path,
#                     output_path = '/',
#                     output_ori_path = '/home/ubuntu/project/v2a-mapper/train/test_inf/test_vgg/ori_videos/')

# %%
gen_output_path = "/home/ubuntu/project/v2a-mapper/train/test_inf/test_vgg/gen_outputs"
output_ori_path = "/home/ubuntu/project/v2a-mapper/train/test_inf/test_vgg/ground_truth"
zip_output_path = "/home/ubuntu/project/v2a-mapper/train/test_inf/test_vgg/output_audio.zip"


# %%
import zipfile
import os

def zip_folders_with_root(folders, zip_filename):
    """
    Compress multiple folders into a single zip file, preserving top-level folder names.
    
    :param folders: List of folder paths to include.
    :param zip_filename: The output zip file path.
    """
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for folder in folders:
            folder_base  = os.path.basename(folder.rstrip("/"))
            for root, dirs, files in os.walk(folder):
                for file in files:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, start=os.path.dirname(folder))
                    zipf.write(full_path, arcname=rel_path)

zip_folders_with_root([gen_output_path[:-1], output_ori_path[:-1]], zip_output_path)

print(f"✅ 成功压缩两个文件夹为 {zip_output_path}")


# %%


# %%
# cnt = 0
# for clip_feat, clap_feat, id, caption in tqdm(test_loader):
#     if cnt <= 21:
#         cnt += 1
#         continue
#     # clip_feat: (1, 64, 512)
#     # clap_feat: (1, 512)
#     # id: (1,)
#     # caption: (1,)
#     print(clip_feat.shape, clap_feat.shape, id, caption)
#     clip_feat = clip_feat.to(device)  # (64, 512)
#     clap_feat = clap_feat.squeeze(0).to(device)  # (512)

#     gen_clap_feat = model(clip_feat).to(device=device)  # (1, 512)
#     # calculate cosine similarity
#     cos = nn.CosineSimilarity(dim=0)  # compare whole vectors
#     cos_sim = cos(gen_clap_feat.squeeze(0), clap_feat.squeeze(0).squeeze(0))  # both are (512,)

#     print("cosine similarity:", cos_sim)
#     # print(cos_sim.shape)
#     # print(cos_sim)
#     image2audio(
#         file_name="/home/ubuntu/project/v2a-mapper/train/test_inf/test_ag/test_outputs/" + id[0][:-4] + "_gen.wav",
#         images=gen_clap_feat.to(device=device),
#         duration=10,
#         guidance_scale=2.5,
#         random_seed=42,
#         n_candidates=3
#     )
#     # !ffmpeg -i {video_path + id[0]} -vn -acodec pcm_s16le -ar 44100 -ac 2 /home/ubuntu/project/v2a-mapper/train/test_inf/test_vgg/test_outputs/{id[0][:-4]}.wav

#     cnt += 1
#     if cnt == 70:
#         break

# %%
# cnt = 0
# for clip_feat, clap_feat, id, caption in tqdm(test_loader):
#     # clip_feat: (1, 64, 512)
#     # clap_feat: (1, 512)
#     # id: (1,)
#     # caption: (1,)
#     print(clip_feat.shape, clap_feat.shape, id, caption)
#     clip_feat = clip_feat.to(device)  # (64, 512)
#     clap_feat = clap_feat.squeeze(0).to(device)  # (512)

#     gen_clap_feat = model(clip_feat).to(device=device)  # (1, 512)
#     # calculate cosine similarity
#     cos = nn.CosineSimilarity(dim=0)  # compare whole vectors
#     cos_sim = cos(gen_clap_feat.squeeze(0), clap_feat.squeeze(0).squeeze(0))  # both are (512,)

#     print("cosine similarity:", cos_sim)
#     # print(cos_sim.shape)
#     # print(cos_sim)
#     image2audio(
#         file_name="/home/ubuntu/project/v2a-mapper/train/test_inf/test_vgg/test_outputs/" + id[0][:-4] + "_gen.wav",
#         images=gen_clap_feat.to(device=device),
#         duration=10,
#         guidance_scale=2.5,
#         random_seed=42,
#         n_candidates=3
#     )
#     !ffmpeg -i {video_path + id[0]} -vn -acodec pcm_s16le -ar 44100 -ac 2 /home/ubuntu/project/v2a-mapper/train/test_inf/test_vgg/test_outputs/{id[0][:-4]}.wav

#     cnt += 1
#     if cnt == 20:
#         break

# %%



