from torch.utils.data import Dataset, DataLoader, random_split
import os
import torch
from tqdm import tqdm
import numpy as np


class LargeVideoDataset(Dataset):
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



from functools import lru_cache

class LRUVideoDataset(Dataset):
    def __init__(self, data_dir, subset_ratio=0.2, cache_size=2000, transform=None):
        super().__init__()
        file_list = []
        for root, dirs, files in os.walk(data_dir):
            for fn in files:
                if fn.endswith(".pth"):
                    file_list.append(os.path.join(root, fn))
        file_list = sorted(file_list)
        num_samples = int(len(file_list) * subset_ratio)
        self.file_paths = file_list[:num_samples]
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    @lru_cache(maxsize=2000)
    def _load(self, idx):
        data = torch.load(self.file_paths[idx], map_location="cpu")
        return data["clip_features"], data["clap_features"]

    def __getitem__(self, idx):
        clip_feat, clap_feat = self._load(idx)
        if self.transform:
            clip_feat, clap_feat = self.transform((clip_feat, clap_feat))
        return clip_feat, clap_feat

class InMemoryVideoDataset(Dataset):
    def __init__(self, data_dir, subset_ratio=0.2, transform=None, dtype=torch.float16):
        """
        一次性把前 subset_ratio 比例的数据全部加载到内存里，
        并转成指定的 dtype（默认 float16）。
        """
        super().__init__()
        file_list = []
        for root, dirs, files in os.walk(data_dir):
            for fn in files:
                if fn.endswith(".pth"):
                    file_list.append(os.path.join(root, fn))
        file_list = sorted(file_list)
        num_samples = int(len(file_list) * subset_ratio)
        self.file_paths = file_list[:num_samples]
        self.transform = transform
        self.dtype = dtype

        # 一次性加载到内存并转 dtype
        self.data = []
        print(f"Loading {len(self.file_paths)} samples into memory as {self.dtype} …")
        for path in tqdm(self.file_paths, desc="Caching data"):
            sample = torch.load(path, map_location="cpu")
            clip = sample["clip_features"].to(self.dtype)   # (64,512) float16
            clap = sample["clap_features"].to(self.dtype)   # (1,512) float16
            self.data.append((clip, clap))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        clip_feat, clap_feat = self.data[idx]
        if self.transform:
            clip_feat, clap_feat = self.transform((clip_feat, clap_feat))
        return clip_feat, clap_feat

