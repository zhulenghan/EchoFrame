import os
from huggingface_hub import snapshot_download
import pandas as pd
from datasets import Dataset

# 配置
repo_id = "lenghanz/VGGSound"
save_dir = "/mnt/new_volume/vgg_sound"
data_dir = "/mnt/new_volume/vgg_sound"

# print(f"Step 1: Downloading dataset to {save_dir}")
# snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=save_dir, local_dir_use_symlinks=False)

print(f"Step 2: Looking for CSV files in {data_dir}")
csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

if not csv_files:
    raise RuntimeError("没有找到任何 CSV 文件，请检查下载路径是否正确。")

all_data = []
for file in csv_files:
    file_path = os.path.join(data_dir, file)
    df = pd.read_csv(file_path, header=None, names=["video", "label"])
    all_data.append(df)

print(f"Step 3: Merging {len(all_data)} CSV files")
full_df = pd.concat(all_data, ignore_index=True)

print("Step 4: Converting to HuggingFace Dataset")
dataset = Dataset.from_pandas(full_df)

# 可选：保存为 arrow 文件方便后续快速加载
arrow_path = os.path.join(save_dir, "vggsound_dataset.arrow")
dataset.save_to_disk(arrow_path)

print(f"✅ Done! Dataset saved to: {arrow_path}")
print(f"📦 Total samples: {len(dataset)}")
print(f"🔍 First sample: {dataset[0]}")
