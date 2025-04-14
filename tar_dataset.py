
# from huggingface_hub import create_repo

# create_repo("vgg_sound_emb", repo_type="dataset")

# from huggingface_hub import upload_folder

# from huggingface_hub import HfApi

# api = HfApi()

# api.upload_large_folder(
#     folder_path="/mnt/new_volume2/vgg_sound_emb",
#     repo_id="zianpan01/vgg_sound_emb",  # Replace with your actual repo path
#     repo_type="dataset",
#      verbose=True 
# )


# upload_folder(
#     repo_id="zianpan01/vgg_sound_emb",  # replace with your actual username/repo
#     folder_path="/mnt/new_volume2/vgg_sound_emb",  # the full path to the folder
#     repo_type="dataset",
#      verbose=True 
# )

from huggingface_hub import HfApi, upload_file
from pathlib import Path
from tqdm import tqdm
import os

api = HfApi()
repo_id = "zianpan01/vgg_sound_emb"

def upload_with_progress(folder_path, path_in_repo=""):
    files = list(Path(folder_path).rglob("*.pth"))
    total = len(files)

    for file in tqdm(files, desc=f"Uploading {path_in_repo}", unit="file"):
        relative_path = file.relative_to(folder_path)
        repo_path = os.path.join(path_in_repo, str(relative_path))
        
        upload_file(
            path_or_fileobj=file,
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type="dataset"
        )

# Example for uploading train and test folders
upload_with_progress("/mnt/new_volume2/vgg_sound_emb/train", "train")
upload_with_progress("/mnt/new_volume2/vgg_sound_emb/test", "test")
