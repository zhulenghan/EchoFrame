import torch
from audioldm_eval import EvaluationHelper

import os

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
os.chdir(current_dir)

# GPU acceleration is preferred
device = torch.device(f"cuda:{0}")

path = "data/output_audio_mlp_res/"

# Initialize a helper instance
evaluator = EvaluationHelper(16000, device, backbone="mert")

# Perform evaluation, result will be print out and saved as json
metrics = evaluator.main(
    path + 'gen_outputs/',
    path + 'ground_truth/',
    limit_num=None # If you only intend to evaluate X (int) pairs of data, set limit_num=X
)