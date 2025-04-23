# %%
import numpy as np
import librosa
import torch
import laion_clap


# %%
import torch
import torchaudio
from transformers import AutoProcessor, AutoModel

# # Load processor and model
# processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")
# model = AutoModel.from_pretrained("laion/clap-htsat-unfused")

# # %%
from extraction.vgg_sound import *

# # %%
vgg_dataset = VGGSound(root="/home/ubuntu/project/subdata/video",
                           csv_path="/home/ubuntu/project/subdata/train_subset.csv")

data = vgg_dataset[0]

# for data in vgg_dataset:
#     # print(data['id'])
#     # print(data['audio'].shape)
#     # print(data['video'].shape)
#     # print(data['video'].shape)
#     # if data['id'] == '1msyXyqRvpY_000000.mp4':
#     #     print(data['audio'].shape)
#     break

# # %%
# data['clip_video']

# # %%
# data['audio']

# # %%

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = model.to(device)
# model.eval()

# # # Load audio file
# # audio_path = "input_audio.wav"
# # waveform, sr = torchaudio.load(audio_path)
# waveform = data['audio']

# # Resample to 48kHz (required by CLAP)
# # if sr != 48000:
# # resampler = torchaudio.transforms.Resample(16000, 48000)
# # waveform = resampler(waveform)

# # Convert to mono
# # if waveform.shape[0] > 1:
# #     waveform = torch.mean(waveform, dim=0, keepdim=True)

# # Prepare inputs for the model
# inputs = processor(audios=waveform, sampling_rate=48000, return_tensors="pt")
# inputs = {k: v.to(device) for k, v in inputs.items()}

# # Extract audio embeddings
# with torch.no_grad():
#     outputs = model.get_audio_features(**inputs)  # shape: [batch, feature_dim]

# audio_embed = outputs  # shape: [1, 512]


# # %%
# audio_embed.shape

# # %%

# # # quantization
# # def int16_to_float32(x):
# #     return (x / 32767.0).astype(np.float32)


# # def float32_to_int16(x):
# #     x = np.clip(x, a_min=-1., a_max=1.)
# #     return (x * 32767.).astype(np.int16)

# # model = laion_clap.CLAP_Module(enable_fusion=False)
# # model.load_ckpt() # download the default pretrained checkpoint.


# # %%
# data['clip_video']

# # %%


# # %%
# img = data['clip_video'][0, :, :, :]

# # %%
# data['clip_video']

# # %%
# img.shape

# # %%
# # Directly get audio embeddings from audio files
# audio_file = [
#     '/home/data/test_clap_short.wav',
#     '/home/data/test_clap_long.wav'
# ]
# audio_embed = model.get_audio_embedding_from_filelist(x = audio_file, use_tensor=False)
# print(audio_embed[:,-20:])
# print(audio_embed.shape)

# # Get audio embeddings from audio data
# audio_data, _ = librosa.load('/home/data/test_clap_short.wav', sr=48000) # sample rate should be 48000
# audio_data = audio_data.reshape(1, -1) # Make it (1,T) or (N,T)
# audio_embed = model.get_audio_embedding_from_data(x = audio_data, use_tensor=False)
# print(audio_embed[:,-20:])
# print(audio_embed.shape)

# # Directly get audio embeddings from audio files, but return torch tensor
# audio_file = [
#     '/home/data/test_clap_short.wav',
#     '/home/data/test_clap_long.wav'
# ]
# audio_embed = model.get_audio_embedding_from_filelist(x = audio_file, use_tensor=True)
# print(audio_embed[:,-20:])
# print(audio_embed.shape)

# # Get audio embeddings from audio data
# audio_data, _ = librosa.load('/home/data/test_clap_short.wav', sr=48000) # sample rate should be 48000
# audio_data = audio_data.reshape(1, -1) # Make it (1,T) or (N,T)
# audio_data = torch.from_numpy(int16_to_float32(float32_to_int16(audio_data))).float() # quantize before send it in to the model
# audio_embed = model.get_audio_embedding_from_data(x = audio_data, use_tensor=True)
# print(audio_embed[:,-20:])
# print(audio_embed.shape)

# # Get text embedings from texts:
# text_data = ["I love the contrastive learning", "I love the pretrain model"] 
# text_embed = model.get_text_embedding(text_data)
# print(text_embed)
# print(text_embed.shape)

# # Get text embedings from texts, but return torch tensor:
# text_data = ["I love the contrastive learning", "I love the pretrain model"] 
# text_embed = model.get_text_embedding(text_data, use_tensor=True)
# print(text_embed)
# print(text_embed.shape)


# %%
from audioldm import image_to_audio, build_model

# %%
MODEL_NAME = "audioldm-s-full-v2"
audioldm=build_model(model_name=MODEL_NAME)

# %%
import soundfile as sf 

# %%
def image2audio(file_name, images, duration=10, guidance_scale=2.5, random_seed=42, n_candidates=3):
    waveform = image_to_audio(
        latent_diffusion=audioldm,
        images=images,
        seed=random_seed,
        duration=duration,
        guidance_scale=guidance_scale,
        n_candidate_gen_per_text=int(n_candidates),
    )  # [bs, 1, samples]

    for i, wave in enumerate(waveform):
        filename = f"../output/{file_name}.wav"
        sf.write(filename, wave[0], 16000, 'PCM_16') 

    return waveform

# %%
image2audio(
    file_name="test",
    images=data['clip_video'],
    duration=10,
    guidance_scale=2.5,
    random_seed=42,
    n_candidates=3
)

# %%
audioldm.first_stage_model


