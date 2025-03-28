import logging
import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import torch
import torchaudio
from torch.utils.data.dataset import Dataset
from torchvision.transforms import v2
from torio.io import StreamingMediaDecoder

#from utils.dist_utils import local_rank
local_rank = 0

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger()

_CLIP_SIZE = 384
_CLIP_FPS = 8.0


class VGGSound(Dataset):

    def __init__(
        self,
        root: Union[str, Path],
        *,
        csv_path: Union[str, Path] = '/ocean/projects/cis250032p/lzhu6/dataset/VGGSound/test.csv',
        sample_rate: int = 16_000,
        duration_sec: float = 10.0,
        audio_samples: Optional[int] = None,
        normalize_audio: bool = False,
    ):
        self.root = Path(root)
        self.normalize_audio = normalize_audio
        if audio_samples is None:
            self.audio_samples = int(sample_rate * duration_sec)
        else:
            self.audio_samples = audio_samples
            effective_duration = audio_samples / sample_rate
            # make sure the duration is close enough, within 15ms
            assert abs(effective_duration - duration_sec) < 0.015, \
                f'audio_samples {audio_samples} does not match duration_sec {duration_sec}'

        videos = sorted(os.listdir(self.root))
        videos = set([Path(v) for v in videos])  # remove extensions
        videos_str = set([str(v) for v in videos])
        self.labels = {}
        self.videos = []
        missing_videos = []

        # read the tsv for subset information
        df_list = pd.read_csv(csv_path, header=None, names=['id', 'label'], dtype={'id': str}).to_dict('records')
        for record in df_list:
            id = record['id']
            label = record['label']
            if id in videos_str:
                self.labels[id] = label
                self.videos.append(id)
            else:
                missing_videos.append(id)

        if local_rank == 0:
            log.info(f'{len(videos)} videos found in {root}')
            log.info(f'{len(self.videos)} videos found in {csv_path}')
            log.info(f'{len(missing_videos)} videos missing in {root}')

        self.sample_rate = sample_rate
        self.duration_sec = duration_sec

        self.expected_audio_length = self.audio_samples
        self.clip_expected_length = int(_CLIP_FPS * self.duration_sec)

        self.clip_transform = v2.Compose([
            v2.Resize((_CLIP_SIZE, _CLIP_SIZE), interpolation=v2.InterpolationMode.BICUBIC),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])


        self.resampler = {}

    def sample(self, idx: int) -> dict[str, torch.Tensor]:
        video_id = self.videos[idx]
        label = self.labels[video_id]

        reader = StreamingMediaDecoder(self.root / (video_id))
        reader.add_basic_video_stream(
            frames_per_chunk=int(_CLIP_FPS * self.duration_sec),
            frame_rate=_CLIP_FPS,
            format='rgb24',
        )
        reader.add_basic_audio_stream(frames_per_chunk=2**30, )

        reader.fill_buffer()
        data_chunk = reader.pop_chunks()

        clip_chunk = data_chunk[0]
        audio_chunk = data_chunk[1]

        if clip_chunk is None:
            raise RuntimeError(f'CLIP video returned None {video_id}')
        if clip_chunk.shape[0] < self.clip_expected_length:
            raise RuntimeError(
                f'CLIP video too short {video_id}, expected {self.clip_expected_length}, got {clip_chunk.shape[0]}'
            )


        # process audio
        sample_rate = int(reader.get_out_stream_info(1).sample_rate)
        audio_chunk = audio_chunk.transpose(0, 1)
        audio_chunk = audio_chunk.mean(dim=0)  # mono
        if self.normalize_audio:
            abs_max = audio_chunk.abs().max()
            audio_chunk = audio_chunk / abs_max * 0.95
            if abs_max <= 1e-6:
                raise RuntimeError(f'Audio is silent {video_id}')

        # resample
        if sample_rate == self.sample_rate:
            audio_chunk = audio_chunk
        else:
            if sample_rate not in self.resampler:
                # https://pytorch.org/audio/stable/tutorials/audio_resampling_tutorial.html#kaiser-best
                self.resampler[sample_rate] = torchaudio.transforms.Resample(
                    sample_rate,
                    self.sample_rate,
                    lowpass_filter_width=64,
                    rolloff=0.9475937167399596,
                    resampling_method='sinc_interp_kaiser',
                    beta=14.769656459379492,
                )
            audio_chunk = self.resampler[sample_rate](audio_chunk)
        if audio_chunk.shape[0] < self.expected_audio_length:
            raise RuntimeError(f'Audio too short {video_id}')
        audio_chunk = audio_chunk[:self.expected_audio_length]

        # truncate the video
        clip_chunk = clip_chunk[:self.clip_expected_length]
        if clip_chunk.shape[0] != self.clip_expected_length:
            raise RuntimeError(f'CLIP video wrong length {video_id}, '
                               f'expected {self.clip_expected_length}, '
                               f'got {clip_chunk.shape[0]}')
        clip_chunk = self.clip_transform(clip_chunk)


        data = {
            'id': video_id,
            'caption': label,
            'audio': audio_chunk,
            'clip_video': clip_chunk,
        }

        return data

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        try:
            return self.sample(idx)
        except Exception as e:
            log.error(f'Error loading video {self.videos[idx]}: {e}')
            return None

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":
    vgg_dataset = VGGSound(root="/jet/home/lzhu6/cis250032p/lzhu6/dataset/VGGSound/train_data/VGGSound_final/video",
                           csv_path="/jet/home/lzhu6/cis250032p/lzhu6/dataset/VGGSound/test.csv")
    for data in vgg_dataset:
        print(data)
        break