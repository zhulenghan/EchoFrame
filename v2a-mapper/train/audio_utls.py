import argparse
import os
from moviepy import VideoFileClip, TextClip, CompositeVideoClip,AudioFileClip

def save_ori_audio(video_path, audio_output_dir):
    """
    Save the original audio from a video file.

    :param video_path: Path to the input video file.
    :param audio_output_dir: Directory to save the extracted audio file.
    """
    video_id = video_path.split('/')[-1].split('.')[0]
    video = VideoFileClip(video_path)
    audio = video.audio

    os.makedirs(audio_output_dir, exist_ok=True)
    audio_path = os.path.join(audio_output_dir, f"{video_id}.wav")
    audio.write_audiofile(audio_path)

    audio.close()
    video.close()

def main():
    parser = argparse.ArgumentParser(description="Extract audio from a video file.")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--output", type=str, required=True, help="Directory to save the extracted audio.")
    
    args = parser.parse_args()
    save_ori_audio(args.video, args.output)

if __name__ == "__main__":
    main()
