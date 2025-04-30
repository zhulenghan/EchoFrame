# EchoFrame: Capturing the Sound of Moments - Video2Audio Generation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) **Authors:** Xuanrui Chen, Zian Pan, Yiming Fu, Lenghan Zhu (Carnegie Mellon University)

## Abstract

This project tackles the challenge of generating realistic, high-quality audio directly from silent video inputs, focusing on semantic relevance and audio fidelity. We employ a cascaded model approach named **EchoFrame**. This method utilizes pre-trained multimodal embeddings – specifically CLIP for visual features and CLAP for audio embeddings – to bridge the gap between visual perception and auditory synthesis. EchoFrame consists of a learned mapping network that translates sequences of visual embeddings into corresponding audio embeddings, which are then used by a pre-trained latent diffusion model (Audio-LDM) to generate the final audio waveform. Our approach demonstrates improved performance over existing methods on metrics like Fréchet Audio Distance (FAD) and Kullback-Leibler Divergence (MKL). As part of this project, we also provide a cleaned and organized version of the VGGSound dataset, along with pre-computed visual (CLIP) and audio (CLAP) embeddings, to facilitate future research.

## Motivation

Visual and auditory senses are intrinsically linked in human perception. Generating synchronized, contextually appropriate audio for silent videos can significantly enhance immersion in applications like virtual reality, gaming, and film. It also holds potential for video restoration, accessibility tools, and enriching AI-generated video content which often lacks sound. Our objective is to develop a robust video-to-audio generation model capable of producing high-quality, temporally aligned, and semantically meaningful audio.

## Features

* **Video-to-Audio Synthesis:** Generates audio waveforms from silent video inputs.
* **Cascaded Architecture:** Leverages powerful pre-trained models (CLIP, CLAP, Audio-LDM).
* **EchoFrame Mapper:** A learned module to translate visual embedding sequences (CLIP) into audio embeddings (CLAP). Several architectures explored (MLP, Bi-LSTM, Transformer).
* **High-Quality Audio Generation:** Utilizes a pre-trained Audio-LDM for synthesizing the final audio.
* **VGGSound Dataset:** Provides scripts and potentially pre-processed data/embeddings for the VGGSound dataset.
* **Publicly Available Resources:** Includes organized VGGSound dataset and embeddings hosted on Hugging Face. 

## Dataset: VGGSound

This project utilizes the **VGGSound dataset**, a large-scale collection of ~200,000 10-second video clips from YouTube, covering 309 diverse audio-visual classes.

* **Challenges:** Downloading and organizing VGGSound can be difficult due to deprecated links and its large size (~300-400 GB).
* **Contribution:** We have curated, fixed, and organized the VGGSound dataset, including labels. We provide this organized version and pre-computed CLIP (visual) and CLAP (audio) embeddings publicly on Hugging Face to aid the research community.
    * **Organized VGGSound Dataset:** [Link to your Hugging Face Dataset Repo] <span style="color:red;">**(Please Add Link)**</span> 
    * **Pre-computed Embeddings:** [Link to your Hugging Face Embeddings Repo] <span style="color:red;">**(Please Add Link)**</span> 

## Methodology: EchoFrame Pipeline

Our approach follows a cascaded pipeline:

1.  **Video Input:** A silent video clip (pre-processed to 10 seconds, frames extracted, resized, and normalized).
2.  **Visual Feature Extraction:** A pre-trained CLIP Image Encoder processes sampled video frames (e.g., 64 frames at 8 fps) into a sequence of visual embeddings ($E_{clip}$).
3.  **Visual-to-Audio Mapping (Echo Framer):** Our trained mapper network ($\mathcal{F}_{\theta}$) takes the sequence of CLIP embeddings ($E_{clip}$) and predicts a single corresponding CLAP audio embedding ($e_{clap\_pred}$). During training, this mapper is optimized using Mean Squared Error (MSE) loss against the target CLAP embedding ($e_{clap\_target}$) derived from the video's original audio using a pre-trained CLAP Audio Encoder.
4.  **Audio Synthesis:** The predicted CLAP embedding ($e_{clap\_pred}$) is fed into a pre-trained Audio-LDM (Latent Diffusion Model) generator ($\mathcal{G}$), which synthesizes the final audio waveform ($A_{gen}$).

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    * Install core dependencies (check for a `requirements.txt` or install manually).
    * Install the `v2a-mapper` package if needed for easier imports:
        ```bash
        pip install -e v2a-mapper
        ```
    * Ensure you have necessary libraries like PyTorch, torchvision, torchaudio, librosa, transformers, etc. (Refer to imports in scripts/notebooks).

4.  **Download Pre-trained Models:**
    * You might need to download pre-trained CLIP, CLAP, and Audio-LDM models manually or via provided scripts/huggingface libraries. Specify paths in configuration files or scripts if necessary.
    * Download our pre-trained EchoFrame mapper weights (`v2a-mapper/v2a_mapper/pre_trained/`).

5.  **Download Dataset:**
    * Use `download.py` or download the organized VGGSound dataset from our Hugging Face link.
    * Use `unzip.sh` / `tar_dataset.py` as needed for data management.

## Usage

### 1. Data Preparation & Feature Extraction

* Download and organize the VGGSound dataset using provided scripts (`download.py`, `unzip.sh`) or the Hugging Face link.
* Extract CLIP visual embeddings for your training/validation/test splits using `v2a-mapper/extraction/extract_video_training_latents.py`.
* Extract target CLAP audio embeddings (if not using pre-computed ones) using `get_encoding.ipynb` or similar CLAP processing scripts.

### 2. Training the EchoFrame Mapper

* Configure training parameters (dataset paths, model architecture, hyperparameters) potentially within `v2a-mapper/train/v2a-train.ipynb` or associated scripts (`run-*.py`).
* Launch training:
    ```bash
    # Example command (adapt based on your setup)
    python v2a-mapper/train/run-XX.py --config config_file.yaml
    # Or run cells within the v2a-train.ipynb notebook
    ```
* Trained model weights will be saved (e.g., in a `checkpoints` directory).

### 3. Inference (Generating Audio from Video)

* Use the main inference script `v2a-mapper/inference.py` or the notebook `v2a-mapper/train/v2a-inference.ipynb`.
* Provide the path to a silent input video and the trained EchoFrame mapper checkpoint.
    ```bash
    # Example command (adapt based on your setup)
    python v2a-mapper/inference.py \
        --video_path path/to/your/video.mp4 \
        --mapper_checkpoint path/to/your/mapper.pth \
        --output_path path/to/output/audio.wav
    ```
* An optional Gradio demo might be available via `v2a-mapper/app.py`.

### 4. Evaluation

* Use the `v2a-mapper/eval/eval.py` script to calculate FAD and MKL metrics.
* You will need generated audio samples and corresponding ground truth audio samples/embeddings.
* The `v2a-mapper/eval/data/` directory contains example output structures from previous evaluation runs.

## Results

We evaluated EchoFrame against baselines like SpecVQGAN, Im2Wav, and Diff-Foley using Fréchet Audio Distance (FAD) and Mean Kullback-Leibler Divergence (MKL). Lower scores are better.

| Model              | Visual Feature | Mapper Arch.   | FAD  ($\downarrow$) | MKL ($\downarrow$) |
| ------------------ | -------------- | -------------- | :-----------------: | :----------------: |
| SpecVQGAN          | ResNet50       | -              |        9.40         |        7.03        |
| Im2Wav             | CLIP           | -              |       11.44         |      **5.20** |
| Diff-Foley         | CAVP           | -              |        9.87         |        6.43        |
| **EchoFrame** | CLIP           | Bi-LSTM        |        7.38         |        6.29        |
| **EchoFrame** | CLIP           | Transformer    |        7.23         |        6.22        |
| **EchoFrame** | CLIP           | MLP w/o Res    |        6.30         |        5.83        |
| **EchoFrame (Best)** | CLIP           | **MLP w/ Res** |      **6.28** |        5.87        |

Our EchoFrame models consistently outperform the baselines on the FAD metric. The MLP-based mappers, particularly with residual connections, achieve the best FAD scores and competitive MKL scores, demonstrating the effectiveness of our approach.

## Limitations

The model's performance is sensitive to the quality and semantic coherence of the audio-visual pairs in the VGGSound training data. It may struggle to generate appropriate audio for scenes where the visual content and the expected sound lack a strong, direct correlation learned during training.

## Future Work


* Explore alternative architectures for the EchoFrame mapper.
* Investigate end-to-end training using audio-visual contrastive learning.
* Incorporate more comprehensive subjective evaluations (human ratings) for quality, relevance, and synchronization.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

* The creators of the VGGSound dataset.
* Developers of CLIP, CLAP, and Audio-LDM models.
* PSC (Pittsburgh Supercomputing Center) for computational resources (if applicable).

## References

(Refer to the `11785_Final_Report.pdf` for a full list of references).

* Wang, H., Ma, J., Pascual, S., Cartwright, R., & Cai, W. (2024). V2a-mapper: A lightweight solution for vision-to-audio generation by connecting foundation models. *AAAI*.
* Chen, H., Xie, W., & Zisserman, A. (2020). Vggsound: A large-scale audio-visual dataset. *CVPR*.
* Koutini, K., Schlüter, J., Eghbal-Zadeh, H., & Widmer, G. (2021). Efficient training of audio transformers with patchout. *arXiv*.
* Kilgour, K., Zuluaga, M., Roblek, D., & Sharifi, M. (2018). Fréchet audio distance. *arXiv*.
* Iashin, V., & Rahtu, E. (2021). Taming visually guided sound generation. *arXiv*.
* Pascual, S., Yeh, C., Tsiamas, I., & Serrà, J. (2024). Masked generative video-to-audio transformers with enhanced synchronicity. *ECCV*.


