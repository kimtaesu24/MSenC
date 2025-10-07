# MultiSensory Conversational Agent

Official PyTorch implementation of the Interspeech 2025 paper:<br>
“Towards Human-like Multimodal Conversational Agent by Generating Engaging Speech”

[![arXiv](https://img.shields.io/badge/Paper-arXiv.14627-b31b1b.svg)](http://arxiv.org/abs/2509.14627)
[![arXiv](https://img.shields.io/badge/Porject-page-00ff00.svg)](https://kimtaesu24.github.io/)

![MSC_description](./assets/model_arch.png)

## Overview
Human communication isn’t just about words — it’s **multisensory**, blending **language, voice, and visual cues** that work together to convey meaning.
While text captures what we say, **speech delivers how we say it** — tone, emotion, and personality.

Most **multimodal LLMs** today focus on generating text responses, but they often miss the **expressiveness** that comes from natural speech.
To bridge this gap, we introduce a **human-like multimodal conversational agent** that generates **speech responses** reflecting the mood and style of a conversation.

To make this possible:

- 🎧 We build the **Multi-Sensory Conversation (MSenC)** dataset, a speech-centered dataset integrating text, audio, and visual cues.

- 🧠 We propose a **multimodal LLM-based model** that generates both **text responses** and **voice descriptions**, which guide the generation of **expressive and engaging speech**.

Experiments show that combining **visual and audio modalities** helps the model produce **more natural, human-like speech**, making conversations sound alive rather than mechanical.

## Requirment
* python 3.8.17
* install requirments
```
pip install -r requirment.txt
```

## Checkpoint

Download pretrained models (text + audio + video) trained on MSenC dataset:
- [<u>Google Drive</u>](https://drive.google.com/file/d/1KHHxHNNxM_fPSiyGQLMP3g-gU1bp--jS/view?usp=sharing)
- (Hugging Face model hub version coming soon!)

## Training

- Train the model with MSC or MELD dataset:
```
python train.py --data_name MELD --stage 1 --QFormer blip2 --max_len 200 --target text_description --audio_type wavlm --bs 6
python train.py --data_name MSC --stage 1 --QFormer blip2 --max_len 200 --target text_description --audio_type wavlm --bs 6
```

<details> <summary>💡 Ablation Studies (click to expand)</summary>

- Run ablations for different modality combinations:
```
python train.py --data_name MSC --stage 1 --LLM mistral1 --target text_description --max_length 200 --QFormer blip2 --bs 6 --modal text --wandb_name Description-ablation --epoch 10 --save 2
python train.py --data_name MSC --stage 1 --LLM mistral1 --target text_description --max_length 200 --QFormer blip2 --bs 6 --modal text_audio --wandb_name Description-ablation --epoch 10 --save 2
python train.py --data_name MSC --stage 1 --LLM mistral1 --target text_description --max_length 200 --QFormer blip2 --bs 6 --modal text_video --wandb_name Description-ablation --epoch 10 --save 2
```
</details>

<details> <summary>💡 LLM Fine-tuning Impact (click to expand)</summary>
    
- Explore the effect of fine-tuning the language model:
```
python train.py --data_name MSC --stage 1 --LLM mistral1 --LLM_freeze --target text_description --max_length 200 --QFormer blip2 --bs 6 --modal text_audio_video --wandb_name "Audio Latent Generation" --epoch 10 --save 2
```
</details>

## Inference
- Once training is done (or using the pretrained checkpoint), you can generate responses and speech directly using inference.py.

Basic Usage:
```
python inference.py \
  --checkpoint_path ./checkpoints/best_model.pt \
  --input_text "That’s amazing!" \
  --video_path ./sample_video.mp4 \
  --audio_path ./sample_audio.wav \
  --save_path ./outputs/
```

Text Only:
```
python inference.py \
  --checkpoint_path ./checkpoints/best_model.pt \
  --input_text "How was your day?" \
  --save_path ./outputs/
```

## Data Processing
- The preprocessing pipeline for the MSenC dataset is located in:
```
./data/preprocessing
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;➡️ See [./data/preprocessing/README.md](https://github.com/kimtaesu24/MSenC/tree/master/data/preprocessing) for details.


- The feature extraction and speech dexcription excraction codes for MSenC dataset is located in:
```
./data
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;➡️ Refer to [./data/README.md](https://github.com/kimtaesu24/MSenC/tree/master/data/) for a complete guide.


## Repository Structure
```
MSenC
    ├── README.md                       
    ├── requirments.txt
    ├── data
    │   ├── preprocessing
    │   ├── audio_feature_wavlm.py
    │   └── video_feature_extract.py
    └── src         
        ├── train.py                 # implements a function for training the model with hyperparameters
        ├── inference.py             # implements a function for inference the model
        ├── utils
        │   └── utils.py             # contains utility functions such as setting random seed and showing hyperparameters
        ├── trainer
        │   └── trainer.py           # processes input arguments of a user for training
        ├── data_loader
        │   └── MSE_data_loader.py
        └── models                      
            └── architecture.py      # implements the forward function and architecture
```


## Citation
If you find this repository useful, please consider citing our work:
```
@inproceedings{kim25m_interspeech,
  title     = {{Towards Human-like Multimodal Conversational Agent by Generating Engaging Speech}},
  author    = {Taesoo Kim and Yongsik Jo and Hyunmin Song and Taehwan Kim},
  year      = {2025},
  booktitle = {{Interspeech 2025}},
  pages     = {4828--4832},
  doi       = {10.21437/Interspeech.2025-1075},
  issn      = {2958-1796},
}
```




