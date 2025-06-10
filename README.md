# MultiSensory Conversational Agent

## Requirment
* python 3.8.17
* install requirments
```
pip install -r requirment.txt
```

## Data Processing
- The pre-processing code for the MSC dataset is located in the `./data/preprocessing` directory.
    - For more detailed information, please refer to the `./data/preprocessing/README.md` file.
- The feature extraction code for MSC dataset is located in the `./data`
    - For more detailed information, please refer to the `./data/README.md` file.


## Repository Structure

The overall file structure of this repository is as follows:

```
Template
    ├── README.md                       
    ├── requirments.txt
    ├── data
    └── src         
        ├── train.py                 # implements a function for training the model with hyperparameters
        ├── inference.py             # implements a function for inference the model
        ├── utils
        │   └── utils.py             # contains utility functions such as setting random seed and showing hyperparameters
        ├── trainer
        │   └── trainer.py           # processes input arguments of a user for training
        ├── data_loader
        │   ├── MSE_data_loader.py
        └── models                      
            ├── architecture.py      # implements the forward function and architecture
            └── modules.py           
```

## training

- You can train the model with MSC dataset with following command:

```
python train.py --data_name MELD --stage 1 --QFormer blip2 --max_len 200 --target text_description --audio_type wavlm --bs 6
python train.py --data_name MSC --stage 1 --QFormer blip2 --max_len 200 --target text_description --audio_type wavlm --bs 6
```
- We provide code for training with text targets (without using voice descriptions).

```
python train.py --data_name MELD --stage 1 --QFormer blip2 --max_len 30 --target text --audio_type wavlm --bs 8
python train.py --data_name MSC --stage 1 --QFormer blip2 --max_len 30 --target text --audio_type wavlm --bs 8
```

## Ablation
```
python train.py --data_name MELD --stage 1 --LLM mistral1 --target text_description --max_length 200 --QFormer blip2 --bs 6 --modal text --wandb_name Description-ablation --epoch 10 --save 2
python train.py --data_name MELD --stage 1 --LLM mistral1 --target text_description --max_length 200 --QFormer blip2 --bs 6 --modal text_audio --wandb_name Description-ablation --epoch 10 --save 2
python train.py --data_name MELD --stage 1 --LLM mistral1 --target text_description --max_length 200 --QFormer blip2 --bs 6 --modal text_video --wandb_name Description-ablation --epoch 10 --save 2

python train.py --data_name MSC --stage 1 --LLM mistral1 --target text_description --max_length 200 --QFormer blip2 --bs 6 --modal text --wandb_name Description-ablation --epoch 10 --save 2
python train.py --data_name MSC --stage 1 --LLM mistral1 --target text_description --max_length 200 --QFormer blip2 --bs 6 --modal text_audio --wandb_name Description-ablation --epoch 10 --save 2
python train.py --data_name MSC --stage 1 --LLM mistral1 --target text_description --max_length 200 --QFormer blip2 --bs 6 --modal text_video --wandb_name Description-ablation --epoch 10 --save 2
```

## Impact of LLM finetuning
```
python train.py --data_name MSC --stage 1 --LLM mistral1 --LLM_freeze --target text_description --max_length 200 --QFormer blip2 --bs 6 --modal text_audio_video --wandb_name "Audio Latent Generation" --epoch 10 --save 2
python train.py --data_name MELD --stage 1 --LLM mistral1 --LLM_freeze --target text_description --max_length 200 --QFormer blip2 --bs 6 --modal text_audio_video --wandb_name "Audio Latent Generation" --epoch 10 --save 2
```
