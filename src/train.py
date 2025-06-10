import torch
import argparse
import os

from utils.util import set_random_seed, log_args
from trainer.trainer import MyTrainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def run_mymodel(device, data_path, args):
    trainer = MyTrainer(device=device,
                        data_path=data_path,
                        args=args,
                        )
    
    trainer.train_with_hyper_param()

def main(args):
    # Step 0. Initialization
    set_random_seed(seed=args.seed, device=args.device)

    # Step 1. Load datasets
    data_path = f'/home2/dataset/Empathetic_Dataset/{args.data_name}/'
    
    # Step 2. Run (train and evaluate) the specified model
    run_mymodel(device=args.device,
                data_path=data_path,
                args=args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_name', default='MSC', help='select dataset for training')
    parser.add_argument('--stage', type=int, default=-1, help='train model on stage 1 or 2')
    parser.add_argument('--LLM', default='mistral1', help='select LLM for generator')
    parser.add_argument('--LLM_freeze', action='store_true', default=False, help='freeze LLM or not')
    parser.add_argument('--QFormer', default='/data4/s20235100/ckpt/stage1/MELD/text_description/LLM_freeze_True/12_epochs.tar', help='select QFormer')
    parser.add_argument('--QFormer_freeze', action='store_true', default=False, help='freeze QFormer or not')
    
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
    parser.add_argument('--act', default='relu', help='type of activation function')
    parser.add_argument('--batch_size', '--bs', type=int, default=4, help='batch size')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--decay_rate', type=float, default=0.98, help='decay rate of learning rate')
    
    parser.add_argument('--max_history', type=int, default=800, help='number of history input to generator')
    parser.add_argument('--audio_type', default='wavlm', help='audio feature type |wavlm|wav2vec2')
    parser.add_argument('--audio_pad_size', type=int, default=800, help='padding size for audio history')
    parser.add_argument('--video_pad_size', type=int, default=50, help='padding size for video history')
    parser.add_argument('--fps', type=int, default=3, help='number of frame per second to input to model')
    parser.add_argument('--max_length', type=int, default=30, help='maximum length of utterance')
    parser.add_argument('--target', default='text', help='which modality to generate')
    parser.add_argument('--modal', default='audio_video_text', help='which modality to train')

    parser.add_argument('--wandb_name', default='Description', help='name of wandb project')
    parser.add_argument('--save_at_every', '--save', type=int, default=3, help='save checkpoint')
    parser.add_argument('--metric_at_every', '--metric', type=int, default=1, help='calculate metric scores')
    parser.add_argument('--resume', type=int, default=0, help='resume train or not')
    parser.add_argument('--debug', action='store_true', default=False, help='debug mode for wandb')
    
    args = parser.parse_args()
    
    if args.LLM == 'mistral1':
        args.LLM = 'mistralai/Mistral-7B-v0.1'
    elif args.LLM == 'mistral3':
        args.LLM ='mistralai/Mistral-7B-v0.3'
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
        
    log_args(args)
    main(args)
