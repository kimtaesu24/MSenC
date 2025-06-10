import os
import torch
import argparse
import json
import sys
import pandas as pd

from datasets import load_dataset
from data_loader.Conversation_data_loader import Conversation_Dataset
from data_loader.Conversation_data_loader import collate_fn
from torch.utils.data import DataLoader
from model.architecture import MyArch
from model.architecture_speechtokenizer import MyArch_spch
from utils.util import set_random_seed, log_args
from eval_metric.coco_eval import calculate_eval_matric
from tqdm import tqdm
from transformers import Blip2Model
from utils.util import get_model_info, dict_to_gpu


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_metrics(output_json, target):
    outputs_sentence = []
    ref_sentence = []
    for dat in output_json:
        outputs_sentence += dat["output_sentence"]
        ref_sentence += dat["ref_sentence"]
    
    if 'description' in target:
        
        utt_list = []
        for sentence in outputs_sentence:
            utt, desc = split_at_first_parenthesis(sentence)
            utt_list.append(utt)
        outputs_sentence = utt_list
        
        utt_list = []
        for sentence in ref_sentence:
            utt, desc = split_at_first_parenthesis(sentence)
            utt_list.append(utt)
        ref_sentence = utt_list

    eval_result = calculate_eval_matric(outputs_sentence, ref_sentence)
    return eval_result

def split_at_first_parenthesis(s):
    before_parenthesis, separator, after_parenthesis = s.partition('(')
    return before_parenthesis, separator + after_parenthesis

def construct_label(csv, mode, target, data_name):
        csv['label_Utterance'] = csv['Utterance'].shift(-1)
        csv['label_audio_path'] = csv['audio_path'].shift(-1)
        
        if 'description' in target:
            if data_name == 'MELD':
                csv['description'] = load_dataset('TAESOO98/meld-transcript-final')[mode]['text_description']
            elif data_name == 'MSC':
                csv['description'] = load_dataset('TAESOO98/msc-transcript-final')[mode]['text_description']
                
            csv['label_description'] = csv['description'].shift(-1)
        
        return csv

def infer(args):
    set_random_seed(seed=args.seed, device=args.device)
    
    save_dir = f"LLM_freeze_{args.LLM_freeze}/{args.data_name}/{args.target}/{args.modal}/"
    save_dir = f'/data4/s20235100/json/{save_dir}/output/{args.target}/stage{args.stage}/ckpt_{args.ckpt}/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    blip2 = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b")
    if args.stage == 1:
        model = MyArch(args, blip2.qformer, blip2.query_tokens, json_directory=save_dir)
    elif args.stage == 2:
        model = MyArch_spch(args, blip2.qformer, blip2.query_tokens, json_directory=save_dir)
    elif args.stage == 0:  # single stage training
        model = MyArch_spch(args, blip2.qformer, blip2.query_tokens, json_directory=save_dir)
    else:
        sys.exit(f"select correct stage. you selected stage: {args.stage}")
        
    model.load_state_dict(torch.load(f"/data4/s20235100/ckpt/LLM_freeze_{args.LLM_freeze}/stage{args.stage}/{args.data_name}/{args.target}/{args.ckpt}_epochs.tar", map_location=args.device)["model_state_dict"])
    
    print('checkpoint model has loaded!')
    model.to(args.device)
    
    data_path = f'/home2/dataset/Empathetic_Dataset/{args.data_name}/'
    csv = pd.read_csv(os.path.join(data_path, f'{args.mode}/{args.mode}_dataset.csv'))
    csv = construct_label(csv, mode=args.mode, target=args.target, data_name=args.data_name)
    
    max_utt = csv.groupby('Dialogue_ID')['Utterance_ID'].max()
    filtered_csv = csv[~csv.apply(lambda row: row['Utterance_ID'] == max_utt[row['Dialogue_ID']], axis=1)]
    ds = Conversation_Dataset(filtered_csv, args=args)
    dl = DataLoader(dataset=ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=torch.cuda.device_count()*4)

    with torch.no_grad():
        model.eval()
        for inputs, labels, d_name in tqdm(dl, position=0, leave=False, desc='batch'):
            inputs[1] = dict_to_gpu(inputs[1], args.device)
            inputs[2] = dict_to_gpu(inputs[2], args.device)
            # print(d_name)
            output = model.predict(inputs, labels, d_name, ckpt_epoch=args.ckpt)
            
    with open(save_dir + f'epoch{args.ckpt}_test.json', 'r') as f:
        output_json = json.load(f)
    eval_result = get_metrics(output_json, args.target)

    test_output_metric_dict = {'test_Bleu-1 (epoch)': eval_result['Bleu_1']*100,
                                'test_Bleu-2 (epoch)': eval_result['Bleu_2']*100,
                                'test_Bleu-3 (epoch)': eval_result['Bleu_3']*100,
                                'test_Bleu-4 (epoch)': eval_result['Bleu_4']*100,
                                'test_METEOR (epoch)': eval_result['METEOR']*100,
                                'test_ROUGE_L (epoch)': eval_result['ROUGE_L']*100,
                                'test_SPICE (epoch)': eval_result['SPICE']*100,
                                'test_CIDEr (epoch)': eval_result['CIDEr']*100,
                                }
    print(f'Metric Result at ckpt {args.ckpt}:', test_output_metric_dict)
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='MSC', help='select dataset for training')
    parser.add_argument('--stage', type=int, default=1, help='train model on stage 1 or 2')
    parser.add_argument('--LLM', default='mistral1', help='select LLM for generator')
    parser.add_argument('--LLM_freeze', action='store_true', default=False, help='train LLM or freeze')
    parser.add_argument('--QFormer', default='/data4/s20235100/ckpt/stage1/MELD/text/30_epochs.tar', help='select QFormer')
    parser.add_argument('--QFormer_freeze', action='store_true', default=False, help='freeze QFormer or not')

    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--act', default='relu', help='type of activation function')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')

    parser.add_argument('--max_history', type=int, default=800, help='number of history input to generator')
    parser.add_argument('--audio_type', default='wavlm', help='audio feature extract type |wavlm')
    parser.add_argument('--audio_pad_size', type=int, default=800, help='padding size for audio history')
    parser.add_argument('--video_pad_size', type=int, default=50, help='padding size for video history')
    parser.add_argument('--fps', type=int, default=3, help='number of frame per second to input to model')
    parser.add_argument('--max_length', type=int, default=30, help='maximum length of utterance')
    parser.add_argument('--target', default='text', help='which modality to generate')
    parser.add_argument('--modal', default='audio_video_text', help='which modality to train')
    
    parser.add_argument('--mode', default='test', help='train | valid | test')
    parser.add_argument('--ckpt', default=30, help='select checkpoint epoch')
    args = parser.parse_args()

    if args.LLM == 'mistral1':
        args.LLM = 'mistralai/Mistral-7B-v0.1'
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    
    log_args(args)
    
    infer(args)
