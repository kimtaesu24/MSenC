import os
import torch
import wandb
import time
import json
import sys
import pandas as pd

from datasets import load_dataset
from data_loader.Conversation_data_loader import Conversation_Dataset
from data_loader.Conversation_data_loader import collate_fn
from torch.utils.data import DataLoader
from model.architecture import MyArch
from model.architecture_speechtokenizer import MyArch_spch
from eval_metric.coco_eval import calculate_eval_matric
from tqdm import tqdm
from transformers import Blip2Model
from loguru import logger
from utils.util import get_model_info, dict_to_gpu


class MyTrainer:
    def __init__(self, device, data_path, args):
        self.device = device
        self.data_path = data_path
        self.args = args

    def train_with_hyper_param(self):
        # save dir create
        save_dir = f"LLM_freeze_{self.args.LLM_freeze}/{self.args.data_name}/{self.args.target}/{self.args.modal}/"
        checkpoint_directory = f"/data4/s20235100/ckpt/{save_dir}/"
        json_directory = f"/data4/s20235100/json/{save_dir}/"
        try:
            if not os.path.exists(checkpoint_directory):
                os.makedirs(checkpoint_directory)
            if not os.path.exists(json_directory):
                os.makedirs(json_directory)
        except OSError:
            logger.info("Error: Failed to create the directory.")
            exit()
        
        
        # model load
        blip2 = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b")
        if self.args.stage == 1:
            model = MyArch(self.args, blip2.qformer, blip2.query_tokens, json_directory)
        elif self.args.stage == 2:
            model = MyArch_spch(self.args, blip2.qformer, blip2.query_tokens, json_directory)
        elif self.args.stage == 0:  # single stage training
            model = MyArch_spch(self.args, blip2.qformer, blip2.query_tokens, json_directory)
        else:
            sys.exit(f"select correct stage. you selected stage: {self.args.stage}")


        # load finetuned Q-former or not
        if self.args.QFormer != 'blip2':  # for convenience of running code
            pretrained_dict = torch.load(self.args.QFormer)["model_state_dict"]
            
            video_qformer_dict = model.video_q_former.state_dict()
            audio_qformer_dict = model.audio_q_former.state_dict()
            pretrained_v_dict = {}
            pretrained_a_dict = {}
            for k, v in pretrained_dict.items():
                if 'video_q_former' in k:
                    if k.split('video_q_former.')[1] in video_qformer_dict.keys():
                        pretrained_v_dict[k.split('video_q_former.')[1]] = v
                if 'audio_q_former' in k:
                    if k.split('audio_q_former.')[1] in audio_qformer_dict.keys():
                        pretrained_a_dict[k.split('audio_q_former.')[1]] = v

            video_qformer_dict.update(pretrained_v_dict)
            model.video_q_former.load_state_dict(video_qformer_dict)

            audio_qformer_dict.update(pretrained_a_dict)
            model.audio_q_former.load_state_dict(audio_qformer_dict)
            logger.info(f'finetuned Q-Former has loaded.')
        
        # resume
        if self.args.resume > 0:
            checkpoint = torch.load(f"{checkpoint_directory}/{str(self.args.resume)}_epochs.tar")
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f'ckpt has loaded at epoch:{self.args.resume}')
            # epoch = args.resume
            
        # freeze
        if self.args.LLM_freeze:
            for parameters in model.generator_model.parameters():
                parameters.requires_grad = False
            logger.info('LLM has freezed')
        
        if self.args.QFormer_freeze:
            for parameters in model.video_q_former.parameters():
                parameters.requires_grad = False
            for parameters in model.audio_q_former.parameters():
                parameters.requires_grad = False
            logger.info('QFormer has freezed')

        model.to(self.device)
        get_model_info(model)
            
            
        # connect to wandb
        if not self.args.debug: 
            wandb.init(project=self.args.wandb_name)
            wandb.run.name = f"{save_dir} - {time.strftime('%c', time.localtime(time.time()))}"
            
            
        # train data loader
        train_csv = pd.read_csv(os.path.join(self.data_path, 'train/train_dataset.csv'))
        train_csv = self.construct_label(train_csv, mode='train')

        train_max_utt = train_csv.groupby('Dialogue_ID')['Utterance_ID'].max()
        train_filtered_csv = train_csv[~train_csv.apply(lambda row: row['Utterance_ID'] == train_max_utt[row['Dialogue_ID']], axis=1)]
        train_dataset = Conversation_Dataset(train_filtered_csv, args=self.args)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=torch.cuda.device_count()*4)
        
        # valid data loader
        valid_csv = pd.read_csv(os.path.join(self.data_path, 'valid/valid_dataset.csv'))
        valid_csv = self.construct_label(valid_csv, mode='valid')
        
        valid_max_utt = valid_csv.groupby('Dialogue_ID')['Utterance_ID'].max()
        valid_filtered_csv = valid_csv[~valid_csv.apply(lambda row: row['Utterance_ID'] == valid_max_utt[row['Dialogue_ID']], axis=1)]
        valid_dataset = Conversation_Dataset(valid_filtered_csv, args=self.args)
        valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=self.args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=torch.cuda.device_count()*4)
        
        # test data loader
        test_csv = pd.read_csv(os.path.join(self.data_path, 'test/test_dataset.csv'))
        test_csv = self.construct_label(test_csv, mode='test')
        
        test_max_utt = test_csv.groupby('Dialogue_ID')['Utterance_ID'].max()
        test_filtered_csv = test_csv[~test_csv.apply(lambda row: row['Utterance_ID'] == test_max_utt[row['Dialogue_ID']], axis=1)]
        test_dataset = Conversation_Dataset(test_filtered_csv, args=self.args)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=self.args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=torch.cuda.device_count()*4)


        # train start
        train_batch_len = len(train_dataloader)
        valid_batch_len = len(valid_dataloader)
        test_batch_len = len(test_dataloader)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.args.decay_rate)
        
        pbar = tqdm(range(1+self.args.resume, self.args.epochs+1), position=0, leave=False, desc='epoch')
        for epoch in pbar:
            train_total_loss = 0
            train_text_loss = 0
            train_speech_loss1 = 0
            train_speech_loss2 = 0
            train_speech_loss3 = 0
            train_speech_loss4 = 0
            train_speech_loss5 = 0
            train_speech_loss6 = 0
            train_speech_loss7 = 0
            train_speech_loss8 = 0
            
            valid_total_loss = 0
            valid_text_loss = 0
            valid_speech_loss1 = 0
            valid_speech_loss2 = 0
            valid_speech_loss3 = 0
            valid_speech_loss4 = 0
            valid_speech_loss5 = 0
            valid_speech_loss6 = 0
            valid_speech_loss7 = 0
            valid_speech_loss8 = 0
            
            test_total_loss = 0
            test_text_loss = 0
            test_speech_loss1 = 0
            test_speech_loss2 = 0
            test_speech_loss3 = 0
            test_speech_loss4 = 0
            test_speech_loss5 = 0
            test_speech_loss6 = 0
            test_speech_loss7 = 0
            test_speech_loss8 = 0
            
            # training
            model.train()
            prog_bar = tqdm(train_dataloader, position=1,leave=False, desc='batch')
            for i, (inputs, labels, d_name) in enumerate(prog_bar):
                # print(d_name)
                inputs[1] = dict_to_gpu(inputs[1], self.device)
                inputs[2] = dict_to_gpu(inputs[2], self.device)
                
                optimizer.zero_grad()
                text_loss, speech_token_loss = model(inputs, labels)
                
                loss = text_loss + sum(speech_token_loss)

                prog_bar.set_postfix({'loss': loss.item()})

                train_total_loss += loss.item()
                train_text_loss += text_loss.item()
                train_speech_loss1 += speech_token_loss[0].item()
                train_speech_loss2 += speech_token_loss[1].item()
                train_speech_loss3 += speech_token_loss[2].item()
                train_speech_loss4 += speech_token_loss[3].item()
                train_speech_loss5 += speech_token_loss[4].item()
                train_speech_loss6 += speech_token_loss[5].item()
                train_speech_loss7 += speech_token_loss[6].item()
                train_speech_loss8 += speech_token_loss[7].item()
                
                loss.backward()
                optimizer.step()


                if not self.args.debug:  # code for debug mode
                    if i % (100//self.args.batch_size) == 0:
                        wandb.log({'train_total_loss': loss.item()})
            
            
            with torch.no_grad():
                model.eval()
                metric_log = (epoch) % self.args.metric_at_every == 0
                # validation
                for inputs, labels, d_name in tqdm(valid_dataloader, position=1, leave=False, desc='batch'):
                    # print(d_name)
                    inputs[1] = dict_to_gpu(inputs[1], self.device)
                    inputs[2] = dict_to_gpu(inputs[2], self.device)
                    
                    text_loss, speech_token_loss = model(inputs, labels, metric_log=metric_log, epoch=epoch, mode='valid')
                    
                    loss = text_loss + sum(speech_token_loss)
                    
                    valid_total_loss += loss.item()
                    valid_text_loss += text_loss.item()
                    valid_speech_loss1 += speech_token_loss[0].item()
                    valid_speech_loss2 += speech_token_loss[1].item()
                    valid_speech_loss3 += speech_token_loss[2].item()
                    valid_speech_loss4 += speech_token_loss[3].item()
                    valid_speech_loss5 += speech_token_loss[4].item()
                    valid_speech_loss6 += speech_token_loss[5].item()
                    valid_speech_loss7 += speech_token_loss[6].item()
                    valid_speech_loss8 += speech_token_loss[7].item()
                    
                # test
                for inputs, labels, d_name in tqdm(test_dataloader, position=1, leave=False, desc='batch'):
                    # print(d_name)
                    inputs[1] = dict_to_gpu(inputs[1], self.device)
                    inputs[2] = dict_to_gpu(inputs[2], self.device)
                    
                    text_loss, speech_token_loss = model(inputs, labels, metric_log=metric_log, epoch=epoch, mode='test')
                    
                    loss = text_loss + sum(speech_token_loss)
                    
                    test_total_loss += loss.item()
                    test_text_loss += text_loss.item()
                    test_speech_loss1 += speech_token_loss[0].item()
                    test_speech_loss2 += speech_token_loss[1].item()
                    test_speech_loss3 += speech_token_loss[2].item()
                    test_speech_loss4 += speech_token_loss[3].item()
                    test_speech_loss5 += speech_token_loss[4].item()
                    test_speech_loss6 += speech_token_loss[5].item()
                    test_speech_loss7 += speech_token_loss[6].item()
                    test_speech_loss8 += speech_token_loss[7].item()
            
            # log
            output_loss_dict = {'train_total_loss (epoch)': train_total_loss/train_batch_len,
                                 'train_text_loss (epoch)': train_text_loss/train_batch_len,
                                 'train_speech_loss1 (epoch)': train_speech_loss1/train_batch_len,
                                 'train_speech_loss2 (epoch)': train_speech_loss2/train_batch_len,
                                 'train_speech_loss3 (epoch)': train_speech_loss3/train_batch_len,
                                 'train_speech_loss4 (epoch)': train_speech_loss4/train_batch_len,
                                 'train_speech_loss5 (epoch)': train_speech_loss5/train_batch_len,
                                 'train_speech_loss6 (epoch)': train_speech_loss6/train_batch_len,
                                 'train_speech_loss7 (epoch)': train_speech_loss7/train_batch_len,
                                 'train_speech_loss8 (epoch)': train_speech_loss8/train_batch_len,
                                 'valid_total_loss (epoch)': valid_total_loss/valid_batch_len,
                                 'valid_text_loss (epoch)': valid_text_loss/valid_batch_len,
                                 'valid_speech_loss1 (epoch)': valid_speech_loss1/valid_batch_len,
                                 'valid_speech_loss2 (epoch)': valid_speech_loss2/valid_batch_len,
                                 'valid_speech_loss3 (epoch)': valid_speech_loss3/valid_batch_len,
                                 'valid_speech_loss4 (epoch)': valid_speech_loss4/valid_batch_len,
                                 'valid_speech_loss5 (epoch)': valid_speech_loss5/valid_batch_len,
                                 'valid_speech_loss6 (epoch)': valid_speech_loss6/valid_batch_len,
                                 'valid_speech_loss7 (epoch)': valid_speech_loss7/valid_batch_len,
                                 'valid_speech_loss8 (epoch)': valid_speech_loss8/valid_batch_len,
                                 'test_total_loss (epoch)': test_total_loss/test_batch_len,
                                 'test_text_loss (epoch)': test_text_loss/test_batch_len,
                                 'test_speech_loss1 (epoch)': test_speech_loss1/test_batch_len,
                                 'test_speech_loss2 (epoch)': test_speech_loss2/test_batch_len,
                                 'test_speech_loss3 (epoch)': test_speech_loss3/test_batch_len,
                                 'test_speech_loss4 (epoch)': test_speech_loss4/test_batch_len,
                                 'test_speech_loss5 (epoch)': test_speech_loss5/test_batch_len,
                                 'test_speech_loss6 (epoch)': test_speech_loss6/test_batch_len,
                                 'test_speech_loss7 (epoch)': test_speech_loss7/test_batch_len,
                                 'test_speech_loss8 (epoch)': test_speech_loss8/test_batch_len,
                                 }
            
            if metric_log:
                if 'text' in self.args.target:
                    with open(json_directory + f'epoch{epoch}_valid.json', 'r') as f:
                        output_json = json.load(f)
                    eval_result = get_metrics(output_json, self.args.target)

                    valid_output_metric_dict = {'valid_Bleu-1 (epoch)': eval_result['Bleu_1']*100,
                                                'valid_Bleu-2 (epoch)': eval_result['Bleu_2']*100,
                                                'valid_Bleu-3 (epoch)': eval_result['Bleu_3']*100,
                                                'valid_Bleu-4 (epoch)': eval_result['Bleu_4']*100,
                                                'valid_METEOR (epoch)': eval_result['METEOR']*100,
                                                'valid_ROUGE_L (epoch)': eval_result['ROUGE_L']*100,
                                                'valid_CIDEr (epoch)': eval_result['CIDEr']*100,
                                                'valid_SPICE (epoch)': eval_result['SPICE']*100,
                                                }
                    with open(json_directory + f'epoch{epoch}_test.json', 'r') as f:
                        output_json = json.load(f)
                    eval_result = get_metrics(output_json, self.args.target)

                    test_output_metric_dict = {'test_Bleu-1 (epoch)': eval_result['Bleu_1']*100,
                                               'test_Bleu-2 (epoch)': eval_result['Bleu_2']*100,
                                               'test_Bleu-3 (epoch)': eval_result['Bleu_3']*100,
                                               'test_Bleu-4 (epoch)': eval_result['Bleu_4']*100,
                                               'test_METEOR (epoch)': eval_result['METEOR']*100,
                                               'test_ROUGE_L (epoch)': eval_result['ROUGE_L']*100,
                                               'test_CIDEr (epoch)': eval_result['CIDEr']*100,
                                               'test_SPICE (epoch)': eval_result['SPICE']*100,
                                               }
                    
            
            if not self.args.debug:  # code for debug mode
                wandb.log(output_loss_dict)
                if metric_log:
                    if 'text' in self.args.target:
                        wandb.log(valid_output_metric_dict)
                        wandb.log(test_output_metric_dict)
            
            # save checkpoint
            if (epoch) % self.args.save_at_every == 0:
                CHECKPOINT_PATH = f"{checkpoint_directory}/{str(epoch)}_epochs.tar"
                torch.save({ # Save our checkpoint loc
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, CHECKPOINT_PATH)
                pbar.write('Checkpoint model has saved at Epoch: {:02} '.format(epoch))
                
            scheduler.step()  # per epochs
            train_dataloader.dataset.shuffle()  # data shuffle
        pbar.close()

        return model
    
    def construct_label(self, csv, mode):
        csv['label_Utterance'] = csv['Utterance'].shift(-1)
        csv['label_audio_path'] = csv['audio_path'].shift(-1)
        
        if 'description' in self.args.target:
            if self.args.data_name == 'MELD':
                csv['description'] = load_dataset('TAESOO98/meld-transcript-final')[mode]['text_description']
            elif self.args.data_name == 'MSC':
                csv['description'] = load_dataset('TAESOO98/msc-transcript-final')[mode]['text_description']
                
            csv['label_description'] = csv['description'].shift(-1)
        
        return csv

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


if __name__=='__main__':
    with open("/data4/s20235100/json/stage1/MELD/text_description/LLM_freeze_True/epoch12.json", 'r') as f:
        output_json = json.load(f)
    eval_result = get_metrics(output_json, 'description')
    output_metric_dict = {'valid_Bleu-1 (epoch)': eval_result['Bleu_1']*100,
                            'valid_Bleu-2 (epoch)': eval_result['Bleu_2']*100,
                            'valid_Bleu-3 (epoch)': eval_result['Bleu_3']*100,
                            'valid_Bleu-4 (epoch)': eval_result['Bleu_4']*100,
                            'valid_METEOR (epoch)': eval_result['METEOR']*100,
                            'valid_ROUGE_L (epoch)': eval_result['ROUGE_L']*100,
                            'valid_CIDEr (epoch)': eval_result['CIDEr']*100,
                            'valid_SPICE (epoch)': eval_result['SPICE']*100,
                            }
    print(output_metric_dict)
    