import torch
import json
import torch.nn.functional as F
import os
import numpy as np

from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType


class q_former(nn.Module):
    def __init__(self, input_dim, output_dim, qformer, query):
        super(q_former, self).__init__()
        
        self.qformer = qformer
        self.query = query
        
        self.feature_proj = nn.Linear(input_dim, 1408, bias=False)  # "openai/clip-vit-base-patch32" to "Salesforce/blip2-opt-2.7b"
        self.out_proj = nn.Linear(768, output_dim, bias=False)
        
    def forward(self, inputs):
        inputs = self.feature_proj(torch.unsqueeze(inputs, dim=0))
        query = self.query.expand(inputs.shape[0], -1, -1)
        
        output = self.qformer(query, encoder_hidden_states=inputs)
        
        proj_output = self.out_proj(output.last_hidden_state)  # language space projection
        
        return proj_output.squeeze()

class MyArch(torch.nn.Module):
    def __init__(
            self,
            args,
            qformer,
            query,
            json_directory=None
    ):
        super(MyArch, self).__init__()
        self.device = args.device
        self.args = args
        self.json_directory = json_directory
        self.act = nn.ReLU()
        
        self.generator_model = AutoModelForCausalLM.from_pretrained(args.LLM, torch_dtype=torch.bfloat16)
        if not self.args.LLM_freeze:
            lora_config = LoraConfig(
                TaskType.CAUSAL_LM,
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                )

            self.lora_model = get_peft_model(self.generator_model, lora_config)
            self.generator_mode = self.lora_model.base_model.model

        if self.args.audio_type == 'wav2vec2':
            self.audio_feature_dim = 768
        elif self.args.audio_type == 'wavlm':
            self.audio_feature_dim = 1024  # check!!
        elif self.args.audio_type == 'spectrogram':
            self.audio_feature_dim = 128
        self.video_feature_dim = 512
            
        self.tokenizer = AutoTokenizer.from_pretrained(args.LLM)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # https://stackoverflow.com/questions/76446228/setting-padding-token-as-eos-token-when-using-datacollatorforlanguagemodeling-fr
        self.tokenizer.padding_side='right'
        self.tokenizer.truncation_side='left'
        
        self.vocab_size = self.generator_model.config.vocab_size  # 32000
        self.hidden_size = self.generator_model.config.hidden_size  # 4096
        
        self.video_q_former = q_former(self.video_feature_dim, self.hidden_size, qformer, query)
        self.audio_q_former = q_former(self.audio_feature_dim, self.hidden_size, qformer, query)
        
        
        self.MAE_loss_function = nn.L1Loss()
        self.MSE_loss_function = nn.MSELoss()
        self.CE_loss_function = nn.CrossEntropyLoss()  # tokenizer.eos_token = '</s>' = tokenizer.decode(2)

    def encode_hist(self, history, mask, type):
        max_idx = int(torch.max(mask).item())
        min_idx = int(min(mask[mask != 0]))  # except '0'
        if (max_idx - min_idx) >= self.args.max_history:
            min_idx = max_idx - self.args.max_history + 1
        output_hist = []
        for idx in range(min_idx, max_idx + 1):
            utt_idx = torch.where(mask==idx)[0].to(self.device)
            utt = history[utt_idx]
            if type == 'video':
                output_hist.append(self.video_q_former(utt))
            elif type == 'audio':
                output_hist.append(self.audio_q_former(utt))
            
        return output_hist
    
    def text_format_matching(self, text):
        text = [list(a) for a in text]
        text = [[row[i] for row in text] for i in range(len(text[0]))]
        return text

    def encode_text(self, unbatched_text_history):
        text_history = []
        for text in unbatched_text_history:
            if text != "":
                logit = self.tokenize(text)['input_ids'][0]
                text_history.append(self.generator_model.model.embed_tokens(logit))
    
        return text_history
        
    def instruction_prompt_generate(self, prompt):
        prompt_logit = self.tokenize(prompt)['input_ids'][0]
        prompt = self.generator_model.model.embed_tokens(prompt_logit)
        return prompt
    
    def build_modality_input_for_llm(self, inputs_for_llm, history_prompt, video_hist_list, audio_hist_list, text_history_list, j):
        if ('video' in self.args.modal) and ('audio' in self.args.modal):
            return torch.cat([inputs_for_llm, history_prompt, video_hist_list[j], audio_hist_list[j], text_history_list[j]], dim=0)
        elif 'video' in self.args.modal:
            return torch.cat([inputs_for_llm, history_prompt, video_hist_list[j], text_history_list[j]], dim=0)
        elif 'audio' in self.args.modal:
            return torch.cat([inputs_for_llm, history_prompt, audio_hist_list[j], text_history_list[j]], dim=0)
        else:
            return torch.cat([inputs_for_llm, history_prompt, text_history_list[j]], dim=0)
        

    def prepare_inputs_for_llm(self, a_historys, a_masks, v_historys, v_masks, t_history, spk_historys, R, target_text=None):
        instruction_prompt = self.instruction_prompt_generate(prompt="###Instruction: Generate a following response of this conversation\n") # instruction prompt
        
        bs= a_historys.shape[0]

        batched_inputs_for_llm = []
        masks = []
        
        for i in range(bs):
            if 'video' in self.args.modal:
                video_hist_list = self.encode_hist(v_historys[i], v_masks[i], type="video")
            else:
                video_hist_list = None
            
            if 'audio' in self.args.modal:
                audio_hist_list = self.encode_hist(a_historys[i], a_masks[i], type="audio")
            else:
                audio_hist_list = None

            text_history_list = self.encode_text(t_history[i])
            
            spk_history_list = spk_historys[i]
            inputs_for_llm = instruction_prompt.to(self.device)
            for j in range(len(text_history_list)):
                history_prompt = self.instruction_prompt_generate(prompt=f"###Speaker_{spk_history_list[j]}:")
                inputs_for_llm = self.build_modality_input_for_llm(inputs_for_llm, history_prompt, video_hist_list, audio_hist_list, text_history_list, j)
                
            if target_text == None:
                response_prompt = self.instruction_prompt_generate(prompt=f"{R}, Speaker_{spk_history_list[-1]}:")
            else:
                response_prompt = self.instruction_prompt_generate(prompt=f"{R}, Speaker_{spk_history_list[-1]}:{target_text[i]}/")
                
            inputs_for_llm = torch.cat([inputs_for_llm, response_prompt], dim=0)
            
            mask = inputs_for_llm.sum(dim=-1) != -9999

            batched_inputs_for_llm.append(inputs_for_llm)
            masks.append(mask)
            
        return batched_inputs_for_llm, masks
        
    
    def padding_for_llm(self, batched_inputs_for_llm, masks):
        max_len = self.args.max_history      
        
        for batch in range(len(batched_inputs_for_llm)):
            seq_len, _ = batched_inputs_for_llm[batch].shape
            if max_len > seq_len:
                batched_inputs_for_llm[batch] = torch.cat([batched_inputs_for_llm[batch], torch.zeros(max_len - seq_len, self.hidden_size).to(self.device)], dim=0) # padding
                masks[batch] = torch.cat([masks[batch]] + [torch.zeros(1, dtype=torch.bool).to(self.device)]*(max_len-seq_len), dim=0) # make mask for llm input
            else:
                batched_inputs_for_llm[batch] = torch.cat([batched_inputs_for_llm[batch][:max_len].to(self.device)], dim=0) # truncation
                masks[batch] = masks[batch][:max_len] # make mask for llm input
            
        batched_inputs_for_llm = torch.stack(batched_inputs_for_llm, dim=0)
        masks = torch.stack(masks, dim=0)
        
        return batched_inputs_for_llm, masks


        
    def audio_label_construct(self,inputs_for_llm, mask, audio_label, audio_label_mask):
        inputs_for_llm = torch.cat([inputs_for_llm, audio_label], dim=1)
        
        m = torch.zeros(audio_label.shape).to(self.device)
        mask_idx = (audio_label != m).sum(dim=-1)  # audio_label has padded with 0
        audio_label_mask = mask_idx > 0
        mask = torch.cat([mask, audio_label_mask], dim=1)
        
        return inputs_for_llm, mask

    def tokenize(self, input):
        text_tokens = self.tokenizer(input, return_tensors='pt')
        
        return text_tokens.to(self.device)
    
    def tokenize_with_eos(self, inputs):
        input_eos = [t + self.tokenizer.eos_token for t in inputs]
        
        text_tokens = self.tokenizer(input_eos,
                                    padding='max_length',
                                    max_length=self.args.max_length,
                                    truncation=True,
                                    return_attention_mask=True,
                                    return_tensors='pt')
        
        return text_tokens.to(self.device)
    
    def eos_expand(self, bs, num_eos):
        eos_id = torch.tensor(self.tokenizer.eos_token_id).to(self.device)
        eos_emb = self.generator_model.model.embed_tokens(eos_id)
        
        return eos_emb.expand(bs, num_eos, eos_emb.shape[0])


    def input_encoding(self, inputs, R, target_text=None):
        '''
        inputs[0]: text_history  (string)
            # shape = [batch, num_history]
        inputs[1]: audio_history, padding_mask  (dictionary)
            # value shape = [batch, history_pad_size, audio_pad_size]
        inputs[2]: video_history, padding_mask  (dictionary)
            # value shape = [batch, history_pad_size, video_pad_size]
        '''
        text_history = inputs[0]  # string
        audio_history = inputs[1]['feature']  # torch.tensor
        audio_history_mask = inputs[1]['mask']  # torch.tensor
        video_history = inputs[2]['feature']  # torch.tensor
        video_history_mask = inputs[2]['mask']  # torch.tensor
        speaker_history = inputs[3]  # string
        
        # print(text_history)
        if len(audio_history.shape) < 3:  # bs = 1
            text_history = [text_history]
            audio_history = torch.unsqueeze(audio_history, dim=0)
            audio_history_mask = torch.unsqueeze(audio_history_mask, dim=0)
            video_history = torch.unsqueeze(video_history, dim=0)
            video_history_mask = torch.unsqueeze(video_history_mask, dim=0)
            speaker_history = [speaker_history]
        else:
            text_history = [list(filter(lambda x: x != '', sublist)) + [''] * sublist.count('') for sublist in text_history]
            speaker_history = [list(filter(lambda x: x != '', sublist)) + [''] * sublist.count('') for sublist in speaker_history]
        
        model_inputs, mask_inputs = self.prepare_inputs_for_llm(audio_history, audio_history_mask,
                                                                video_history, video_history_mask,
                                                                text_history,
                                                                speaker_history,
                                                                R=R,
                                                                target_text=target_text)
        
        return model_inputs, mask_inputs, text_history

        
    def text_label_construct(self,inputs_for_llm, mask, text_label):
        text_label_tokens = self.generator_model.model.embed_tokens(text_label['input_ids'])
        inputs_for_llm = torch.cat([inputs_for_llm, text_label_tokens], dim=1)
        
        text_label_mask = text_label['attention_mask']
        mask = torch.cat([mask, text_label_mask], dim=1)
        return inputs_for_llm, mask

        
    def get_text_loss(self, pad_model_inputs, pad_mask_inputs, text_label):
        text_label_tokens = self.tokenize_with_eos(text_label)
        target_model_inputs, target_mask_inputs = self.text_label_construct(pad_model_inputs, pad_mask_inputs, text_label_tokens)
        
        # generate with pretrained LLM
        model_output = self.generator_model(inputs_embeds=target_model_inputs, attention_mask=target_mask_inputs, output_hidden_states=True)

        # text generation loss
        output = model_output.logits[:, -self.args.max_length-1:-1].contiguous().view(-1, self.vocab_size)
        label = text_label_tokens['input_ids'][:, :].contiguous().view(-1)
        text_loss = self.CE_loss_function(output, label)

        
        return text_loss

    def forward(self, inputs, labels, metric_log=False, epoch=None, mode='train'):
        '''
        metric_log: whether calculate metric or not
        '''
        text_label = labels[0]  # string
        label_audio = labels[1]  # string
        voice_description = labels[2]  # string
        
        if 'description' in self.args.target:
            model_inputs, mask_inputs, text_history = self.input_encoding(inputs, R="\n###Generate a response format as [text of response (voice description)]")
        else:
            model_inputs, mask_inputs, text_history = self.input_encoding(inputs, R="\n###Generate a text response ")
        
        pad_model_inputs, pad_mask_inputs = self.padding_for_llm(model_inputs, mask_inputs)
        
        # Step4. generate
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16, enabled=True):
            if 'text' in self.args.target:
                if 'description' in self.args.target: 
                    text_label = [x + y for x, y in zip(text_label, voice_description)]  # add description
                
                text_loss = self.get_text_loss(pad_model_inputs, 
                                                pad_mask_inputs, 
                                                text_label,
                                                # voice_description,
                                                )
                if metric_log == True:
                    logit = self.generator_model.generate(inputs_embeds=pad_model_inputs,  # [batch, _, _]
                                                        attention_mask=pad_mask_inputs,
                                                        pad_token_id=self.tokenizer.eos_token_id,
                                                        do_sample=True,
                                                        top_p=0.8, top_k=30, temperature=0.8,
                                                        max_new_tokens=self.args.max_length,
                                                        )
                    outputs_sentence = self.tokenizer.batch_decode(logit, skip_special_tokens=True)
                    
                    self.save_output(outputs_sentence, text_label, epoch, mode)
            else:
                text_loss = torch.tensor([0.0], requires_grad=True).to(self.device)
                
            
        return text_loss, 0
        
        
    def predict(self, inputs, labels=None, d_name=None, ckpt_epoch=3):
        '''
        input: list of inputs
        '''
        if labels is not None:
            text_label = labels[0]  # string
            label_audio = labels[1]  # string
            voice_description = labels[2]  # string
            
        if 'description' in self.args.target:
            model_inputs, mask_inputs, text_history = self.input_encoding(inputs, R="\n###Generate a response format as [text of response (voice description)]")
        else:
            model_inputs, mask_inputs, text_history = self.input_encoding(inputs, R="\n###Generate a text response ")
        
        pad_model_inputs, pad_mask_inputs = self.padding_for_llm(model_inputs, mask_inputs)
        
        # Step4. generate
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16, enabled=True):
            if 'text' in self.args.target: 
                if 'description' in self.args.target: 
                    text_label = [x + y for x, y in zip(text_label, voice_description)]
                logit = self.generator_model.generate(inputs_embeds=pad_model_inputs,
                                                             attention_mask=pad_mask_inputs,
                                                            pad_token_id=self.tokenizer.eos_token_id,
                                                            do_sample=True,
                                                            top_p=0.8, top_k=30, temperature=0.8,
                                                            max_new_tokens=self.args.max_length,
                                                            )
                
                output = self.tokenizer.batch_decode(logit, skip_special_tokens=True)
                
                self.save_sample(output, text_label, text_history, d_name)
                self.save_output(output, text_label, ckpt_epoch, mode='test')

        return output
        
    def save_output(self, output_sentence, ref_sentence, epoch, mode):
        json_name = f'epoch{epoch}_{mode}.json'
        # 1. load data
        try:
            with open(self.json_directory + json_name, 'r') as json_file:
                existing_data = json.load(json_file)
        except FileNotFoundError:
            existing_data = []

        # 2. add new data
        new_entry = {
            'output_sentence': output_sentence,
            'ref_sentence': ref_sentence
        }

        existing_data.append(new_entry)

        # 3. saver updated data
        with open(self.json_directory + json_name, 'w') as json_file:
            json.dump(existing_data, json_file, indent=4)

    def save_sample(self, output_sentence, ref_sentence, text_history, d_name):
        json_name = f'predict_output.json'
        
        for i in range(len(output_sentence)):
            # 1. load data
            try:
                with open(self.json_directory + json_name, 'r') as json_file:
                    existing_data = json.load(json_file)
            except FileNotFoundError:
                existing_data = []

            # 2. add new data
            new_entry = {
                'data_name': d_name[i],
                'history': text_history[i],
                'output_sentence': output_sentence[i],
                'ref_sentence': ref_sentence[i]
            }

            existing_data.append(new_entry)

            # 3. saver updated data
            with open(self.json_directory + json_name, 'w') as json_file:
                json.dump(existing_data, json_file, indent=4)
