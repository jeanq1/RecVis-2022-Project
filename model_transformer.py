import os
import time
import copy
import pickle
import json
from math import ceil
from pathlib import Path
import datetime
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import math

from utils import Bar
from utils.viz import viz_results_paper
from utils.averagemeter import AverageMeter
from utils.utils import torch_to_list, get_num_signs
from eval import Metric

from kornia.losses import focal


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def exponential_descrease(idx_decoder, p=3):
    return math.exp(-p*idx_decoder)

class AttentionHelper(nn.Module):
    def __init__(self):
        super(AttentionHelper, self).__init__()
        self.softmax = nn.Softmax(dim=-1)


    def scalar_dot_att(self, proj_query, proj_key, proj_val, padding_mask):
        '''
        scalar dot attention.
        :param proj_query: shape of (B, C, L) => (Batch_Size, Feature_Dimension, Length)
        :param proj_key: shape of (B, C, L)
        :param proj_val: shape of (B, C, L)
        :param padding_mask: shape of (B, C, L)
        :return: attention value of shape (B, C, L)
        '''
        m, c1, l1 = proj_query.shape
        m, c2, l2 = proj_key.shape
        
        assert c1 == c2
        
        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)  # out of shape (B, L1, L2)
        attention = energy / np.sqrt(c1)
        attention = attention + torch.log(padding_mask + 1e-6) # mask the zero paddings. log(1e-6) for zero paddings
        attention = self.softmax(attention) 
        attention = attention * padding_mask
        attention = attention.permute(0,2,1)
        out = torch.bmm(proj_val, attention)
        return out, attention

class AttLayer(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type): # r1 = r2
        super(AttLayer, self).__init__()
        
        self.query_conv = nn.Conv1d(in_channels=q_dim, out_channels=q_dim // r1, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=k_dim, out_channels=k_dim // r2, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=v_dim, out_channels=v_dim // r3, kernel_size=1)
        
        self.conv_out = nn.Conv1d(in_channels=v_dim // r3, out_channels=v_dim, kernel_size=1)

        self.bl = bl
        self.stage = stage
        self.att_type = att_type
        assert self.att_type in ['normal_att', 'block_att', 'sliding_att']
        assert self.stage in ['encoder','decoder']
        
        self.att_helper = AttentionHelper()
        self.window_mask = self.construct_window_mask()
        
    
    def construct_window_mask(self):
        '''
            construct window mask of shape (1, l, l + l//2 + l//2), used for sliding window self attention
        '''
        window_mask = torch.zeros((1, self.bl, self.bl + 2* (self.bl //2)))
        for i in range(self.bl):
            window_mask[:, :, i:i+self.bl] = 1
        return window_mask.to(device)
    
    def forward(self, x1, x2, mask):
        # x1 from the encoder
        # x2 from the decoder
        
        query = self.query_conv(x1)
        key = self.key_conv(x1)
         
        if self.stage == 'decoder':
            assert x2 is not None
            value = self.value_conv(x2)
        else:
            value = self.value_conv(x1)
            
        if self.att_type == 'normal_att':
            return self._normal_self_att(query, key, value, mask)
        elif self.att_type == 'block_att':
            return self._block_wise_self_att(query, key, value, mask)
        elif self.att_type == 'sliding_att':
            return self._sliding_window_self_att(query, key, value, mask)

    
    def _normal_self_att(self,q,k,v, mask):
        m_batchsize, c1, L = q.size()
        _,c2,L = k.size()
        _,c3,L = v.size()
        padding_mask = torch.ones((m_batchsize, 1, L)).to(device) * mask[:,0:1,:]
        output, attentions = self.att_helper.scalar_dot_att(q, k, v, padding_mask)
        output = self.conv_out(F.relu(output))
        output = output[:, :, 0:L]
        return output * mask[:, 0:1, :]  
        
    def _block_wise_self_att(self, q,k,v, mask):
        m_batchsize, c1, L = q.size()
        _,c2,L = k.size()
        _,c3,L = v.size()
        
        nb = L // self.bl
        if L % self.bl != 0:
            q = torch.cat([q, torch.zeros((m_batchsize, c1, self.bl - L % self.bl)).to(device)], dim=-1)
            k = torch.cat([k, torch.zeros((m_batchsize, c2, self.bl - L % self.bl)).to(device)], dim=-1)
            v = torch.cat([v, torch.zeros((m_batchsize, c3, self.bl - L % self.bl)).to(device)], dim=-1)
            nb += 1

        padding_mask = torch.cat([torch.ones((m_batchsize, 1, L)).to(device) * mask[:,0:1,:], torch.zeros((m_batchsize, 1, self.bl * nb - L)).to(device)],dim=-1)

        q = q.reshape(m_batchsize, c1, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb, c1, self.bl)
        padding_mask = padding_mask.reshape(m_batchsize, 1, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb,1, self.bl)
        k = k.reshape(m_batchsize, c2, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb, c2, self.bl)
        v = v.reshape(m_batchsize, c3, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb, c3, self.bl)
        
        output, attentions = self.att_helper.scalar_dot_att(q, k, v, padding_mask)
        output = self.conv_out(F.relu(output))
        
        output = output.reshape(m_batchsize, nb, c3, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize, c3, nb * self.bl)
        output = output[:, :, 0:L]
        return output * mask[:, 0:1, :]  
    
    def _sliding_window_self_att(self, q,k,v, mask):
        m_batchsize, c1, L = q.size()
        _, c2, _ = k.size()
        _, c3, _ = v.size()
        
        
        assert m_batchsize == 1  # currently, we only accept input with batch size 1
        # padding zeros for the last segment
        nb = L // self.bl 
        if L % self.bl != 0:
            q = torch.cat([q, torch.zeros((m_batchsize, c1, self.bl - L % self.bl)).to(device)], dim=-1)
            k = torch.cat([k, torch.zeros((m_batchsize, c2, self.bl - L % self.bl)).to(device)], dim=-1)
            v = torch.cat([v, torch.zeros((m_batchsize, c3, self.bl - L % self.bl)).to(device)], dim=-1)
            nb += 1
        padding_mask = torch.cat([torch.ones((m_batchsize, 1, L)).to(device) * mask[:,0:1,:], torch.zeros((m_batchsize, 1, self.bl * nb - L)).to(device)],dim=-1)
        
        # sliding window approach, by splitting query_proj and key_proj into shape (c1, l) x (c1, 2l)
        # sliding window for query_proj: reshape
        q = q.reshape(m_batchsize, c1, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb, c1, self.bl)
        
        # sliding window approach for key_proj
        # 1. add paddings at the start and end
        k = torch.cat([torch.zeros(m_batchsize, c2, self.bl // 2).to(device), k, torch.zeros(m_batchsize, c2, self.bl // 2).to(device)], dim=-1)
        v = torch.cat([torch.zeros(m_batchsize, c3, self.bl // 2).to(device), v, torch.zeros(m_batchsize, c3, self.bl // 2).to(device)], dim=-1)
        padding_mask = torch.cat([torch.zeros(m_batchsize, 1, self.bl // 2).to(device), padding_mask, torch.zeros(m_batchsize, 1, self.bl // 2).to(device)], dim=-1)
        
        # 2. reshape key_proj of shape (m_batchsize*nb, c1, 2*self.bl)
        k = torch.cat([k[:,:, i*self.bl:(i+1)*self.bl+(self.bl//2)*2] for i in range(nb)], dim=0) # special case when self.bl = 1
        v = torch.cat([v[:,:, i*self.bl:(i+1)*self.bl+(self.bl//2)*2] for i in range(nb)], dim=0) 
        # 3. construct window mask of shape (1, l, 2l), and use it to generate final mask
        padding_mask = torch.cat([padding_mask[:,:, i*self.bl:(i+1)*self.bl+(self.bl//2)*2] for i in range(nb)], dim=0) # of shape (m*nb, 1, 2l)
        final_mask = self.window_mask.repeat(m_batchsize * nb, 1, 1) * padding_mask 
        
        output, attention = self.att_helper.scalar_dot_att(q, k, v, final_mask)
        output = self.conv_out(F.relu(output))

        output = output.reshape(m_batchsize, nb, -1, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize, -1, nb * self.bl)
        output = output[:, :, 0:L]
        return output * mask[:, 0:1, :]


class MultiHeadAttLayer(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type, num_head):
        super(MultiHeadAttLayer, self).__init__()
#         assert v_dim % num_head == 0
        self.conv_out = nn.Conv1d(v_dim * num_head, v_dim, 1)
        self.layers = nn.ModuleList(
            [copy.deepcopy(AttLayer(q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type)) for i in range(num_head)])
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x1, x2, mask):
        out = torch.cat([layer(x1, x2, mask) for layer in self.layers], dim=1)
        out = self.conv_out(self.dropout(out))
        return out
            

class ConvFeedForward(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(ConvFeedForward, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)


class FCFeedForward(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FCFeedForward, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),  # conv1d equals fc
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(out_channels, out_channels, 1)
        )
        
    def forward(self, x):
        return self.layer(x)
    

class AttModule(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, r1, r2, att_type, stage, alpha):
        super(AttModule, self).__init__()
        self.feed_forward = ConvFeedForward(dilation, in_channels, out_channels)
        self.instance_norm = nn.InstanceNorm1d(in_channels, track_running_stats=False)
        self.att_layer = AttLayer(in_channels, in_channels, out_channels, r1, r1, r2, dilation, att_type=att_type, stage=stage) # dilation
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()
        self.alpha = alpha
        
    def forward(self, x, f, mask):
        out = self.feed_forward(x)
        out = self.alpha * self.att_layer(self.instance_norm(out), f, mask) + out
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, max_len=10000):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).permute(0,2,1) # of shape (1, d_model, l)
        self.pe = nn.Parameter(pe, requires_grad=True)
#         self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :, 0:x.shape[2]]

class Encoder(nn.Module):
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, att_type, alpha):
        super(Encoder, self).__init__()
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1) # fc layer
        self.layers = nn.ModuleList(
            [AttModule(2 ** i, num_f_maps, num_f_maps, r1, r2, att_type, 'encoder', alpha) for i in # 2**i faire attention à changer l'attention ici 
             range(num_layers)])
        
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.dropout = nn.Dropout2d(p=channel_masking_rate)
        self.channel_masking_rate = channel_masking_rate

    def forward(self, x, mask):
        '''
        :param x: (N, C, L)
        :param mask:
        :return:
        '''

        if self.channel_masking_rate > 0:
            x = x.unsqueeze(2)
            x = self.dropout(x)
            x = x.squeeze(2)

        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, None, mask)
        
        out = self.conv_out(feature) * mask[:, 0:1, :]

        return out, feature


class Decoder(nn.Module):
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, att_type, alpha):
        super(Decoder, self).__init__()#         self.position_en = PositionalEncoding(d_model=num_f_maps)
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [AttModule(2 ** i, num_f_maps, num_f_maps, r1, r2, att_type, 'decoder', alpha) for i in # 2 ** i
             range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, fencoder, mask):

        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, fencoder, mask)

        out = self.conv_out(feature) * mask[:, 0:1, :]

        return out, feature
    
class MyTransformer(nn.Module):
    def __init__(self, num_decoders, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate):
        super(MyTransformer, self).__init__()
        self.encoder = Encoder(num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, att_type='normal_att', alpha=1)
        self.decoders = nn.ModuleList([copy.deepcopy(Decoder(num_layers, r1, r2, num_f_maps, num_classes, num_classes, att_type='normal_att', alpha=exponential_descrease(s))) for s in range(num_decoders)]) # num_decoders
        
        
    def forward(self, x, mask):
        out, feature = self.encoder(x, mask)
        outputs = out.unsqueeze(0)
        
        for decoder in self.decoders:
            out, feature = decoder(F.softmax(out, dim=1) * mask[:, 0:1, :], feature* mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
 
        return outputs



class Trainer:
    def __init__(self, num_blocks, num_layers, num_f_maps, dim, num_classes, device, weights, save_dir):
        
        
        #self.model = MultiStageModel(num_blocks, num_layers, num_f_maps, dim, num_classes)
        self.model = MyTransformer(3, 10, 2, 2, 32, 1024, 2, 0.3)

        #self.model = make_model(1024, 2 )

        #import pdb; pdb.set_trace()
        self.floss = focal.FocalLoss(alpha= torch.tensor(weights).to(device) , gamma=2.0)

        if weights is None:
            self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        else:
            self.weights = torch.tensor(weights).to(device)
            self.ce = nn.CrossEntropyLoss(weight=torch.tensor(weights).to(device), ignore_index=-100)

        self.mse = nn.MSELoss(reduction='none')
        self.mse_red = nn.MSELoss(reduction='mean')
        self.sm = nn.Softmax(dim=1)
        self.num_classes = num_classes
        self.writer = SummaryWriter(log_dir=f'{save_dir}/logs')
        self.global_counter = 0
        self.train_result_dict = {}
        self.test_result_dict = {}

    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, device, eval_args, pretrained=''):
        self.model.train()
        self.model.to(device)

        # load pretrained model
        if pretrained != '':
            pretrained_dict = torch.load(pretrained)
            self.model.load_state_dict(pretrained_dict)

        criterion = self.ce

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)

        for epoch in range(num_epochs):
            epoch_loss = 0
            end = time.time()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            bar = Bar("E%d" % (epoch + 1), max=batch_gen.get_max_index())
            count = 0
            get_metrics_train = Metric('train')


            while batch_gen.has_next():
                self.global_counter += 1
                batch_input, batch_target, batch_target_eval, mask = batch_gen.next_batch(batch_size)
                batch_input, batch_target, batch_target_eval, mask = batch_input.to(device), batch_target.to(device), batch_target_eval.to(device), mask.to(device)
                optimizer.zero_grad()
                
                predictions = self.model(batch_input, mask)
                                
                loss = 0
                # loss for each stage
                for ix, p in enumerate(predictions):
                    if self.num_classes == 1:
                        loss += self.mse_red(p.transpose(2, 1).contiguous().view(-1, self.num_classes).squeeze(), batch_target.view(-1))
                    else:
                        ## original loss
                        lambdaa = 0.15
                        to = 5.0
                        loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                        loss += lambdaa*torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=to**2)*mask[:, :, 1:])
                        
                        
                        ##focal loss -> true version
                        #loss += torch.mean(
                        #                    self.floss(
                         #                               p.transpose(2, 1).contiguous().view(-1, self.num_classes),
                        #                                (batch_target*(mask[:, 0:1, :].squeeze())).view(-1).long()
                         #                               ) 

                         #                    *  (mask[:, 0:1, :].squeeze().reshape(-1)))  

                        
                        
                        ## only cross entropy
                        #loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))

                        ##Kulback lieber
                        #klloss = torch.nn.KLDivLoss(log_target=True)
                        #loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                        #loss += 0.15*torch.mean(
                         #                       klloss( 
                         #                                   F.log_softmax(p[:, :, 1:], dim=1),
                         #                                   F.log_softmax(p[:, :, :-1], dim=1)
                         #                              )
                         #                       )

                        
                        
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                if self.num_classes == 1:
                    predicted = torch.round(predictions[-1].data.squeeze())
                    gt = torch.round(batch_target)
                    gt_eval = batch_target_eval

                else:
                    _, predicted = torch.max(predictions[-1].data, 1)
                    gt = batch_target
                    gt_eval = batch_target_eval

                get_metrics_train.calc_scores_per_batch(predicted, gt, gt_eval, mask)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # plot progress
                bar.suffix = "({batch}/{size}) Batch: {bt:.1f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:}".format(
                    batch=count + 1,
                    size=batch_gen.get_max_index() / batch_size,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=datetime.timedelta(seconds=ceil((bar.eta_td/batch_size).total_seconds())),
                    loss=loss.item()
                )
                count += 1
                bar.next()

            batch_gen.reset()
            torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
            torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")

            get_metrics_train.calc_metrics()
            result_dict = get_metrics_train.save_print_metrics(self.writer, save_dir, epoch, epoch_loss/(len(batch_gen.list_of_examples)/batch_size))
            self.train_result_dict.update(result_dict)

            eval_args[7] = epoch
            eval_args[1] = save_dir + "/epoch-" + str(epoch+1) + ".model"
            self.predict(*eval_args)

        with open(f'{save_dir}/train_results.json', 'w') as fp:
            json.dump(self.train_result_dict, fp, indent=4)
        with open(f'{save_dir}/eval_results.json', 'w') as fp:
            json.dump(self.test_result_dict, fp, indent=4)
        self.writer.close()


    def predict(
            self,
            args,
            model_dir,
            results_dir,
            features_dict,
            gt_dict,
            gt_dict_dil,
            vid_list_file,
            epoch,
            device,
            mode,
            classification_threshold,
            uniform=0,
            save_pslabels=False,
            CP_dict=None,
            ):

        save_score_dict = {}
        metrics_per_signer = {}
        get_metrics_test = Metric(mode)

        self.model.eval()
        with torch.no_grad():
            
            if CP_dict is None:
                self.model.to(device)
                self.model.load_state_dict(torch.load(model_dir))

            epoch_loss = 0
            for vid in tqdm(vid_list_file):
                features = np.swapaxes(features_dict[vid], 0, 1)
                if CP_dict is not None:
                    predicted = torch.tensor(CP_dict[vid]).to(device)
                    pred_prob = CP_dict[vid]
                    gt = torch.tensor(gt_dict[vid]).to(device)
                    gt_eval = torch.tensor(gt_dict_dil[vid]).to(device)
                else:
                    input_x = torch.tensor(features, dtype=torch.float)
                    input_x.unsqueeze_(0)
                    input_x = input_x.to(device)
                    predictions = self.model(input_x, torch.ones(input_x.size(), device=device))
                    if self.num_classes == 1:
                        # regression
                        num_iter = 1
                        pred_prob = predictions[-1].squeeze()
                        pred_prob = torch_to_list(pred_prob)
                        predicted = torch.tensor(np.where(np.asarray(pred_prob) > args.classification_threshold, 1, 0)).to(device)
                        
                        gt = torch.tensor(gt_dict[vid]).to(device)
                        gt_eval = torch.tensor(gt_dict_dil[vid]).to(device)

                    else:
                        num_iter = 1
                        pred_prob = torch_to_list(self.sm(predictions[-1]))[0][1]
                        predicted = torch.tensor(np.where(np.asarray(pred_prob) > args.classification_threshold, 1, 0)).to(device)
                        gt = torch.tensor(gt_dict[vid]).to(device)
                        gt_eval = torch.tensor(gt_dict_dil[vid]).to(device)

                    if uniform:
                        num_signs = get_num_signs(gt_dict[vid])
                        len_clip = len(gt_dict[vid])
                        predicted = [0]*len_clip
                        dist_uni = len_clip / num_signs
                        for i in range(1, num_signs):
                            predicted[round(i*dist_uni)] = 1
                            predicted[round(i*dist_uni)+1] = 1
                        pred_prob = predicted
                        predicted = torch.tensor(predicted).to(device)

                    if save_pslabels:
                        save_score_dict[vid] = {}
                        save_score_dict[vid]['scores'] = np.asarray(pred_prob)
                        save_score_dict[vid]['preds'] = np.asarray(torch_to_list(predicted))
                        continue
                
                loss = 0
                mask = torch.ones(self.num_classes, np.shape(gt)[0]).to(device)
                # loss for each stage
                for ix, p in enumerate(predictions):
                    if self.num_classes == 1:
                        loss += self.mse_red(p.transpose(2, 1).contiguous().view(-1, self.num_classes).squeeze(), gt.view(-1))
                    else:
                        loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), gt.view(-1))
                        loss += 0.15*torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16)*mask[:, 1:])

                epoch_loss += loss.item()


                cut_endpoints = True
                if cut_endpoints:
                    if sum(predicted[-2:]) > 0 and sum(gt_eval[-4:]) == 0:
                        for j in range(len(predicted)-1, 0, -1):
                            if predicted[j] != 0:
                                predicted[j] = 0
                            elif predicted[j] == 0 and j < len(predicted) - 2:
                                break

                    if sum(predicted[:2]) > 0 and sum(gt_eval[:4]) == 0:
                        check = 0
                        for j, item in enumerate(predicted):
                            if item != 0:
                                predicted[j] = 0
                                check = 1
                            elif item == 0 and (j > 2 or check):
                                break

                get_metrics_test.calc_scores_per_batch(predicted.unsqueeze(0), gt.unsqueeze(0), gt_eval.unsqueeze(0))
                
                save_score_dict[vid] = {}
                save_score_dict[vid]['scores'] = np.asarray(pred_prob)
                save_score_dict[vid]['gt'] = torch_to_list(gt)

                if mode == 'test' and args.viz_results:
                    if not isinstance(vid, int):
                        f_name = vid.split('/')[-1].split('.')[0]
                    else:
                        f_name = str(vid)

                    viz_results_paper(
                        gt,
                        torch_to_list(predicted),
                        name=results_dir + "/" + f'{f_name}',
                        pred_prob=pred_prob,
                    )
            
            if save_pslabels:
                PL_labels_dict = {}
                PL_scores_dict = {}
                for vid in vid_list_file:
                    if args.test_data == 'phoenix14':
                        episode = vid.split('.')[0]
                        part = vid.split('.')[1]
                    elif args.test_data == 'bsl1k':
                        episode = vid.split('_')[0]
                        part = vid.split('_')[1]

                    if episode not in PL_labels_dict:
                        PL_labels_dict[episode] = []
                        PL_scores_dict[episode] = []

                    PL_labels_dict[episode].extend(save_score_dict[vid]['preds'])
                    PL_scores_dict[episode].extend(save_score_dict[vid]['scores'])

                for episode in PL_labels_dict.keys():
                    PL_root = str(Path(results_dir).parent).replace(f'exps/results/regression', 'data/pseudo_labels/PL').replace(f'exps/results/classification', f'data/pseudo_labels/PL')
                    # print(f'Save PL to {PL_root}/{episode}')
                    if not os.path.exists(f'{PL_root}/{episode}'):
                        os.makedirs(f'{PL_root}/{episode}')
                        pickle.dump(PL_labels_dict[episode], open(f'{PL_root}/{episode}/preds.pkl', "wb"))
                        pickle.dump(PL_scores_dict[episode], open(f'{PL_root}/{episode}/scores.pkl', "wb"))
                    else:
                        print('PL already exist!!')
                return

            if mode == 'test':
                pickle.dump(save_score_dict, open(f'{results_dir}/scores.pkl', "wb"))

            get_metrics_test.calc_metrics()
            save_dir = results_dir if mode == 'test' else Path(model_dir).parent
            result_dict = get_metrics_test.save_print_metrics(self.writer, save_dir, epoch, epoch_loss/len(vid_list_file))
            self.test_result_dict.update(result_dict)
        
        if mode == 'test':
            with open(f'{results_dir}/eval_results.json', 'w') as fp:
                json.dump(self.test_result_dict, fp, indent=4)