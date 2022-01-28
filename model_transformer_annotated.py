# -*- coding: utf-8 -*-
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##annotated transformer begin
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        #encoded = self.encode(src, src_mask)
        #print("Encoding successful !")
        #decoded = self.decode(self.encode(src, src_mask), src_mask,
                         #   tgt, tgt_mask)
        #print("Decoding successful !")
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        #import pdb; pdb.set_trace()
        return self.encoder(self.src_embed(src), src_mask) #on encode l'embedding de src avec le mask (virer l'embedding)
    
    def decode(self, memory, src_mask, tgt, tgt_mask): #memory c'est la sortie de l'encoder, avec src mask son mask et tgt c'est l'output voulu
        #manually replacing -100 by 2
        
        
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask) #on fait passer l'input et le mask dans chaque layer
        return self.norm(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
                

        return x + self.dropout(sublayer(self.norm(x)))

    
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."

        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask): # x c'est l'output voulu
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        #import pdb; pdb.set_trace()
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask)) #attention sur l'output en entrée du decoder, x est la query, la key et la value ? 
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask)) #attention sur memory, l'output de l'encoder, x est la query, et m la key et la value ? 
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    #import pdb;pdb.set_trace()
    
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    p_attn = F.softmax(scores, dim = -1)
    
    if dropout is not None:
        p_attn = dropout(p_attn)
        
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model, padding_idx=2)
        self.d_model = d_model

    def forward(self, x):
        lut = self.lut(x)
        return  lut * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)


def make_model(src_vocab, tgt_vocab, N=6, 
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        encoder = Encoder(layer = EncoderLayer(size = d_model,
                                               self_attn = c(attn),
                                               feed_forward = c(ff),
                                               dropout = dropout),
                          N = N),
        
        decoder = Decoder(layer = DecoderLayer(size = d_model,
                                               self_attn = c(attn),
                                               src_attn = c(attn),
                                               feed_forward = c(ff),
                                               dropout = dropout),
                          N = N),
        
        src_embed = nn.Sequential(#Embeddings(d_model = d_model,
                                     #        vocab = src_vocab),
                                  c(position)),
        
        tgt_embed = nn.Sequential(Embeddings(d_model = d_model,
                                             vocab = tgt_vocab),
                                  c(position)),
        
        generator = Generator(d_model = d_model,
                              vocab = tgt_vocab))
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, mask_src = None, pad=0):
        
        self.src = src.permute(0,2,1)
        
        if mask_src is None:
            assert False
            self.src_mask = (src != pad).unsqueeze(-2) #probablement pas ouf
        else:
            self.src_mask = mask_src
            
        if trg is not None:
            #self.trg = trg
            # we will try to run the model without the BOS and EOS tokens
            #import pdb;pdb.set_trace() 
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
                
            self.ntokens = (self.trg != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        #import pdb; pdb.set_trace()
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

    

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        #import pdb;pdb.set_trace()
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

    
class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        #import pdb;pdb.set_trace()
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data * norm

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
        
def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    #import pdb;pdb.set_trace()
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        
        out = model.decode(memory, src_mask, Variable(ys).long(), Variable(subsequent_mask(ys.size(1)).type_as(src.data)).long())
        
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, 
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

##annotated transformer end



class Trainer:
    def __init__(self, num_blocks, num_layers, num_f_maps, dim, num_classes, device, weights, save_dir):
        
        #Original
        #self.model = MultiStageModel(num_blocks, num_layers, num_f_maps, dim, num_classes)
        
        #ASFormer params: #self, num_decoders, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate
        #self.model = MyTransformer(3, 10, 2, 2, 64, 1024, 2, 0.3)

        #Annotated
        #(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=2, dropout=0.3)
        self.model = make_model(src_vocab = 1024, tgt_vocab = 3, N = 6, d_model = 1024, d_ff = 2048, h = 4, dropout = 0.1) #ok seulement si h = 1 et bz = 1
        
        
        #import pdb; pdb.set_trace()
        
        if weights is None:
            self.ce = nn.CrossEntropyLoss(ignore_index=2)
        else:
            self.ce = nn.CrossEntropyLoss(weight=torch.tensor(weights).to(device), ignore_index=2)

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
                
                batch_target[batch_target==-100] = 2
                batch_target_eval[batch_target_eval==-100] = 2
                
                #import pdb;pdb.set_trace()

                #adding begin token 
                new_row = torch.zeros(batch_input.shape[0], batch_input.shape[1], 1).to(device)
                batch_input = torch.cat((new_row, batch_input), 2)
                
                new_row = torch.zeros(batch_target.shape[0], 1).to(device).long()
                batch_target = torch.cat((new_row, batch_target), 1)
                
                new_row = torch.zeros(batch_target_eval.shape[0], 1).to(device).long()
                batch_target_eval = torch.cat((new_row, batch_target_eval), 1)

                #import pdb;pdb.set_trace()
                new_row = torch.ones(mask.shape[0], mask.shape[1], 1).to(device)
                mask = torch.cat((new_row, mask), 2)
                
                
                # il faut arriver à comprendre pourquoi mask a une deuxieme dimension égale à 2 (peut etre pour la loss)
                optimizer.zero_grad()

                mask_input = mask[:, 0:1, :] 
                #import pdb;pdb.set_trace()
                
                annotated_batch = Batch(batch_input, batch_target, mask_src = mask_input, pad = 2)
                
                #import pdb; pdb.set_trace()
                
                predictions = self.model(annotated_batch.src, annotated_batch.trg, 
                            annotated_batch.src_mask, annotated_batch.trg_mask)
                
                
                #predictions = predictions.permute(0,2,1)
                loss = 0
                
                criterion = LabelSmoothing(size=3, padding_idx= 2, smoothing=0.1)
                model_opt = NoamOpt(1024, 1, 400,
                torch.optim.Adam(self.model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

                loss_compute = SimpleLossCompute(self.model.generator, criterion, model_opt)


                
                #import pdb; pdb.set_trace()

                loss = loss_compute(predictions, annotated_batch.trg_y, annotated_batch.ntokens)

                #predictions = F.softmax(predictions, dim=1) * mask[:, 0:1, 1:]

                #conv_out = nn.Conv1d(1024, 2, 1).to(device)
                
                #predictions = conv_out(predictions)* mask[:, 0:1, 1:]
                
                
                #import pdb; pdb.set_trace()
                #shape de predictions :  torch.Size([8, 2, 167])
                #predictions = predictions.unsqueeze(0)
                
                #loss = 0
                # loss for each stage
               # for ix, pvar in enumerate(predictions):
               #     if self.num_classes == 1:
               #         loss += self.mse_red(pvar.transpose(2, 1).contiguous().view(-1, self.num_classes).squeeze(), batch_target.view(-1))
              #      else:
              #          import pdb; pdb.set_trace()
             #           loss += self.ce(pvar.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target[:, 1:].reshape(-1))
             #           loss += 0.15*torch.mean(torch.clamp(self.mse(F.log_softmax(pvar[:, :, 1:], dim=1), F.log_softmax(pvar.detach()[:, :, :-1], dim=1)), min=0, max=16)*mask[:, :, 1:])

                epoch_loss += loss.item()
                #loss.backward()
                #optimizer.step()
                
                #import pdb; pdb.set_trace()

                #if self.num_classes == 1:
                #    predicted = torch.round(predictions[-1].data.squeeze())
               ##     gt = torch.round(batch_target)
                #    gt_eval = batch_target_eval

               # else:
                    #_, predicted = torch.max(predictions[-1].data, 1)
                    #gt = batch_target[:, 1:]
                    #gt_eval = batch_target_eval[:, 1:]
                
                
                ## à garder
                #self.model.eval()
                #full_decoded = torch.zeros(1, batch_target.shape[1]).to(device)
                #for ix, pvar in enumerate(predictions):
                    #for ix in range(2):
                  #  src = Variable(annotated_batch.src[ix]).unsqueeze(0)
                   # src_mask = Variable(annotated_batch.src_mask[ix]).unsqueeze(0)
                  #  decoded = greedy_decode(self.model, src, src_mask, max_len=annotated_batch.src.shape[1], start_symbol=0)
                  #  full_decoded = torch.cat((full_decoded, decoded), 0)
                
                #full_decoded = full_decoded[1:,:-1]
                #import pdb; pdb.set_trace()
                #get_metrics_train.calc_scores_per_batch(full_decoded,  batch_target[:, 1:], batch_target_eval[:, 1:], mask[:, 0:1, 1:])
                ## à garder
                
                
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
                    loss=loss#.item()
                )
                count += 1
                bar.next()

            batch_gen.reset()
            #torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
            #torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")

            #get_metrics_train.calc_metrics()
            #result_dict = get_metrics_train.save_print_metrics(self.writer, save_dir, epoch, epoch_loss/(len(batch_gen.list_of_examples)/batch_size))
            #self.train_result_dict.update(result_dict)

            #eval_args[7] = epoch
            #eval_args[1] = save_dir + "/epoch-" + str(epoch+1) + ".model"
            
           # self.predict(*eval_args)

       # with open(f'{save_dir}/train_results.json', 'w') as fp:
        #    json.dump(self.train_result_dict, fp, indent=4)
       # with open(f'{save_dir}/eval_results.json', 'w') as fp:
       #     json.dump(self.test_result_dict, fp, indent=4)
       # self.writer.close()


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
