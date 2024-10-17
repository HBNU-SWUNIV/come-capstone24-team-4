# -*- coding: euc-kr -*
import sys
import io
import unicodedata
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')
import subprocess
print('test',flush=True)
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
import re
from io import open
import unicodedata
import string
import re
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import glob
from konlpy.tag import Okt
from konlpy.tag import Kkma
from konlpy.tag import Komoran
SOS_token=0
EOS_token=1
mode=True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 5
num_layers = 1
embed_size = 256
hidden_dim=512
num_iteration = 37000

class Lang:
    def __init__(self, name):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "<SOS>", 1: "<EOS>"}
        self.n_words = 2 

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(lang) : 
    lang = unicodeToAscii(lang.lower().strip())
    return lang


mode=True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def filter_pair(pair):
    return len(pair[0].split(' ')) < MAX_LENGTH and len(pair[1].split(' ')) < MAX_LENGTH

def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]
def readLangs():
    print("Reading...")
    lines=pd.read_csv(r'path_here', encoding='utf-8',sep='\t')
    print(lines.shape)
    a=lines.iloc[:,1]
    b=lines.iloc[:,0]
    c=[]
    pairs=[]
    a2=[]
    b2=[]
    for iii in range(len(a)):
        i1 = re.sub(r"[^A-Za-z0-9ㄱ-ㅣ가-힝+\s]","",a[iii])
        a2.append(i1)
    for iii in range(len(a)):
        i1 = re.sub(r"[^A-Za-z0-9ㄱ-ㅣ가-힝+\s]","",b[iii])
        b2.append(i1)
    for i in range(len(a)):
        pairs.append([a2[i],b[i]])
    print('len_pair')
    print(len(pairs))
    input_lang = Lang(a2)
    output_lang = Lang(b2)
    
    return input_lang, output_lang, pairs

def prepareData():
    input_lang, output_lang, pairs = readLangs()
    pairs = filter_pairs(pairs)
    print('datas')
    print(pairs[0][0:13],flush=True)
    print(pairs[1][0:13],flush=True)
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    return input_lang, output_lang, pairs



class Encoder(nn.Module):
    def __init__(self, input_size, hidden_dim,dropout=0.05):
        super(Encoder, self).__init__()
        self.input_size = input_size        
        self.hidden_dim = hidden_dim
        self.dropout=dropout
        self.embedding = nn.Embedding(input_size, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return hidden, output

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_dim, device=device)







class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropout=0.05, max_length=MAX_LENGTH):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
         return torch.zeros(1, 1, self.hidden_size, device=device)




def indexesFromSentence(lang, sentence):
         
    global mode
    if mode==True:
        return [lang.word2index[word] for word in sentence.split(' ')]
    emsi1=[]
    if mode==False:
        emsi10=[lang.word2index[word] for word in sentence.split(' ') if word in list(lang.word2index)]

        print(emsi10,flush=True)
        return emsi10


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder2, device, MAX_LENGTH=MAX_LENGTH):
        super().__init__()

        self.encoder = encoder
        self.decoder2 = decoder2
        self.device = device
     
    def forward(self, input_lang, output_lang, input_tensor,teacher_forcing=True):
        encoder_hidden = self.encoder.initHidden()
        input_length = input_lang.size(0)
        batch_size = output_lang.shape[1] 
        target_length = output_lang.shape[0]
        vocab_size = self.decoder2.output_size
        outputs = torch.zeros(target_length, batch_size, vocab_size).to(self.device)
        encoder_outputs = torch.zeros(MAX_LENGTH, self.encoder.hidden_dim, device=device)

        for ei in range(input_length):
            encoder_hidden ,encoder_output= self.encoder( input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]
        decoder_hidden = encoder_hidden.to(device)
        decoder_input = torch.tensor([SOS_token], device=device)  
        for t in range(target_length):   
            decoder_output, decoder_hidden, decoder_attention = decoder2(  decoder_input, decoder_hidden, encoder_outputs)
            outputs[t] = decoder_output
            teacher_force = teacher_forcing
            topv, topi = decoder_output.topk(1)
            input = (output_lang[t] if teacher_force else topi)
            if (teacher_force == False and input.item() == EOS_token) :
                break
        return outputs



def Model(model, input_tensor, target_tensor, criterion,encoder_optimizer,decoder_optimizer):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    input_length = input_tensor.size(0)
    loss = 0
    epoch_loss = 0
    output = model(input_tensor, target_tensor,input_tensor)
    num_iter = output.size(0)

    for ot in range(num_iter):
        loss += criterion(output[ot], target_tensor[ot])

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    epoch_loss = loss.item() / num_iter
    return epoch_loss

def trainModel(model, input_lang, output_lang, pairs, num_iteration,encoder,decoder):
    print("start to train...",flush=True)
    model.train()
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=0.01)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=0.01)
    criterion = nn.NLLLoss()
    total_loss_iterations = 0

    training_pairs = [tensorsFromPair( random.choice(pairs))
                      for i in range(num_iteration)]
  
    for iter in range(1, num_iteration+1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        loss = Model(model, input_tensor, target_tensor, criterion,encoder_optimizer,decoder_optimizer)
        total_loss_iterations += loss

        if iter % 100 == 0:
            average_loss= total_loss_iterations / 100
            total_loss_iterations = 0
            print('%d %.4f' % (iter, average_loss),flush=True)
          
    return model







input_lang, output_lang, pairs = prepareData()
randomize = random.choice(pairs)
print('random sentence {}'.format(randomize))
input_size = input_lang.n_words
output_size = output_lang.n_words
print('Input : {} Output : {}'.format(input_size, output_size))


encoder1 = Encoder(input_lang.n_words, hidden_dim)
decoder2= Decoder(hidden_dim, output_lang.n_words, dropout=0.1, max_length=MAX_LENGTH)
model = Seq2Seq(encoder1, decoder2, device).to(device)

model = trainModel(model, input_lang, output_lang, pairs, num_iteration,encoder1,decoder2)

torch.save(encoder1.state_dict(), 'encoder.dict')
torch.save(decoder2.state_dict(), 'decoder.dict')
torch.save(model.state_dict(), 'seq2seq.dict')
