#-*- coding: utf-8 -*-
from socket import *
import threading
import time
import sys
import select
import time
import board
import adafruit_dht
import RPi.GPIO as GPIO
import time
import board
import adafruit_dht
import io
import cv2
from PIL import Image
from time import sleep
port = "port"
ip= "ip_address"
received=""
serverSock = socket(AF_INET, SOCK_STREAM)
serverSock.bind((ip, port))
serverSock.listen(1)
from PIL import Image


import sys
import io
import unicodedata
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')
import subprocess

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
import json
import re
""""
install('scikit_learn')
install('numpy')
install('pandas')
install('tqdm')
install('pyarrow')
"""

from io import open
import unicodedata
import string
import re
import base64
from io import BytesIO
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
import base64
SOS_token = 0
EOS_token=1
mode=False
#GPIO.setmode(GPIO.BCM)
#GPIO.setup(12,GPIO.OUT)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(32,GPIO.OUT)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 8
enc_dropout = 0.15
servoPin          = 12
SERVO_MAX_DUTY    = 12
SERVO_MIN_DUTY    = 3

GPIO.setup(servoPin, GPIO.OUT)
servo = GPIO.PWM(servoPin, 50)
servo.start(0)
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



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




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


def filter_pair(pair):
    return len(pair[0].split(' ')) < MAX_LENGTH and len(pair[1].split(' ')) < MAX_LENGTH

def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]
def readLangs():
    print("Reading...")
    lines=pd.read_csv(r'/home/ysh/models/translate2/kor.txt', encoding='utf-8',sep='\t')
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
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    return input_lang, output_lang, pairs



class Encoder(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(input_size, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_dim, device=device)







class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

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
            encoder_output, encoder_hidden = self.encoder( input_tensor[ei], encoder_hidden)
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



def Model(model, input_tensor, target_tensor, model_optimizer, criterion):
    model_optimizer.zero_grad()
    input_length = input_tensor.size(0)
    loss = 0
    epoch_loss = 0
    output = model(input_tensor, target_tensor,input_tensor)
    num_iter = output.size(0)

    for ot in range(num_iter):
        loss += criterion(output[ot], target_tensor[ot])

    loss.backward()
    model_optimizer.step()
    epoch_loss = loss.item() / num_iter
    return epoch_loss





sw1=False
input_lang, output_lang, pairs = prepareData()
randomize = random.choice(pairs)
num_layers = 2
input_size = input_lang.n_words
output_size = output_lang.n_words
model = torch.load('/home/ysh/models/translate2/translate2.pt', map_location=device)
embed_size = 256
hidden_dim=512
num_layers = 2
num_iteration = 5200
enc_emb_dim=256
encoder1 = Encoder(input_lang.n_words, hidden_dim)
decoder2= Decoder(hidden_dim, output_lang.n_words, dropout_p=0.1, max_length=MAX_LENGTH)





def evaluate(model,encoder, decoder, input_sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, input_sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_dim, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder( input_tensor[ei],  encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words

def evaluateAndShowAttention(input_sentence,encoder, decoder):
    output_words = evaluate(model, encoder, decoder, input_sentence)
    print('input =', input_sentence,flush=True)
    
    result = list(set(output_words))
    print('result is:',flush=True)

    final_result=re.sub("<EOF>","",(' '.join(result)))
    print(final_result,flush=True)    
    return final_result
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput
from threading import Condition
import time
import base64
import io
sw=0
class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (            60, 30            )}))
output = StreamingOutput()
picam2.start_recording(MJPEGEncoder(), FileOutput(output))
received=""
mydht11=adafruit_dht.DHT11(board.D4)
a=0
data_result=""
temp_result=""
result=""
def setServoPos(degree):
	if degree > 180:
		degree = 180	
	duty = SERVO_MIN_DUTY+(degree*(SERVO_MAX_DUTY-SERVO_MIN_DUTY)/180.0)
	print("Degree: {} to {}(Duty)".format(degree, duty))
	servo.ChangeDutyCycle(duty)
	
def receive_data(sock):
	global data_result
	global received
	global a
	print("start receiving")
	recvData = sock.recv(1024)
	received=recvData.decode('utf-8')
	if received!=""or received!=None:
		data_result=received
		return data_result
	if received=='' or received==None:
		a+=1        
	if a>10:
		serverSock.close()
		quit()  
sw2=False
sw3=False
def control_light(sock):
    global result
    global sw
    try:
        while True:
            global sw2
            global sw3
            recvData = sock.recv(4096)
            received=recvData.decode('utf-8')
            if received=="test":
                print("test",flush=True)
            if received=="lighton":
                print("lighton",flush=True)
                GPIO.output(32,True)
            if received=="lightoff":
                print("lightoff",flush=True)            
                GPIO.output(32,False)

            if received=="command11" and sw2==False:
                print('th')
                humidity_data=mydht11.humidity
                temperature_data=mydht11.temperature
                result1=humidity_data
                result2=temperature_data
                result1="t"+str(result1)+"h"+str(result2)
                sock.send(result1.encode('utf-8'))
		
                sw2=True
            if received=="dooropen":
                print("dooropen",flush=True)  
                setServoPos(0)
                sleep(1)
                setServoPos(90)
                sleep(1)
            if received=="doorclose":
                print("doorclose",flush=True)            

                setServoPos(0)
                sleep(1)
                servo.stop()
            if received.startswith("ke"):
                print('translatemode')
                result=re.sub('ke','',received)
                result='ke'+evaluateAndShowAttention(result,encoder1,decoder2)
                sock.send(result.encode('utf-8'))
            if received=="cam":
                 print('camtesting',flush=True)
                 while True:
                    if sw>=1 and sw<10:
                        print(type(output.frame))
                        frames=output.frame
                        result=frames
                        result=base64.b64encode(result)
                        sock.send(result)
                        print('\n\n')
                        print(result)
                    sw+=0.5
                    time.sleep(0.5)
                    print(sw)
                    if sw >=10:
                        break
                 picam2.stop_recording()
                 picam2.close()
		
            #received=""
    except KeyboardInterrupt:
        print('stopping')
        servo.stop()
        GPIO.output(32,False)
        GPIO.cleanup()
        socket.close()
	
		



serverSock = socket(AF_INET, SOCK_STREAM)
serverSock.bind((ip, port))
serverSock.listen(1)

print('접속 대기중...',flush=True)

connectionSock, addr = serverSock.accept()

print('접속되었습니다.',flush=True)
light = threading.Thread(target=control_light, args=(connectionSock,))
light.start()
light.join()
print("exited")

while True:
    time.sleep(1000)
    pass
