#-*- coding: utf-8 -*-
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
import base64
from io import BytesIO
import os
import glob
from konlpy.tag import Okt
from konlpy.tag import Kkma
from konlpy.tag import Komoran
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
from PIL import Image
from time import sleep
SOS_token=0
EOS_token=1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 5
num_layers = 2
embed_size = 256
hidden_dim=512
mode=False
GPIO.setmode(GPIO.BOARD)
GPIO.setup(32,GPIO.OUT)
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
    lines=pd.read_csv(r'file', encoding='utf-8',sep='\t')
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


received=""
mydht11=adafruit_dht.DHT11(board.D4)
a=0
data_result=""
result=""

	
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



input_lang, output_lang, pairs = prepareData()
randomize = random.choice(pairs)
print('random sentence {}'.format(randomize))
input_size = input_lang.n_words
output_size = output_lang.n_words
print('Input : {} Output : {}'.format(input_size, output_size))


encoder1 = Encoder(input_lang.n_words, hidden_dim)
decoder2= Decoder(hidden_dim, output_lang.n_words, dropout=0.1, max_length=MAX_LENGTH)
encoder1.load_state_dict(torch.load('encoder_dict_location'))
decoder2.load_state_dict(torch.load('decoder_dict_location'))




def evaluate(encoder, decoder, input_sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, input_sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_dim, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder( input_tensor[ei],  encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)

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
    output_words = evaluate( encoder, decoder, input_sentence)
    print('input =', input_sentence,flush=True)
    
    result = list(set(output_words))
    print('result is:',flush=True)

    final_result=re.sub("<EOF>","",(' '.join(result)))
    print(final_result,flush=True)    
    return final_result

def setServoPos(degree):
	if degree > 180:
		degree = 180	
	duty = SERVO_MIN_DUTY+(degree*(SERVO_MAX_DUTY-SERVO_MIN_DUTY)/180.0)
	servo.ChangeDutyCycle(duty)
	


def setServoPos(degree):
	if degree > 180:
		degree = 180	
	duty = SERVO_MIN_DUTY+(degree*(SERVO_MAX_DUTY-SERVO_MIN_DUTY)/180.0)
	print("Degree: {} to {}(Duty)".format(degree, duty))
	servo.ChangeDutyCycle(duty)


sw2=False
sw3=False
def control(sock):
    global result
    try:
        while True:
            global sw2
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
                 os.system("libcamera-jpeg -t 2000 -o test.jpg")
                 sleep(3)
                 img=Image.open('test.jpg')
                 img=img.resize((90,55))
                 buff=BytesIO()
                 img.save(buff,format='JPEG',quality=20)
                 encoded_string=base64.b64encode(buff.getvalue())
                 camtxt='cam'
                 print('caminfo',flush=True)
                 print(encoded_string,flush=True)
                 print('--',flush=True)
                 sock.send(encoded_string)
                 print('send_end',flush=True)
                 received=""
		
            received=""
    except KeyboardInterrupt:
        print('stopping')
        servo.stop()
        GPIO.output(32,False)
        GPIO.cleanup()
        socket.close()
	
		




port = 0000
ip = "192.000.00.00" 
serverSock = socket(AF_INET, SOCK_STREAM)
serverSock.bind((ip, port))
serverSock.listen(1)

print('접속 대기중...',flush=True)

connectionSock, addr = serverSock.accept()

print('접속되었습니다.',flush=True)
program = threading.Thread(target=control, args=(connectionSock,))
program.start()
program.join()
print("exited")

while True:
    time.sleep(1000)
    pass
