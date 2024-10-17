#-*- coding: utf-8 -*-
ip_address=''
#final ver10173
try:
    import time
    import os
    print("Waiting...0",flush=True)
    time.sleep(15)
except KeyboardInterrupt:
    import sys
    sys.exit(0)
for i in range(0,11):
    time.sleep(20)
    try:
        from subprocess import check_output
        ip_address_list= (check_output(['hostname','-I']))
        ip_address_list=str(ip_address_list)
        ip_address_list=ip_address_list[2:]
        if ' ' in ip_address_list:
            ip_address_list=str(ip_address_list).split(' ')
            for i in ip_address_list:
                if '192' in i:
                    ip_address=i
                    break
            if '192' in ip_address:
                break
            else:
                raise
        else:
            ip_address=ip_address_list
        if '192' in ip_address:
            break
        else:
            raise
    except:
        print('retry... ',str(i),flush=True)
        if os.path.exists('face_test.jpg'):
            os.remove('face_test.jpg')
        if i==10:
            if os.path.exists('face_test.jpg'):
                os.remove('face_test.jpg')
            os.system('sudo shutdown -h now')

print('using '+ip_address,flush=True)
print("Waiting...1",flush=True)
try:
    print("Start service",flush=True)
    from subprocess import run
    from socket import *
    import threading
    import time
    import sys
    import traceback
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
    import xgboost 
    from time import sleep
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.utils
    import torch.utils.data
    import torch.utils.data.dataset
    import torchvision.utils as utils
    import torchvision.datasets as dsets
    import torchvision.transforms as transforms
    from torchvision.utils import save_image
    import torchvision
    import os
    import numpy as np
    import random 
    from PIL import Image, ImageEnhance
    sw300=0
    from torchvision.transforms.functional import to_pil_image
    print("Ready for connect",flush=True)
    time.sleep(1)
    port = 12345
    ip= ip_address
    received=""
    serverSock = socket(AF_INET, SOCK_STREAM)
    serverSock.bind((ip, port))
    serverSock.listen(1)
            



    face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade=cv2.CascadeClassifier('haarcascade_eye.xml')
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
    import time
    import os


    import pandas as pd
    import numpy as np
    import xgboost as xgb
    from xgboost import XGBClassifier
    import time



    timedata=[]
    device=[]
    light=[]
    measure=pd.DataFrame(columns=["Time","Device","light"])

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
    import base64
    SOS_token = 0
    EOS_token=1
    idx=0
    mode=False
    GPIO.cleanup()
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(32,GPIO.OUT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MAX_LENGTH = 8
    enc_dropout = 0.15
    servoPin          = 12
    SERVO_MAX_DUTY    = 12
    SERVO_MIN_DUTY    = 3
    imsi100=False

    latent_size = 100
    hidden_size = 512
    img_size=160
    image_size =  img_size * img_size*3
    num_epochs = 380
    batch_size = 1
    augment_cnt = 1



    GPIO.setup(servoPin, GPIO.OUT)
    servo = GPIO.PWM(servoPin, 50)
    GPIO.setup(10, GPIO.IN)   

    import datetime
    from datetime import date
    imsi30=False



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





    def filter_pair(pair):
        return len(pair[1].split(' ')) < MAX_LENGTH and len(pair[0].split(' ')) < MAX_LENGTH

    def filter_pairs(pairs):
        return [pair for pair in pairs if filter_pair(pair)]
    def readLangs():

        print("Reading...")
        lines=pd.read_csv('bigdata_command.csv', encoding='utf-8',sep=',')
        print(lines.shape)
        a=lines.iloc[:,0]
        b=lines.iloc[:,1]
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
    model = torch.load('translate2.pt', map_location=device)
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



    m=2

    G = nn.Sequential(
        nn.Linear(latent_size, 64*m),
        nn.Linear(64*m, 128*m),
        nn.Linear(128*m, 256*m),
        nn.Linear(256*m, 512*m),
        nn.ReLU(),
        nn.Linear(512*m, image_size),  
        nn.Tanh())


    weights = torch.load('G_8340.ckpt')
    G.load_state_dict(weights)







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
    sw4=False
    sw5=False
    timer1=0
    timer2=0
    timer3=0
    m1=0
    m2=0
    m3=0
    m_0=[]
    m_1=[]
    m_2=[]
    m_3=[]
    imsi70=[]
    m_4=[]
    m_5=[]
    m_6=[]
    imsi10=[]
    imsi20=0
    imsi21=0
    imsi22=0
    imsi23=0
    imsi24=0
    imsi25=0
    imsi26=0
    swlast=0
    v1=[]
    v2=[]
    v3=[]
    v11=[]
    v12=[]
    v13=[]
    v21=[]
    sw500=False
    imsi100=False
    imsi450=False
    p=""
    import pandas as pd
    df=pd.DataFrame()
    from datetime import datetime, timedelta
    GPIO.setup(35, GPIO.IN)  
    sw10=False
    model2=cv2.face.LBPHFaceRecognizer_create()
    model2.read("face_model.model")
    def control_1(sock):
        global imsi70
        global imsi100
        global sw300
        global result
        global sw
        global idx
        global p
        global m_2
        global m_1
        global m_0
        global m_3
        global df
        global v1
        global v2
        global v3
        global v11
        global v12
        global v13
        global v21
        global imsi450
        global swlast
        
        while True:
            global sw500
            global sw2
            global sw3
            global sw4
            global sw5
            global model2
            global state
            recvData = sock.recv(4096)
            received=recvData.decode('utf-8')
            if not received:
                print('an error')
                try:
                    result1="error"
                    try:
                        if os.path.exists('face_test.jpg'):
                            os.remove('face_test.jpg')
                        sock.send(result1.encode('utf-8'))
                        serverSock.close()
                        os.system('sudo shutdown -h now')
                    except:
                        pass
                    swlast=1
                    if os.path.exists('face_test.jpg'):
                        os.remove('face_test.jpg')
                    GPIO.cleanup()
                    os.system('sudo shutdown -h now')
                except:
                    swlast=1
                    if os.path.exists('face_test.jpg'):
                        os.remove('face_test.jpg')
                    os.system('sudo shutdown -h now')
            if received=="test":
                print("test",flush=True)
            if received=="lighton":
                print("lighton",flush=True)
                GPIO.output(32,True)
                sw10=True
            if received=="lightoff":
                print("lightoff",flush=True)            
                GPIO.output(32,False)
                sw10=False


            if  sw2==False:
                print('th')
                humidity_data=mydht11.humidity
                temperature_data=mydht11.temperature
                result1=humidity_data
                result2=temperature_data
                result1="t"+str(result1)+"h"+str(result2)
                print("info"+(str(result1)),flush=True)
                sock.send(result1.encode('utf-8'))
                sw2=True
            if sw300==0:
                if sw500==False:
                    now=str(pd.to_datetime('now').date())
                    sw500=True
                df=pd.DataFrame(columns=['time','a','b'])
                df['time']=pd.date_range('2024-08-25',periods=500)
                for i in range(0,500):
                    v11.append(int(0))
                    v12.append(int(0))
                    
                df['a']=v11
                df['b']=v12
                a2=list(df.columns.values)
                if 'Unnamed: 0' in a2:
                    df.drop(['Unnamed: 0'],axis=1,inplace=True)
                a2=list(df.columns.values)
                if 'Unnamed: 0' in a2:
                    df.drop(['Unnamed: 0'],axis=1,inplace=True)
                for i in df.iloc[:,0]:
                    v21.append(str(i)) 
                imsi5=df['time'].astype(str).values.tolist().index(now)
            sw300=2
            if received.startswith("light_0"):
                df.iloc[imsi5,1]+=1*3.3
            if received.startswith("device_"):
                df.iloc[imsi5,2]+=1*25
                a2=list(df.columns.values)
                if 'Unnamed: 0' in a2:
                    df.drop(['Unnamed: 0'],axis=1,inplace=True)
            if received.startswith("report_"):
                df=df.dropna(axis=0)
                xgb_model=xgboost.XGBRegressor(n_estimators=100, learning_rate=0.01, max_depth=3)
                a2=list(df.columns.values)
                if 'Unnamed: 0' in a2:
                    df.drop(labels='Unnamed: 0',axis=1,inplace=True)
                abc=[]
                for i in range(len(df.iloc[:,2])):
                    abc.append(df.iloc[i,1]+df.iloc[i,2])
                xgb_model.fit(df.iloc[:,1], abc)  
                imsi500=[df.iloc[:,1]]
                p=xgb_model.predict([df.iloc[imsi5+30,1]])
                p1=str(round(float(p*((1/1000)*93.3)),6))
                p1="predict"+p1
                print("predict",flush=True)
                sock.send(p1.encode('utf-8'))
            if received=="dooropen":
                servo.start(0)
                setServoPos(0)
                sleep(1)
                setServoPos(90)
                sleep(1)
                servo.stop()
            if received=="doorclose":
                servo.start(0)   
                setServoPos(0)
                sleep(1)
                servo.stop()
            if received.startswith("ke"):
                print('translatemode')
                result=re.sub('ke','',received)
                input_lang, output_lang, pairs = prepareData()
                randomize = random.choice(pairs)
                num_layers = 2
                input_size = input_lang.n_words
                output_size = output_lang.n_words
                embed_size = 256
                hidden_dim=512
                num_layers = 2
                num_iteration = 5200
                enc_emb_dim=256
                encoder1 = Encoder(input_lang.n_words, hidden_dim)
                decoder2= Decoder(hidden_dim, output_lang.n_words, dropout_p=0.1, max_length=MAX_LENGTH)
                result='ke'+evaluateAndShowAttention(result,encoder1,decoder2)
                sock.send(result.encode('utf-8'))
            if received=="album":
                G.eval()
                with torch.no_grad():
                    fixed_noise = torch.randn(1, 100, device=device)
                    img_fake = G(fixed_noise).detach().cpu()
                fake_images  = img_fake .reshape( 3, img_size, img_size)
                save_image((fake_images), os.path.join('flower.jpg')) 
                img=Image.open('flower.jpg')
                img=img.resize((50,50))
                buf=BytesIO()
                img.save(buf,'jpeg')                    
                buf.seek(0)
                b=buf.read()
                result=base64.b64encode(b)
                sock.send(result)


            if received=="cam":
                picam2 = Picamera2()
                picam2.configure(picam2.create_video_configuration(main={"size": (            120, 60            )}))
                output = StreamingOutput()
                picam2.start_recording(MJPEGEncoder(), FileOutput(output))
                print('camtesting',flush=True)
                try:
                    while True:
                        if sw>=1 and sw<100:
                            frames=output.frame
                            result=frames
                            result=base64.b64encode(result)
                            sock.send(result)
                        sw+=1
                        time.sleep(1)
                        if sw >23:
                            sw=0
                            picam2.stop_recording()
                            picam2.close()
                            received="waiting"
                            break
                        if received=="stopcam":
                            sw=0
                            picam2.stop_recording()
                            picam2.close()
                            received="waiting"
                            break
              
                except:
                    sw=0
                    picam2.stop_recording()
                    picam2.close()
                    received="waiting"
                    break
                    
            if imsi100==True:
                try:
                    gray_cropped=0
                    os.system("libcamera-jpeg -t 1000 -o face_test.jpg")
                    img=cv2.imread("face_test.jpg")
                    face=face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
                    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    for (x, y, w, h) in face:
                        gray_cropped=gray[y:y+h,x:x+w]
                    gray_cropped=cv2.resize(gray_cropped, (512,512),interpolation=cv2.INTER_AREA)
                    result=model2.predict(gray_cropped)
                    if result[1]>=0 and result[1]<117:
                        servo.start(0)
                        setServoPos(0)
                        sleep(1)
                        setServoPos(90)
                        sleep(1)
                        servo.stop()
                        result1="open_door"
                        sw5=True
                        sock.send(result1.encode('utf-8'))
                        

                    else:
                        result1="notmatching"
                        sock.send(result1.encode('utf-8'))
                    if os.path.exists('face_test.jpg'):                        
                        os.remove('face_test.jpg')
                    sw4=False
                except:
                    imsi100==False
                    pass


            
    def control_2():
        global imsi100
        global swlast
        while True:
            if swlast ==0:     
                state =  GPIO.input(35)
                if(state == True):
                    pass
                    imsi100=True
                else:
                    imsi100=False
                time.sleep(0.2) 



    serverSock = socket(AF_INET, SOCK_STREAM)
    serverSock.bind((ip, port))
    serverSock.listen(1)

    print('waiting...',flush=True)

    connectionSock, addr = serverSock.accept()

    print('Connected.',flush=True)
    c1 = threading.Thread(target=control_1, args=(connectionSock,))
    c1.start()
    c2=threading.Thread(target=control_2)
    c2.start()

    print("exited")

    while True:
        time.sleep(1000)
        pass

except BrokenPipeError as e:
    print('an error occured (Broken Pipe)',flush=True)
    print(e)
    traceback.print_exc()
    try:
        GPIO.cleanup()
        if os.path.exists('face_test.jpg'):
            os.remove('face_test.jpg')
        result1="error"
        sock.send(result1.encode('utf-8'))
        serverSock.close()
        import os
        os.system('sudo shutdown -h now')
       
    except:
        pass
    pass
except:
    print('an error occured',flush=True)
    traceback.print_exc()
    try:
        GPIO.cleanup()
        if os.path.exists('face_test.jpg'):
            os.remove('face_test.jpg')
        result1="error"
        sock.send(result1.encode('utf-8'))
        serverSock.close()
        os.system('sudo shutdown -h now')
        time.sleep(10)
       
    except:
        print('error')
        if os.path.exists('face_test.jpg'):
            os.remove('face_test.jpg')
        os.system('sudo shutdown -h now')
    if os.path.exists('face_test.jpg'):
        os.remove('face_test.jpg')
    os.system('sudo shutdown -h now')
