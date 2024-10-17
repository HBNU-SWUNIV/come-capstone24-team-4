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
import matplotlib.pyplot as plt
import random 
from PIL import Image, ImageEnhance
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device = {device}")
sample_dir ="path_here"
latent_size = 100
hidden_size = 512
img_size=160
image_size =  img_size * img_size*3
num_epochs = 900
batch_size = 1
augment_cnt = 1
mode1=4#4
count=0
repeat_num=9 #13
if mode1==4:
    aa=0
    path="path_here"
    path2="path_here"
    p=os.listdir(path)
    for ii in p:   
        image = Image.open(path+'/'+ii)
        for i2 in range(0, repeat_num):
            random_augment =random.randint(1,6)
            if(random_augment == 1):
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                image.save(path2+'/' + str(aa)  +str(ii))
            elif(random_augment == 2):
                image = image.rotate(random.randrange(-34, 34))
                image.save(path2+'/' + str(aa)  +str(ii))
            elif(random_augment == 3):
                image = ImageEnhance.Color(image)
                image= image.enhance(round(random.uniform(1, 1.6),2))
                image.save(path2+'/' + str(aa)  +str(ii))
            elif(random_augment == 4):
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                image.save(path2+'/' + str(aa)  +str(ii))
            if(random_augment == 5):
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                image.save(path2+'/' + str(aa)  +str(ii))
            aa=aa+1


transform = transforms.Compose([ transforms.ToTensor(),  transforms.Resize((img_size,img_size))])


trainset=torchvision.datasets.ImageFolder(root=,transform=transform)
data_loader = torch.utils.data.DataLoader(dataset=trainset,  batch_size=batch_size,   shuffle=True)


m=2
D = nn.Sequential(
    
    nn.Linear(image_size, 512*m),
    nn.LeakyReLU(0.2),
    nn.Linear(512*m, 256*m),
    nn.LeakyReLU(0.2),
    nn.Linear(256*m, 128*m),
    nn.LeakyReLU(0.2),
    nn.Linear(128*m, 64*m),
    nn.LeakyReLU(0.2),
    nn.Linear(64*m, 1),
    nn.Sigmoid()) 


G = nn.Sequential(
    nn.Linear(latent_size, 64*m),
    nn.Linear(64*m, 128*m),
    nn.Linear(128*m, 256*m),
    nn.Linear(256*m, 512*m),
    nn.ReLU(),
    nn.Linear(512*m, image_size),  
    nn.Tanh())

D = D.to(device)
G = G.to(device)

criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)

def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()
dx_epoch = []
dgx_epoch = []
total_step = len(data_loader)
imsi2=0
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(data_loader):

        images = images.reshape(batch_size,-1).to(device)

        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        outputs = D(images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs
        

        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs
        
 
        d_loss = d_loss_real + d_loss_fake
        reset_grad()
        d_loss.backward()
        d_optimizer.step()
        

        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
    
        g_loss = criterion(outputs, real_labels)

        reset_grad()
        g_loss.backward()
        g_optimizer.step()
        
        if (i+1) % 2 == 0:
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}' 
                  .format(epoch, num_epochs, i+1, total_step, d_loss.item(), g_loss.item(), 
                          real_score.mean().item(), fake_score.mean().item()),flush=True)
    dx_epoch.append(real_score.mean().item())            
    dgx_epoch.append(fake_score.mean().item())

        
    images = images.reshape(3, img_size, img_size)
    fake_images  = fake_images.reshape( 3, img_size, img_size)
    save_image((images), os.path.join(sample_dir, 'q512_final_5_1_m_real_images-{}.png'.format(epoch+1)))
    save_image((fake_images), os.path.join(sample_dir, 'q512_final_5_1_m_fake_images-{}.png'.format(epoch+1)))
    count=count+1
    if count%4==0:
        torch.save(G.state_dict(), 'G_8'+str(count)+'.ckpt')
        torch.save(D.state_dict(), 'D_8'+str(count)+'.ckpt')
