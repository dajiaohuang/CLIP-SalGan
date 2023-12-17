#this code is modified from https://github.com/HotThoughts/SalGAN and https://github.com/imatge-upc/salgan

import torch.nn as nn
import torch

# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.conv1 = nn.Sequential( 
#             nn.Conv2d(1, 3, 1, padding=1), nn.ReLU(), 
#             nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), 
#             nn.MaxPool2d(2, stride=2),
#             nn.Conv2d(32, 64, 3, padding=1),  nn.ReLU(),
#             nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
#             nn.MaxPool2d(2, stride=2),
#         )
        
        
        
#         self.linear = nn.Sequential(
#             #nn.Linear(65536, 100), nn.Tanh(),
#             nn.Linear(65536, 100), nn.Tanh(),
#             nn.Linear(100,2), nn.Tanh(),
#             nn.Linear(2,1), nn.Sigmoid()
#         )
        
#         self.conv2= nn.Sequential(
            
#             nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
#             nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
#             nn.MaxPool2d(2, stride=2)
#         )
        
#         self.text_conv = nn.Conv2d(512, 64, kernel_size=1)
    

#     def forward(self, x, text):
#         #print(x.shape)
#         x = self.conv1(x)
#         #print(x.shape)
        
#         #print(text.shape)
#         #text = text.view(text.size(0), 512, 1, 1).float()
#         #print(text.shape)
#         #text = self.text_conv(text)
#         #print(text.shape)
#         #text = text.expand(-1, -1, 64, 64)
#         #print(text.shape)
        
        
#         #x = torch.cat([x,text],dim=1)
#         x = self.conv2(x)
#         x = torch.flatten(x, start_dim=1)
#         #print(x.shape)
        
        
#         x = self.linear(x)
#         return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(128,128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )
        

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 128 * 128, 1),
            nn.Sigmoid()
        )
        
        self.text_conv = nn.Conv2d(512,64,1)
        

    def forward(self, x,text):
        text = text.view(text.size(0), 512, 1, 1).float()
        text=self.text_conv(text).expand(-1, -1, 256, 256)
        
        x = self.conv1(x)
        x = self.conv2(torch.cat([x, text], dim=1))
        
        x = self.maxpool(x)
        x = self.fc(x)
        return x


