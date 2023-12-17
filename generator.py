#this code is modified from https://github.com/HotThoughts/SalGAN and https://github.com/imatge-upc/salgan

import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision

# class Generator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Encoder
#         self.encoder = torchvision.models.vgg16(pretrained=True).features[:-1]
#         # freeze all layers except the last two conv layers
#         for i, param in enumerate(self.encoder.parameters()):
#             if i == 24: break
#             param.requires_grad = False
#         # Decoder
#         self.decoder1 = nn.Sequential(
#             nn.Conv2d(1024, 512, 3, padding=1), nn.ReLU(), 
#             nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(), 
#             nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        
#         self.decoder2 = nn.Sequential(
#             nn.Conv2d(1024, 512, 3, padding=1),nn.ReLU(), 
#             nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(), 
#             nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        
#         self.decoder3 = nn.Sequential(
#             nn.Conv2d(1024, 256, 3, padding=1),nn.ReLU(), 
#             nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), 
#             nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), 
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        
#         self.decoder4 = nn.Sequential(
#             nn.Conv2d(512, 128, 3, padding=1),nn.ReLU(),  
#             nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        
#         self.decoder5 = nn.Sequential(
#             nn.Conv2d(256, 64, 3, padding=1),nn.ReLU(), 
#             nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
#             nn.Conv2d(64, 1, 1, padding=0)
#         )
        
#         self.text_conv1 = nn.Conv2d(512,256,1,padding=0)
#         self.text_conv2 = nn.Conv2d(512,128,1,padding=0)
        
        
        
        
#     def forward(self, x,text):
        
#         x = self.encoder(x)
#         text = text.view(text.size(0), 512, 1, 1).float()
#         #text = self.text_conv(text)
#         text1 = text.expand(-1, -1, 16, 16)
        
#         #print(x.device,text.device)
#         #x = x.cuda()
#         x = torch.cat([x,text1],dim=1)
#         #print(x.shape)
#         x = self.decoder1(x)
        
#         text2 = text.expand(-1, -1, 32, 32)
#         x=torch.cat([x,text2],dim=1)
#         x=self.decoder2(x)
        
#         text3 = text.expand(-1, -1, 64, 64)
#         x=torch.cat([x,text3],dim=1)
#         x=self.decoder3(x)
        
#         text4 = text.expand(-1, -1, 128, 128)
#         text4 = self.text_conv1(text4)
#         x=torch.cat([x,text4],dim=1)
#         x=self.decoder4(x)
        
#         text5 = text.expand(-1, -1, 256, 256)
#         text5 = self.text_conv2(text5)
#         x=torch.cat([x,text5],dim=1)
#         x=self.decoder5(x)
        
#        # print(x.shape)
#         return x



class Generator(nn.Module):
    def __init__(self, text_embedding_dim=256):
        super(Generator, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        

        self.decoder1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.decoder2=nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.decoder3=nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.text_conv1 = nn.Conv2d(512, 64, kernel_size=1)
        self.text_conv2 = nn.Conv2d(512, 64, kernel_size=1)
        self.text_conv3 = nn.Conv2d(512, 64, kernel_size=1)
        self.text_conv4 = nn.Conv2d(512, 64, kernel_size=1)
        
        



    def forward(self, x, text):
        x = self.encoder1(x)
        text = text.view(text.size(0), 512, 1, 1).float()
        # text = self.text_conv1(text)
        # text1 = text.expand(-1, -1, 256, 256)
        # text2 = text.expand(-1, -1, 128, 128)
        
        text1=self.text_conv1(text).expand(-1, -1, 256, 256)
        x = self.encoder2(torch.cat([x, text1], dim=1))
        # x=self.encoder2(x)
        
        text2=self.text_conv2(text).expand(-1, -1, 256, 256)
        x = self.encoder3(torch.cat([x, text2], dim=1))
        # x=self.encoder3(x)
        
        
        x=self.decoder1(x)
        
        text3=self.text_conv3(text).expand(-1, -1, 128, 128)
        x=self.decoder2(torch.cat([x, text3], dim=1))
        
        text4=self.text_conv4(text).expand(-1, -1, 128, 128)
        x=self.decoder3(torch.cat([x, text4], dim=1))
        

        return x
    
class Generator2(nn.Module):
    def __init__(self, text_embedding_dim=256):
        super(Generator, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        

        self.decoder1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.decoder2=nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.decoder3=nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # self.text_conv1 = nn.Conv2d(512, 64, kernel_size=1)
        # self.text_conv2 = nn.Conv2d(512, 64, kernel_size=1)
        self.text_conv3 = nn.Conv2d(512, 64, kernel_size=1)
        self.text_conv4 = nn.Conv2d(512, 64, kernel_size=1)
        
        



    def forward(self, x, text):
        x = self.encoder1(x)
        text = text.view(text.size(0), 512, 1, 1).float()
        # text = self.text_conv1(text)
        # text1 = text.expand(-1, -1, 256, 256)
        # text2 = text.expand(-1, -1, 128, 128)
        
        # text1=self.text_conv1(text).expand(-1, -1, 256, 256)
        # x = self.encoder2(torch.cat([x, text1], dim=1))
        x=self.encoder2(x)
        
        # text2=self.text_conv2(text).expand(-1, -1, 256, 256)
        # x = self.encoder3(torch.cat([x, text2], dim=1))
        x=self.encoder3(x)
        
        
        x=self.decoder1(x)
        
        text3=self.text_conv3(text).expand(-1, -1, 128, 128)
        x=self.decoder2(torch.cat([x, text3], dim=1))
        
        text4=self.text_conv4(text).expand(-1, -1, 128, 128)
        x=self.decoder3(torch.cat([x, text4], dim=1))
        

        return x