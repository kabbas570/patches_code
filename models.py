import torch
import torch.nn as nn

def double_conv01(in_channels, out_channels,f_size):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, f_size, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, f_size, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
    )


def double_conv11(in_channels, out_channels,f_size,p_size):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, f_size,padding=p_size,stride=2),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
        
        nn.Conv2d(out_channels, out_channels, f_size, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, f_size, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
    ) 

def double_conv_u1(in_channels, out_channels,f_size):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, f_size, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, f_size, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
    ) 


def trans_1(in_channels, out_channels,f_size,st_size):
    return nn.Sequential(
       nn.ConvTranspose2d(in_channels,out_channels, kernel_size=f_size, stride=st_size),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
    ) 


class m_unet_33_1(nn.Module):

    def __init__(self, input_channels=3):
        super().__init__()
                
        self.dconv_down1 = double_conv01(input_channels, 64,(3,1))
        self.dconv_down2 = double_conv11(64, 128,(3,3),(1,1))
        self.dconv_down3 = double_conv11(128, 256,(3,3),(1,1))
        self.dconv_down4 = double_conv11(256, 512,(3,3),(1,1))
        self.dconv_down5 = double_conv11(512, 512,(3,3),(1,1))
        
        self.up0 = trans_1(512,256, 2,2)
        self.up1 = trans_1(512,256,  2,2)
        self.up2 = trans_1(256, 128, 2,2)
        self.up3 = trans_1(128, 64, 2,2)
        self.up4 = trans_1(128, 64,  2,2)

        
        self.dconv_up0 = double_conv_u1(256 + 512, 512,(3,3))
        self.dconv_up1 = double_conv_u1(256 + 256, 256,(3,3))
        self.dconv_up2 = double_conv_u1(128+128, 128,(3,3))
        self.dconv_up3 = double_conv_u1(64+64,64,(3,3))
        
        self.conv_last = nn.Conv2d(64, 1, 1)
        self.activation = torch.nn.Sigmoid()
        
        
    def forward(self, x_in):
        
        conv1 = self.dconv_down1(x_in)      
        conv2 = self.dconv_down2(conv1)
        conv3 = self.dconv_down3(conv2)
        conv4 = self.dconv_down4(conv3)
        conv5 = self.dconv_down5(conv4)
        

        # ## decoder ####
        u0=self.up0(conv5)
        u0 = torch.cat([u0, conv4], dim=1) 
        u0=self.dconv_up0(u0)
        
        u1=self.up1(u0)
        u1 = torch.cat([u1, conv3], dim=1) 
        u1=self.dconv_up1(u1)

        u2=self.up2(u1)
        u2 = torch.cat([u2, conv2], dim=1) 
        u2=self.dconv_up2(u2)
          
        u3=self.up3(u2)
        u3 = torch.cat([u3, conv1], dim=1) 
        u3=self.dconv_up3(u3)
        
        out=self.conv_last(u3)
        return self.activation(out)
        
class m_unet_33_2(nn.Module):

    def __init__(self, input_channels=2):
        super().__init__()
                
        self.dconv_down1 = double_conv01(input_channels, 64,(3,1))
        self.dconv_down2 = double_conv11(64, 128,(3,3),(1,1))
        self.dconv_down3 = double_conv11(128, 256,(3,3),(1,1))
        self.dconv_down4 = double_conv11(256, 512,(3,3),(1,1))
        self.dconv_down5 = double_conv11(512, 512,(3,3),(1,1))
        
        self.up0 = trans_1(512,256, 2,2)
        self.up1 = trans_1(512,256,  2,2)
        self.up2 = trans_1(256, 128, 2,2)
        self.up3 = trans_1(128, 64, 2,2)
        self.up4 = trans_1(128, 64,  2,2)

        
        self.dconv_up0 = double_conv_u1(256 + 512, 512,(3,3))
        self.dconv_up1 = double_conv_u1(256 + 256, 256,(3,3))
        self.dconv_up2 = double_conv_u1(128+128, 128,(3,3))
        self.dconv_up3 = double_conv_u1(64+64,64,(3,3))
        
        self.conv_last = nn.Conv2d(64, 1, 1)
        self.activation = torch.nn.Sigmoid()
        
        
    def forward(self, x_in):
        
        conv1 = self.dconv_down1(x_in)      
        conv2 = self.dconv_down2(conv1)
        conv3 = self.dconv_down3(conv2)
        conv4 = self.dconv_down4(conv3)
        conv5 = self.dconv_down5(conv4)
        

        # ## decoder ####
        u0=self.up0(conv5)
        u0 = torch.cat([u0, conv4], dim=1) 
        u0=self.dconv_up0(u0)
        
        u1=self.up1(u0)
        u1 = torch.cat([u1, conv3], dim=1) 
        u1=self.dconv_up1(u1)

        u2=self.up2(u1)
        u2 = torch.cat([u2, conv2], dim=1) 
        u2=self.dconv_up2(u2)
          
        u3=self.up3(u2)
        u3 = torch.cat([u3, conv1], dim=1) 
        u3=self.dconv_up3(u3)
        
        out=self.conv_last(u3)
        return self.activation(out)

class m_unet_33_3(nn.Module):

    def __init__(self, input_channels=1):
        super().__init__()
                
        self.dconv_down1 = double_conv01(input_channels, 64,(3,1))
        self.dconv_down2 = double_conv11(64, 128,(3,3),(1,1))
        self.dconv_down3 = double_conv11(128, 256,(3,3),(1,1))
        self.dconv_down4 = double_conv11(256, 512,(3,3),(1,1))
        self.dconv_down5 = double_conv11(512, 512,(3,3),(1,1))
        
        self.up0 = trans_1(512,256, 2,2)
        self.up1 = trans_1(512,256,  2,2)
        self.up2 = trans_1(256, 128, 2,2)
        self.up3 = trans_1(128, 64, 2,2)
        self.up4 = trans_1(128, 64,  2,2)

        
        self.dconv_up0 = double_conv_u1(256 + 512, 512,(3,3))
        self.dconv_up1 = double_conv_u1(256 + 256, 256,(3,3))
        self.dconv_up2 = double_conv_u1(128+128, 128,(3,3))
        self.dconv_up3 = double_conv_u1(64+64,64,(3,3))
        
        self.conv_last = nn.Conv2d(64, 1, 1)
        self.activation = torch.nn.Sigmoid()
        
        
    def forward(self, x_in):
        
        conv1 = self.dconv_down1(x_in)      
        conv2 = self.dconv_down2(conv1)
        conv3 = self.dconv_down3(conv2)
        conv4 = self.dconv_down4(conv3)
        conv5 = self.dconv_down5(conv4)
        

        # ## decoder ####
        u0=self.up0(conv5)
        u0 = torch.cat([u0, conv4], dim=1) 
        u0=self.dconv_up0(u0)
        
        u1=self.up1(u0)
        u1 = torch.cat([u1, conv3], dim=1) 
        u1=self.dconv_up1(u1)

        u2=self.up2(u1)
        u2 = torch.cat([u2, conv2], dim=1) 
        u2=self.dconv_up2(u2)
          
        u3=self.up3(u2)
        u3 = torch.cat([u3, conv1], dim=1) 
        u3=self.dconv_up3(u3)
        
        out=self.conv_last(u3)
        return self.activation(out)
        
        
# def model() -> m_unet_33_1:
#     model = m_unet_33_1()
#     return model


# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# from torchsummary import summary
# model = model()
# model.to(device=DEVICE,dtype=torch.float)
# summary(model, (3, 48,48))