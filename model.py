

import torch
from torch import nn
import parameters as p
import torchsummary


class GeneratorLayer(nn.Module):
    '''
        notice : activation has tanh or relu optional.The activation will be Tanh If last_layer is True,else relu.
    '''

    def __init__(self,
                 in_channels: int, out_channels: int,
                 last_layer=False,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, stride=2, kernel_size=4, padding=1, bias=False)

        self.norm = nn.Identity() if last_layer else nn.BatchNorm2d(
            out_channels)

        self.act = nn.Tanh() if last_layer else nn.ReLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class DiscriminatorLayer(nn.Module):
    def __init__(self,
                 in_channels: int, out_channels: int,
                 kernel_size: int, stride: int,
                 padding:int=1,
                 use_norm = True,
                 use_act = True,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,padding=padding ,bias=False)
        
        factor = int(out_channels / p.FILTER_BASE_LINE)

        size = int(32 / factor if padding != 0 else 1)

        self.norm = nn.LayerNorm(
            [out_channels,size,size]) if use_norm else nn.Identity()

        self.act = nn.LeakyReLU(0.2) if use_act else nn.Identity() 

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class Discriminator(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(
            DiscriminatorLayer(in_channels=3, out_channels=p.FILTER_BASE_LINE, kernel_size=4, stride=2,
                               use_norm=False),#32
            DiscriminatorLayer(in_channels=p.FILTER_BASE_LINE, out_channels=p.FILTER_BASE_LINE*2, kernel_size=4, stride=2,
                               use_act=True),#16
            DiscriminatorLayer(in_channels=p.FILTER_BASE_LINE*2, out_channels=p.FILTER_BASE_LINE*4, kernel_size=4, stride=2,
                               use_act=True),#8
            DiscriminatorLayer(in_channels=p.FILTER_BASE_LINE*4, out_channels=p.FILTER_BASE_LINE*8, kernel_size=4, stride=2,
                               use_act=True),#4
            DiscriminatorLayer(in_channels=p.FILTER_BASE_LINE*8, out_channels=1, kernel_size=4, stride=2,
                               padding=0,use_norm=False,use_act=False,)#1

        )

    def forward(self, x):
        '''
        input of size [b,3,64,64]
        '''
        return self.model(x)

class Generator(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(
            nn.ConvTranspose2d(100, p.FILTER_BASE_LINE*8, 4, 1, 0, bias=False),#4
            
            GeneratorLayer(in_channels=p.FILTER_BASE_LINE*8, out_channels=p.FILTER_BASE_LINE*4,
                           last_layer=False),#8
            GeneratorLayer(in_channels=p.FILTER_BASE_LINE*4, out_channels=p.FILTER_BASE_LINE*2,
                           last_layer=False),#16
            GeneratorLayer(in_channels=p.FILTER_BASE_LINE*2, out_channels=p.FILTER_BASE_LINE,
                           last_layer=False),#32
            GeneratorLayer(in_channels=p.FILTER_BASE_LINE, out_channels=3,
                           last_layer=True)#64
        )

    def forward(self, x):
        '''
        input : a tensor of shape [b,100,1,1]
        '''
        return self.model(x)

class Discriminator128(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(
            DiscriminatorLayer(in_channels=3, out_channels=p.FILTER_BASE_LINE, kernel_size=4, stride=2,
                               use_norm=False),#64
            DiscriminatorLayer(in_channels=p.FILTER_BASE_LINE, out_channels=p.FILTER_BASE_LINE*2, kernel_size=4, stride=2,
                               use_act=True),#32
            DiscriminatorLayer(in_channels=p.FILTER_BASE_LINE*2, out_channels=p.FILTER_BASE_LINE*4, kernel_size=4, stride=2,
                               use_act=True),#16
            DiscriminatorLayer(in_channels=p.FILTER_BASE_LINE*4, out_channels=p.FILTER_BASE_LINE*8, kernel_size=4, stride=2,
                               use_act=True),#8
            DiscriminatorLayer(in_channels=p.FILTER_BASE_LINE*8, out_channels=p.FILTER_BASE_LINE*16, kernel_size=4, stride=2,
                               use_act=True),#4
            DiscriminatorLayer(in_channels=p.FILTER_BASE_LINE*16, out_channels=1, kernel_size=4, stride=2,
                               padding=0,use_norm=False,use_act=False,)#1

        )

    def forward(self, x):
        '''
        input of size [b,3,64,64]
        '''
        return self.model(x)



class Generator128(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(
            nn.ConvTranspose2d(100, p.FILTER_BASE_LINE*16, 4, 1, 0, bias=False),#4
            
            GeneratorLayer(in_channels=p.FILTER_BASE_LINE*16, out_channels=p.FILTER_BASE_LINE*8,
                           last_layer=False),#8
            GeneratorLayer(in_channels=p.FILTER_BASE_LINE*8, out_channels=p.FILTER_BASE_LINE*4,
                           last_layer=False),#16
            GeneratorLayer(in_channels=p.FILTER_BASE_LINE*4, out_channels=p.FILTER_BASE_LINE*2,
                           last_layer=False),#32
            GeneratorLayer(in_channels=p.FILTER_BASE_LINE*2, out_channels=p.FILTER_BASE_LINE,
                           last_layer=False),#32
            GeneratorLayer(in_channels=p.FILTER_BASE_LINE, out_channels=3,
                           last_layer=True)#64
        )

    def forward(self, x):
        '''
        input : a tensor of shape [b,100,1,1]
        '''
        return self.model(x)









if __name__ == '__main__':
    # g = Generator()
    # z = torch.randn(size=[3, 100, 1, 1])
    # y = g(z)
    # print(y.shape)
    d = Discriminator()
    img = torch.randn(size=[3, 3, 64, 64])
    score = d(img)
    print(score)
