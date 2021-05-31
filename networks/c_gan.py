import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.utils import spectral_norm as SpectralNorm
from networks import safa

def init_weights(net, init_type='normal', init_gain=0.02):

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_R(ret_method, polar, gpu_ids=[]):
    net = None
    if 'SAFA' == ret_method:
        sa_num = 8
        sate_size = (112, 616) if polar else (256, 256)
        pano_size = (112, 616)
        net = safa.SAFA(sa_num=sa_num, H1=sate_size[0], W1=sate_size[1], H2=pano_size[0], W2=pano_size[1])
    else:
        raise NotImplementedError('Retrieval model name [%s] is not recognized' % ret_method)
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    return net


def define_G(netG, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    if netG == 'unet-skip':
        net = UnetGeneratorSkip()
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_c, output_c, ndf, netD, condition, n_layers_D=3, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    if condition ==1:
        input_c_tmp = input_c
    else:
        input_c_tmp = 0
    if netD == 'basic':
        net = defineD_basic(input_c_tmp, output_c, ndf)
    elif netD == 'n_layers':
        net = NLayerDiscriminator(input_c_tmp, output_c, ndf, n_layers_D)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


def defineD_basic(input_c, output_c, ndf):
    n_layers=3
    return NLayerDiscriminator(input_c, output_c, ndf, n_layers)

class GANLoss(nn.Module):

    def __init__(self, gan_mode, target_real_label=0.9, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp', 'hinge']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):

        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        elif self.gan_mode == 'hinge':
            if target_is_real:
                loss = torch.nn.ReLU()(1.0 - prediction).mean()
            else:
                loss = torch.nn.ReLU()(1.0 + prediction).mean()
        return loss


#####GENERATOR CLASSES#####
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None):
        super(ConvLayer, self).__init__()
        if padding is None:
            padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.conv2d = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride))

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

##### non-local block #####
class Attention(nn.Module):
  def __init__(self, ch):
    super(Attention, self).__init__()

    self.ch = ch
    self.theta = SpectralNorm(nn.Conv2d(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False))
    self.phi = SpectralNorm(nn.Conv2d(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False))
    self.g = SpectralNorm(nn.Conv2d(self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False))
    self.o = SpectralNorm(nn.Conv2d(self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False))

    self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)
  def forward(self, x, y=None):

    theta = self.theta(x)
    phi = F.max_pool2d(self.phi(x), [2,2])
    g = F.max_pool2d(self.g(x), [2,2])

    theta = theta.view(-1, self. ch // 8, x.shape[2] * x.shape[3])
    phi = phi.view(-1, self. ch // 8, phi.shape[2]*phi.shape[3])
    g = g.view(-1, self. ch // 2, g.shape[2] * g.shape[3])

    beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)

    o = self.o(torch.bmm(g, beta.transpose(1,2)).view(-1, self.ch // 2, x.shape[2], x.shape[3]))
    return self.gamma * o + x

class UnetGeneratorSkip(nn.Module):

    def __init__(self, input_nc=3):
        super(UnetGeneratorSkip, self).__init__()

        self.in_dim = input_nc
        input_ch  = input_nc

        self.begin_pad = nn.ReflectionPad2d(3)
        self.begin_conv = SpectralNorm(nn.Conv2d(input_ch, 32, kernel_size=7, padding=0))
        self.begini_e = nn.InstanceNorm2d(32, affine=True)
        self.conv1 = ResidualBlockDown(32, 64)
        self.in1_e = nn.InstanceNorm2d(64, affine=True)
        self.conv2 = ResidualBlockDown(64, 128)
        self.in2_e = nn.InstanceNorm2d(128, affine=True)
        self.conv3 = ResidualBlockDown(128, 256)
        self.in3_e = nn.InstanceNorm2d(256, affine=True)

        self.res1 = ResidualBlock(256)
        self.res2 = ResidualBlock(256)
        self.res3 = ResidualBlock(256)
        self.res4 = ResidualBlock(256)
        self.res5 = ResidualBlock(256)
        self.res6 = ResidualBlock(256)

        self.attention2 = Attention(256)
        self.deconv3 = ResidualBlockUp(512, 128, upsample=2)
        self.in3_d = nn.InstanceNorm2d(128, affine=True)
        self.deconv2 = ResidualBlockUp(256, 64, upsample=2)
        self.in2_d = nn.InstanceNorm2d(64, affine=True)
        self.deconv1 = ResidualBlockUp(128, 64, upsample=2)
        self.in1_d = nn.InstanceNorm2d(64, affine=True)

        self.conv_end = nn.Sequential(nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1), nn.Tanh())

    def forward(self, x):

        x = self.begin_conv(self.begin_pad(x))
        x = self.begini_e(x)
        enc1 = self.in1_e(self.conv1(x))
        enc2 = self.in2_e(self.conv2(enc1))
        enc3 = self.in3_e(self.conv3(enc2))

        # Residual layers
        resb1 = self.res1(enc3)
        resb2 = self.res2(resb1)
        resb3 = self.res3(resb2)
        resb4 = self.res4(resb3)
        resb5 = self.res5(resb4)
        resb6 = self.res6(resb5)

        dec1 = torch.cat((resb6, enc3), 1)
        dec2 = self.in3_d(self.deconv3(dec1))
        dec2 = torch.cat((dec2, enc2), 1)
        dec2att = self.attention2(dec2)
        dec3 = self.in2_d(self.deconv2(dec2att))
        dec3 = torch.cat((dec3, enc1), 1)
        dec4 = self.in1_d(self.deconv1(dec3))

        out = self.conv_end(dec4)
        return out, resb6

#####Residual Block Down#####
class ResidualBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None):
        super(ResidualBlockDown, self).__init__()
        # Right Side
        self.conv_r1 = ConvLayer(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_r2 = ConvLayer(out_channels, out_channels, kernel_size, stride, padding)

        # Left Side
        self.conv_l = ConvLayer(in_channels, out_channels, 1, 1)
    def forward(self, x):
        residual = x

        out = F.relu(x)
        out = self.conv_r1(out)
        out = F.relu(out)
        out = self.conv_r2(out)
        out = F.avg_pool2d(out, 2)

        residual = self.conv_l(residual)
        residual = F.avg_pool2d(residual, 2)
        out = residual + out
        return out

#####Residual Block Up#####
class ResidualBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, upsample=None, out_spatial=None):
        super(ResidualBlockUp, self).__init__()
        if upsample != None:
            self.upsample = nn.Upsample(scale_factor=upsample, mode='nearest')
        elif out_spatial !=None:
            self.upsample = nn.Upsample(size=out_spatial, mode='nearest')

        # Right Side
        self.norm_r1 = nn.InstanceNorm2d(in_channels, affine=True)
        self.conv_r1 = ConvLayer(in_channels, out_channels, kernel_size, stride)

        self.norm_r2 = nn.InstanceNorm2d(out_channels, affine=True)
        self.conv_r2 = ConvLayer(out_channels, out_channels, kernel_size, stride)

        # Left Side
        self.conv_l = ConvLayer(in_channels, out_channels, 1, 1)

    def forward(self, x):
        residual = x

        out = self.norm_r1(x)
        out = F.relu(out)
        out = self.upsample(out)
        out = self.conv_r1(out)
        out = self.norm_r2(out)
        out = F.relu(out)
        out = self.conv_r2(out)

        residual = self.upsample(residual)
        residual = self.conv_l(residual)

        out = residual + out
        return out


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.in1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.in2(out)

        out = out + residual
        return out

#####DISCRIMINATOR CLASSES#####
class PixelDiscriminator(nn.Module):

    def __init__(self, input_c, output_c, ndf=64, norm_layer=nn.InstanceNorm2d):

        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.net = nn.Sequential(
            nn.Conv2d(input_c+output_c, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias),
            nn.Sigmoid())


    def forward(self, input):
        return self.net(input)


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_c, output_c, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if n_layers == 0:
            PixelDiscriminator(input_c, output_c, ndf)
        else:
            sequence = [nn.Conv2d(input_c+output_c, ndf, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.1, True)]
            nf_mult = 1
            nf_mult_prev = 1

            for n in range(1, n_layers):
                nf_mult_prev = nf_mult
                nf_mult = min(2 ** n, 8)
                if n==1:
                    sequence += [
                        SpectralNorm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1,
                                               bias=use_bias)),
                        nn.LeakyReLU(0.1, True),
                        Attention(128)
                ]
                else:
                    sequence += [
                        SpectralNorm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1, bias=use_bias)),
                        nn.LeakyReLU(0.1, True)
                ]

            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n_layers, 8)

            sequence += [
                SpectralNorm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1, bias=use_bias)),
                nn.LeakyReLU(0.1, True)
            ]

            sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]
            self.model = nn.Sequential(*sequence)

    def forward(self, input):
        x = self.model(input)
        return x