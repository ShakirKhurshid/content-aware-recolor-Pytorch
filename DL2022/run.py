from PIL import Image
import cv2
import numpy as np
from skimage.color import rgb2lab, lab2rgb
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from functools import partial
import argparse 

# initialize the logger
from logger import log_to_stdout
logger = log_to_stdout(level ='DEBUG')

def process_arguments():
    '''Collect the input argument's
	   Return a parser with the arguments
    '''
    parser = argparse.ArgumentParser(description = 'Input the image and the target palette')

    parser.add_argument(
        '-i',            
        '--input_image',
		type = str,
		required=True,
		default=None,
		help='path of the input image'
    )
    
    parser.add_argument(
        '-p',
        '--palette',
        nargs='+',
        default = [],
        help='The Target palette'
    )

    parser.add_argument(
        '-m',
        '--model',
        type = str,
        required=True,
        default=None,
        help='path for the model'
    )

    parser.add_argument(
        '-o',
        '--out_path',
        type = str,
        required=False,
        default='out.jpg',
        help='output_path'
    )

    return parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
device

class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (
        self.kernel_size[0] // 2, self.kernel_size[1] // 2)  # dynamic add padding based on the kernel_size


conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)


def activation_func(activation):
    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]


def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(conv(in_channels, out_channels, *args, **kwargs), nn.InstanceNorm2d(out_channels))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.blocks = nn.Identity()
        self.activate = activation_func(activation)
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = nn.Sequential(
            nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                      stride=self.downsampling, bias=False),
            nn.BatchNorm2d(self.expanded_channels)) if self.should_apply_shortcut else None

    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels


class ResNetBasicBlock(ResNetResidualBlock):
    """
    Basic ResNet block composed by two layers of 3x3conv/batchnorm/activation
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation_func(self.activation),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )


class ResNetLayer(nn.Module):
    """
    A ResNet layer composed by `n` blocks stacked one after the other
    """

    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1
        self.blocks = nn.Sequential(
            block(in_channels, out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion,
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x


class FeatureEncoder(nn.Module):
    def __init__(self):
        super(FeatureEncoder, self).__init__()

        # Convolutional blocks
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        logger.debug('self.conv1_1 : {}'.format(self.conv1_1.type))
        self.norm1_1 = nn.InstanceNorm2d(64)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Residual blocks
        self.res1 = ResNetLayer(64, 128, block=ResNetBasicBlock, n=1)
        self.res2 = ResNetLayer(128, 256, block=ResNetBasicBlock, n=1)
        self.res3 = ResNetLayer(256, 512, block=ResNetBasicBlock, n=1)

    def forward(self, x):
        x = F.relu(self.norm1_1(self.conv1_1(x)))
        c4 = self.pool1(x)
        c3 = self.res1(c4)
        c2 = self.res2(c3)
        c1 = self.res3(c2)
        return c1, c2, c3, c4


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.InstanceNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.InstanceNorm2d(out_channels),
    )


class RecoloringDecoder(nn.Module):
    # c => (bz, channel, h, w)
    # [Pt, c1]: (18 + 512) -> (256)
    # [c2, d1]: (256 + 256) -> (128)
    # [Pt, c3, d2]: (18 + 128 + 128) -> (64)
    # [Pt, c4, d3]: (18 + 64 + 64) -> 64
    # [Illu, d4]: (1 + 64) -> 3

    def __init__(self):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up_4 = double_conv(18 + 512, 256)
        self.dconv_up_3 = double_conv(256 + 256, 128)
        self.dconv_up_2 = double_conv(18 + 128 + 128, 64)
        self.dconv_up_1 = double_conv(18 + 64 + 64, 64)
        self.conv_last = nn.Conv2d(1 + 64, 3, 3, padding=1)

    def forward(self, c1, c2, c3, c4, target_palettes_1d, illu):
        logger.debug('target_Palette_1dshape: {} '.format(target_palettes_1d.shape))
        bz, h, w = c1.shape[0], c1.shape[2], c1.shape[3]
        target_palettes = torch.ones(bz, 18, h, w).float().to(device)
        target_palettes = target_palettes.reshape(h, w, bz * 18) * target_palettes_1d
        target_palettes = target_palettes.permute(2, 0, 1).reshape(bz, 18, h, w)
        logger.debug('target Palette shape: {} '.format(target_palettes.shape))

        # Concatenate target_palettes with c1
        logger.info('\n Concatenate target_palettes with C1')
        x = torch.cat((c1.float(), target_palettes.float()), 1)

        logger.debug('X shape: {} '.format(x.shape))
        x = self.dconv_up_4(x)
        logger.debug('X shape: {} '.format(x.shape))
        x = self.upsample(x)
        logger.debug('X shape: {} '.format(x.shape))


        # Concatenate c2 with x
        logger.info('\n Concatenate c2 with x')
        
        # Reshape x to match the shape of the content feature c2
        if not x.shape[2:] == c2.shape[2:]:
            x = F.interpolate(x, (c2.shape[2:]))
            logger.debug('X shape after interpolatiion: {} '.format(x.shape))

        x = torch.cat([c2, x], dim=1)
        logger.debug('X shape: {} '.format(x.shape))

        x = self.dconv_up_3(x)
        logger.debug('X shape: {} '.format(x.shape))

        x = self.upsample(x)
        logger.debug('X shape: {} '.format(x.shape))

        # concatenate target_palettes and c3 with x
        logger.info('\n Concatenate target_palettes and c3 with x')

        # Reshape x to match the shape of the content feature c2
        if not x.shape[2:] == c3.shape[2:]:
            x = F.interpolate(x, (c3.shape[2:]))
            logger.debug('X shape after interpolatiion: {} '.format(x.shape))

        bz, h, w = x.shape[0], x.shape[2], x.shape[3]
        target_palettes = torch.ones(bz, 18, h, w).float().to(device)
        target_palettes = target_palettes.reshape(h, w, bz * 18) * target_palettes_1d
        target_palettes = target_palettes.permute(2, 0, 1).reshape(bz, 18, h, w)
        logger.debug('target Palette shape: {} '.format(target_palettes.shape))


        x = torch.cat([target_palettes.float(), c3, x], dim=1)
        logger.debug('X shape: {} '.format(x.shape))
        x = self.dconv_up_2(x)
        logger.debug('X shape: {} '.format(x.shape))
        x = self.upsample(x)
        logger.debug('X shape: {} '.format(x.shape))

        # Concatenate target_palettes and c4 with x
        logger.info('\n Concatenate target_palettes and c4 with x')

        if not x.shape[2:] == c4.shape[2:]:
            x = F.interpolate(x, (c4.shape[2:]))
            logger.debug('X shape after interpolatiion: {} '.format(x.shape))

        bz, h, w = x.shape[0], x.shape[2], x.shape[3]
        target_palettes = torch.ones(bz, 18, h, w).float().to(device)
        target_palettes = target_palettes.reshape(h, w, bz * 18) * target_palettes_1d
        target_palettes = target_palettes.permute(2, 0, 1).reshape(bz, 18, h, w)
        logger.debug('target Palette shape: {} '.format(target_palettes.shape))


        x = torch.cat([target_palettes.float(), c4, x], dim=1)
        logger.debug('X shape: {} '.format(x.shape))
        x = self.dconv_up_1(x)
        logger.debug('X shape: {} '.format(x.shape))
        x = self.upsample(x)
        logger.debug('X shape: {} '.format(x.shape))
        illu = illu.view(illu.size(0), 1, illu.size(1), illu.size(2))
        logger.debug('illu shape: {} '.format(illu.shape))
        x = F.interpolate(x, (illu.shape[2:]))
        x = torch.cat((x, illu), dim=1)
        logger.debug('X shape: {} '.format(x.shape))
        x = self.conv_last(x)
        logger.debug('X shape: {} '.format(x.shape))
        return x


def pooling(image):
    m = nn.AdaptiveAvgPool2d((510,51))
    output = m(image)
    return output


def inputs(image_path, hex_codes):
    '''Input image preprocessing'''
    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.ToTensor(), ])

    image = Image.open(image_path)
    image = transform(np.array(image))
    #image = pooling(image)
    image = image.double()
    logger.debug('image double shape: {} '.format(image.shape))

    illu = get_illuminance(image)
    illu = illu.double()
    illu = illu[None, :, :]
    logger.debug('illu double: {}'.format(illu.shape))

    # Converting palette HEX Codes into RGB Values and the
    palette = hex_to_rgb(hex_codes)
    flat_palette = palette.flatten()

    return image, illu, flat_palette


def run_model(image, illu, palette, FE, RD):
    image = image[None, :, :]
    # Get content features via the Feature encoder
    c1, c2, c3, c4 = FE.forward(image.float().to(device))
    logger.debug('c1 shape: {} ,  c2 shape: {} , c3 shape: {} , c4 shape: {}'.format(c1.shape,c2.shape,c3.shape,c4.shape))
    # Run the decoder network
    out = RD.forward(c1, c2, c3, c4, palette.float().to(device), illu.float().to(device))
    logger.debug( 'out shape: {} '.format(out.shape))
    out = out.detach().cpu().numpy()
    out = out.squeeze()
    out = np.transpose(out, (1, 2, 0))
    out = cv2.convertScaleAbs(out, alpha=(255.0))
    out = Image.fromarray(out)
    return out

def get_illuminance(img):
    """
    Get the luminance of an image. Shape: (h, w)
    """
    logger.info('starting get_illumiance function')
    logger.debug('illu shape: {}'.format(img.shape))
    img = img.permute(1, 2, 0)  # (h, w, channel)
    logger.debug('illu shape: {}'.format(img.shape))
    img = img.numpy()
    img = img.astype(np.float) / 255.0
    img_LAB = rgb2lab(img)
    img_L = img_LAB[:, :, 0]  # luminance  # (h, w)
    logger.debug('iluu shape: {}'.format(img_L.shape))
    return torch.from_numpy(img_L)


def hex_to_rgb(hex_array):
    "Hex code to RGB convertor"
    logger.info('Starting hex_to_rgb Function .....')
    palette = []
    for hexcode in hex_array:
        rgb = np.array(list(int(hexcode.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4)))
        palette.append(rgb)
    palette = np.array(palette)[np.newaxis, :, :]
    logger.debug('palette: {}'.format(palette.shape))
    palette = palette[:, :6, :].ravel() / 255.0
    palette = torch.from_numpy(palette).double()
    logger.debug('palette: {}'.format(palette.shape))
    palette = palette[None, :]
    logger.debug('palette: {}'.format(palette.shape))

    return palette

def main():
    # Process the arguments
    input_arguments = process_arguments()
    # Get the palette
    colors = input_arguments.palette
    # Get the input image
    input_image = input_arguments.input_image
    # Get the model
    state = torch.load(input_arguments.model)
    FE = FeatureEncoder().float().to(device)
    RD = RecoloringDecoder().float().to(device)
    FE.load_state_dict(state['FE'])
    RD.load_state_dict(state['RD'])
    # Preprocessing
    image , illu, palette = inputs(input_image, colors)
    # FE + RE pass
    out = run_model(image , illu, palette, FE, RD)
    out.save(input_arguments.out_path)

if __name__== '__main__':
    main()


