import torch
from torch import nn
import torch.nn.functional as F

from models.inceptionresnetv2 import inceptionresnetv2


# last modules in the blocks (a block has a constant spatial size)
encoder_last_modules = ['conv2d_2b', 'conv2d_4a', 'repeat', 'repeat_1', 'conv2d_7b']
# number of channels at the end of each encoder block
encoder_channels = [64, 192, 320, 1088, 1536]
assert len(encoder_last_modules) == len(encoder_channels)


def get_encoder_blocks(model):
    blocks = []
    cur_block = nn.Sequential()
    for name, module in model.named_children():
        cur_block.add_module(name, module)
        if name in encoder_last_modules:
            blocks.append(cur_block)
            cur_block = nn.Sequential()
        if len(blocks) == len(encoder_last_modules):
            break
    return blocks


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, inputs):
        return self.seq(inputs)


class UNetUp(nn.Module):
    def __init__(self, bottom_channels, left_channels, right_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(bottom_channels, right_channels,
                                     kernel_size=3, stride=2)
        self.seq1 = ConvBNAct(left_channels + right_channels, right_channels)
        self.seq2 = ConvBNAct(right_channels, right_channels)

    def forward(self, left, bottom):
        from_bottom = self.up(bottom)
        result = self.seq1(torch.cat([left, from_bottom], 1))
        result = self.seq2(result)
        return result


class UNetUp2(nn.Module):
    def __init__(self, bottom_channels, left_channels, right_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(bottom_channels, left_channels, kernel_size=3, stride=2)
        self.seq1 = ConvBNAct(left_channels, left_channels)
        self.seq2 = ConvBNAct(left_channels, right_channels)

    def forward(self, left, bottom):
        from_bottom = self.up(bottom)
        result = self.seq1(left + from_bottom)
        result = self.seq2(result)
        return result


class UNetInceptionResnetV2(nn.Module):
    def __init__(self, encoder_blocks, num_classes, out_logits=False, up_block=UNetUp):
        super().__init__()
        self.num_classes = num_classes
        self.out_logits = out_logits
        self.depth = len(encoder_blocks)
        self.encoder_blocks = nn.ModuleList(encoder_blocks)
        self.decoder_blocks = nn.ModuleList()
        for i in range(1, self.depth):
            bottom_channels = encoder_channels[-i]
            left_channels = encoder_channels[-(i + 1)]
            self.decoder_blocks.append(up_block(bottom_channels, left_channels, left_channels))
        # this module's left input is the expanded input
        self.decoder_blocks.append(UNetUp(encoder_channels[0], 32, 64))
        self.expand_input = ConvBNAct(3, 32)
        self.conv_out = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        iw, ih = x.shape[-2:]
        assert iw >= 95 and iw % 32 == 31, 'x width must have form 31 + 32 * k, k >= 2'
        assert ih >= 95 and ih % 32 == 31, 'x height must have form 31 + 32 * k, k >= 2'

        expand_input = self.expand_input(x)
        encoder_outputs = [expand_input]
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            encoder_outputs.append(x)

        for i, decoder_block in enumerate(self.decoder_blocks):
            left_x = encoder_outputs[-(i + 2)]
            x = decoder_block(left_x, x)

        x = self.conv_out(x)
        
        if self.out_logits or self.num_classes == 1:
            return x

        return F.log_softmax(x, dim=1)


def unet_inceptionresnetv2(num_classes, pretrained=True, up_block=UNetUp, **kwargs):
    model = inceptionresnetv2(1, pretrained=pretrained, **kwargs)
    encoder_blocks = get_encoder_blocks(model)
    unet = UNetInceptionResnetV2(encoder_blocks, num_classes, up_block=up_block)
    return unet.cuda()
