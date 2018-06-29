import torch.nn as nn
from models.abstract_model import EncoderDecoder

class Resnet(EncoderDecoder):
    def __init__(self, num_classes, num_channels, encoder_name):
        super().__init__(num_classes, num_channels, encoder_name)

    def get_encoder(self, encoder, layer):
        if layer == 0:
            return nn.Sequential(
                encoder.conv1,
                encoder.bn1,
                encoder.relu)
        elif layer == 1:
            return nn.Sequential(
                encoder.maxpool,
                encoder.layer1)
        elif layer == 2:
            return encoder.layer2
        elif layer == 3:
            return encoder.layer3
        elif layer == 4:
            return encoder.layer4


class Resnet34_upsample(Resnet):
    def __init__(self, num_classes, num_channels=3):
        super().__init__(num_classes, num_channels, encoder_name='resnet34')
        

class Resnet50(Resnet):
    def __init__(self, num_classes, num_channels=3):
        super().__init__(num_classes, num_channels, encoder_name='resnet50')


class MobilenetV2(EncoderDecoder):
    def __init__(self, num_classes, num_channels):
        super().__init__(num_classes, num_channels, encoder_name='mobilenetv2')

    def get_encoder(self, encoder, layer):
        if layer == 0:
            return encoder.block1
        if layer == 1:
            return encoder.block2
        if layer == 2:
            return encoder.block3
        if layer == 3:
            return encoder.block4
        if layer == 4:
            return encoder.block5

class InceptionResnetV2(EncoderDecoder):
    def __init__(self, num_classes, num_channels):
        super().__init__(num_classes, num_channels, encoder_name='inceptionresnetv2')
        
    def get_encoder(self, encoder, layer):
        encoder_blocks = ['conv2d_2b', 'conv2d_4a', 'repeat', 'repeat_1', 'conv2d_7b']
        blocks = []
        cur_block = nn.Sequential()
        for name, module in encoder.named_children():
            cur_block.add_module(name, module)
            if name in encoder_blocks:
                blocks.append(cur_block)
                cur_block = nn.Sequential()
        assert len(cur_block) == 0, 'not all layer belong to encoder'
        
        return blocks[layer]