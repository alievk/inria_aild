import os
import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from models import resnet, mobilenetv2, inceptionresnetv2


std_encoder_channels = [128, 128, 128, 256, 256]
encoder_params = {
    'mobilenetv2': {
        'channels': [16, 24, 32, 96, 1280],
        'init_op': mobilenetv2.mobilenetv2,
        'url': './snapshots/mobilenetv2_718_b.pth.tar'
    },
    'resnet34': {
        'channels': [64, 64, 128, 256, 512],
        'init_op': resnet.resnet34,
        'url': resnet.model_urls['resnet34']
    },
    'resnet50': {
        'channels': [64, 64, 128, 256, 512],
        'init_op': resnet.resnet50,
        'url': resnet.model_urls['resnet50']
    },
    'inceptionresnetv2': {
        'channels': [64, 192, 320, 1088, 1536],
        'init_op': inceptionresnetv2.inceptionresnetv2,
        'url': 'http://data.lip6.fr/cadene/pretrainedmodels/inceptionresnetv2-520b38e4.pth'
    }
}


class ConvBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, dec, enc):
        x = torch.cat([dec, enc], dim=1)
        return self.seq(x)


class PlusBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

    def forward(self, dec, enc):
        return enc + dec


class UnetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)
    

class UnetDecoderBlock_k3s2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, stride=2),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)


class AbstractModel(nn.Module):
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                # Kaiming He normal initialization
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def initialize_encoder(self, model, model_url):
        if 'http' in model_url:
            pretrained_dict = model_zoo.load_url(model_url)
        else:
            dirname = os.path.dirname(__file__)
            pretrained_dict = torch.load(os.path.join(dirname, model_url))
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model.load_state_dict(pretrained_dict)


def _get_layers_params(layers):
    return sum((list(l.parameters()) for l in layers), [])


class EncoderDecoder(AbstractModel):
    def __init__(self, num_classes, num_channels=3, encoder_name='resnet34',
                 bottleneck_type=ConvBottleneck, upsample_type=UnetDecoderBlock):
        super().__init__()
        self.encoder_channels = std_encoder_channels
        self.upsample_type = upsample_type
        n_enc = len(self.encoder_channels)
        
        if encoder_params[encoder_name]['channels'] != self.encoder_channels:
            self.encoder_squeeze = nn.ModuleList()
            for idx in range(n_enc):
                pre_ch = encoder_params[encoder_name]['channels'][idx]
                ch = self.encoder_channels[idx]
                self.encoder_squeeze.append(nn.Conv2d(pre_ch, ch, 1))

        self.bottlenecks = nn.ModuleList()
        for ch in reversed(self.encoder_channels[:-1]):
            self.bottlenecks.append(bottleneck_type(ch * 2, ch))

        self.decoder_stages = nn.ModuleList()
        for idx in range(1, n_enc):
            self.decoder_stages.append(self.get_decoder(idx))

        self.last_upsample = upsample_type(self.encoder_channels[0], self.encoder_channels[0] // 2)
        self.final = self.make_final_classifier(self.encoder_channels[0] // 2, num_classes)

        self._initialize_weights()

        encoder = encoder_params[encoder_name]['init_op']()
        if num_channels == 3 and encoder_params[encoder_name]['url'] is not None:
            self.initialize_encoder(encoder, encoder_params[encoder_name]['url'])
        
        self.encoder_stages = nn.ModuleList()
        for idx in range(n_enc):
            encoder_stage = self.get_encoder(encoder, idx)
            self.encoder_stages.append(encoder_stage)

    # noinspection PyCallingNonCallable
    def forward(self, x):
        # Encoder
        enc_results = []
        for idx, stage in enumerate(self.encoder_stages):
            x = stage(x)
            if idx < len(self.encoder_stages) - 1:
                _x = x.clone()
                if hasattr(self, 'encoder_squeeze'):
                    _x = self.encoder_squeeze[idx](_x)
                enc_results.append(_x)

        if hasattr(self, 'encoder_squeeze'):
            x = self.encoder_squeeze[-1](x)

        for idx, bottleneck in enumerate(self.bottlenecks):
            rev_idx = - (idx + 1)
            x = self.decoder_stages[rev_idx](x)
            x = bottleneck(x, enc_results[rev_idx])

        x = self.last_upsample(x)
        f = self.final(x)

        return f

    def get_decoder(self, layer):
        return self.upsample_type(self.encoder_channels[layer], 
                                  self.encoder_channels[max(layer - 1, 0)])

    def make_final_classifier(self, in_filters, num_classes):
        return nn.Sequential(
            nn.Conv2d(in_filters, num_classes, 3, padding=1)
        )

    def get_encoder(self, encoder, layer):
        raise NotImplementedError

    @property
    def first_layer_params_names(self):
        raise NotImplementedError
        
    def summary(self):
        def prm_count(module):
            return sum(p.numel() for p in module.parameters())
        
        print('child module, million params')
        total = 0
        for name, child in self.named_children():
            pc = prm_count(child) / 1e6
            total += pc
            print(f'{name}: {pc:.3f}')
        print(f'---\ntotal: {total:.3f}')
