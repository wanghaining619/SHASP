"""SHASP networks for visible-to-infrared image generation.

The generator separates a spatial, modality-invariant structure map from a
compact modality-specific vector. A residual vector mapper predicts the target
modality representation using source-only information, and an IR-conditioned
tone adapter prevents visible reflectance from being copied directly as
thermal intensity.
"""

import functools
import math

import torch
import torch.nn as nn

from . import networks


def _use_bias(norm_layer):
    if isinstance(norm_layer, functools.partial):
        return norm_layer.func == nn.InstanceNorm2d
    return norm_layer == nn.InstanceNorm2d


class ResidualBlock(nn.Module):
    def __init__(self, channels, norm_layer, use_dropout=False):
        super().__init__()
        bias = _use_bias(norm_layer)
        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=bias),
            norm_layer(channels),
            nn.ReLU(True),
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        layers += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=bias),
            norm_layer(channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, feature):
        return feature + self.model(feature)


class SpatialEncoder(nn.Module):
    """7x7 feature stem followed by two 2x downsampling stages."""

    def __init__(self, input_nc, ngf, norm_layer):
        super().__init__()
        bias = _use_bias(norm_layer)
        self.model = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, 7, bias=bias),
            norm_layer(ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, ngf * 2, 3, 2, 1, bias=bias),
            norm_layer(ngf * 2),
            nn.ReLU(True),
            nn.Conv2d(ngf * 2, ngf * 4, 3, 2, 1, bias=bias),
            norm_layer(ngf * 4),
            nn.ReLU(True),
        )

    def forward(self, image):
        return self.model(image)


class StructureContentEncoder(nn.Module):
    """Modality-specific shallow branches and three shared residual blocks."""

    def __init__(self, vis_nc, ir_nc, ngf, norm_layer):
        super().__init__()
        self.vis_branch = SpatialEncoder(vis_nc, ngf, norm_layer)
        self.ir_branch = SpatialEncoder(ir_nc, ngf, norm_layer)
        self.shared_layers = nn.Sequential(*[
            ResidualBlock(ngf * 4, norm_layer) for _ in range(3)
        ])

    def encode_vis(self, image):
        return self.shared_layers(self.vis_branch(image))

    def encode_ir(self, image):
        return self.shared_layers(self.ir_branch(image))


class GlobalSpecificEncoder(nn.Module):
    """Encode modality attributes as a compact global vector."""

    def __init__(self, input_nc, ngf, specific_dim, norm_layer):
        super().__init__()
        bias = _use_bias(norm_layer)
        self.feature_extractor = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, 7, bias=bias),
            norm_layer(ngf),
            nn.ReLU(True),
            nn.ReflectionPad2d(2),
            nn.Conv2d(ngf, ngf * 2, 5, 2, bias=bias),
            norm_layer(ngf * 2),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 2, ngf * 4, 3, 2, bias=bias),
            norm_layer(ngf * 4),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.dynamic_mapping = nn.Linear(ngf * 4, specific_dim)

    def reset_mapping(self):
        if self.dynamic_mapping.in_features == self.dynamic_mapping.out_features:
            nn.init.eye_(self.dynamic_mapping.weight)
            nn.init.zeros_(self.dynamic_mapping.bias)

    def forward(self, image):
        return self.dynamic_mapping(
            self.feature_extractor(image).flatten(1)
        )


class ResidualMLPBlock(nn.Module):
    """Residual block operating on a global modality vector."""

    def __init__(self, channels, use_dropout=False):
        super().__init__()
        layers = [
            nn.Linear(channels, channels),
            nn.LayerNorm(channels),
            nn.ReLU(True),
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        layers += [
            nn.Linear(channels, channels),
            nn.LayerNorm(channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, vector):
        return vector + self.model(vector)


class CrossModalitySpecificMapper(nn.Module):
    """Predict a target-specific vector from source shared/specific features."""

    def __init__(self, shared_nc, specific_dim, hidden_dim, n_blocks=3,
                 use_dropout=False):
        super().__init__()
        self.shared_pool = nn.AdaptiveAvgPool2d(1)
        self.feature_body = nn.Sequential(
            nn.Linear(shared_nc + specific_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(True),
            *[
                ResidualMLPBlock(hidden_dim, use_dropout)
                for _ in range(n_blocks)
            ],
        )
        self.output_linear = nn.Linear(hidden_dim, specific_dim)

    def reset_to_identity(self):
        nn.init.zeros_(self.output_linear.weight)
        nn.init.zeros_(self.output_linear.bias)

    def forward(self, shared, source_specific):
        shared_global = self.shared_pool(shared).flatten(1)
        correction = self.output_linear(
            self.feature_body(torch.cat((shared_global, source_specific), 1))
        )
        return source_specific + correction


class ReconstructionDecoder(nn.Module):
    """Broadcast and fuse a global specific vector with a spatial map."""

    def __init__(self, output_nc, ngf, specific_dim, norm_layer, n_blocks=6,
                 use_dropout=False):
        super().__init__()
        bias = _use_bias(norm_layer)
        latent_nc = ngf * 4
        self.body = nn.Sequential(
            nn.Conv2d(latent_nc + specific_dim, latent_nc, 1, bias=bias),
            # Immediate InstanceNorm would remove the spatially constant
            # contribution of the broadcast global vector.
            nn.Identity(),
            nn.ReLU(True),
            *[
                ResidualBlock(latent_nc, norm_layer, use_dropout)
                for _ in range(n_blocks)
            ],
            nn.ConvTranspose2d(
                latent_nc, ngf * 2, 3, 2, 1, output_padding=1, bias=bias
            ),
            norm_layer(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                ngf * 2, ngf, 3, 2, 1, output_padding=1, bias=bias
            ),
            norm_layer(ngf),
            nn.ReLU(True),
            nn.ReflectionPad2d(3),
        )
        self.output_conv = nn.Conv2d(ngf, output_nc, 7)

    def forward(self, shared, specific):
        if specific.ndim != 2:
            raise ValueError(
                'specific representation must be [N, D], got {}'.format(
                    tuple(specific.shape)
                )
            )
        specific_map = specific[:, :, None, None].expand(
            -1, -1, shared.size(2), shared.size(3)
        )
        return self.output_conv(
            self.body(torch.cat((shared, specific_map), dim=1))
        )


class ThermalToneAdapter(nn.Module):
    """Predict visible-base gain and bias from the target IR vector."""

    min_gain = 0.05
    max_gain = 1.00
    max_abs_bias = 0.50

    def __init__(self, specific_dim, hidden_dim=64, initial_gain=0.5):
        super().__init__()
        if not self.min_gain < initial_gain < self.max_gain:
            raise ValueError(
                'initial thermal base gain must be in ({}, {}), got {}'
                .format(self.min_gain, self.max_gain, initial_gain)
            )
        self.initial_gain = float(initial_gain)
        self.feature_body = nn.Sequential(
            nn.Linear(specific_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(True),
        )
        self.output_linear = nn.Linear(hidden_dim, 2)

    def reset_to_initial_tone(self):
        nn.init.zeros_(self.output_linear.weight)
        probability = (
            (self.initial_gain - self.min_gain)
            / (self.max_gain - self.min_gain)
        )
        gain_logit = math.log(probability / (1.0 - probability))
        with torch.no_grad():
            self.output_linear.bias[0] = gain_logit
            self.output_linear.bias[1] = 0.0

    def forward(self, predicted_ir_specific):
        parameters = self.output_linear(
            self.feature_body(predicted_ir_specific)
        )
        gain = self.min_gain + (
            self.max_gain - self.min_gain
        ) * torch.sigmoid(parameters[:, 0:1])
        bias = self.max_abs_bias * torch.tanh(parameters[:, 1:2])
        return gain[:, :, None, None], bias[:, :, None, None]


class SHASPGenerator(nn.Module):
    """Bidirectional SHASP generator."""

    def __init__(self, vis_nc=3, ir_nc=1, ngf=64,
                 norm_layer=nn.InstanceNorm2d, use_dropout=False,
                 decoder_blocks=6, specific_dim=256, mapper_blocks=3,
                 tone_hidden_dim=64, thermal_base_gain=0.5):
        super().__init__()
        self.vis_nc = vis_nc
        self.ir_nc = ir_nc
        shared_nc = ngf * 4
        mapper_hidden_dim = max(ngf * 2, specific_dim // 2)

        self.structure_encoder = StructureContentEncoder(
            vis_nc, ir_nc, ngf, norm_layer
        )
        self.vis_specific_encoder = GlobalSpecificEncoder(
            vis_nc, ngf, specific_dim, norm_layer
        )
        self.ir_specific_encoder = GlobalSpecificEncoder(
            ir_nc, ngf, specific_dim, norm_layer
        )
        self.vis_to_ir_specific_mapper = CrossModalitySpecificMapper(
            shared_nc, specific_dim, mapper_hidden_dim, mapper_blocks,
            use_dropout
        )
        self.ir_to_vis_specific_mapper = CrossModalitySpecificMapper(
            shared_nc, specific_dim, mapper_hidden_dim, mapper_blocks,
            use_dropout
        )
        self.vis_decoder = ReconstructionDecoder(
            vis_nc, ngf, specific_dim, norm_layer, decoder_blocks,
            use_dropout
        )
        self.ir_decoder = ReconstructionDecoder(
            ir_nc, ngf, specific_dim, norm_layer, decoder_blocks,
            use_dropout
        )
        self.thermal_tone_adapter = ThermalToneAdapter(
            specific_dim, tone_hidden_dim, thermal_base_gain
        )

    @staticmethod
    def _gray(image):
        if image.size(1) == 1:
            return image
        return (
            image[:, 0:1] * 0.299
            + image[:, 1:2] * 0.587
            + image[:, 2:3] * 0.114
        )

    @staticmethod
    def _residual_output(base, residual):
        base = base.clamp(-0.999, 0.999)
        return torch.tanh(torch.atanh(base) + residual)

    def reset_translation_paths(self):
        self.vis_specific_encoder.reset_mapping()
        self.ir_specific_encoder.reset_mapping()
        self.vis_to_ir_specific_mapper.reset_to_identity()
        self.ir_to_vis_specific_mapper.reset_to_identity()
        self.thermal_tone_adapter.reset_to_initial_tone()

    def translate_vis_to_ir(self, vis, return_features=False):
        shared_vis = self.structure_encoder.encode_vis(vis)
        specific_vis = self.vis_specific_encoder(vis)
        predicted_specific_ir = self.vis_to_ir_specific_mapper(
            shared_vis, specific_vis
        )
        residual = self.ir_decoder(shared_vis, predicted_specific_ir)
        tone_gain, tone_bias = self.thermal_tone_adapter(
            predicted_specific_ir
        )
        visible_base = self._gray(vis).clamp(-0.999, 0.999)
        output = torch.tanh(
            tone_gain * torch.atanh(visible_base) + tone_bias + residual
        )
        if return_features:
            return output, shared_vis, predicted_specific_ir, specific_vis
        return output

    def translate_ir_to_vis(self, ir, return_features=False):
        shared_ir = self.structure_encoder.encode_ir(ir)
        specific_ir = self.ir_specific_encoder(ir)
        predicted_specific_vis = self.ir_to_vis_specific_mapper(
            shared_ir, specific_ir
        )
        residual = self.vis_decoder(shared_ir, predicted_specific_vis)
        base = self._gray(ir).expand(-1, self.vis_nc, -1, -1)
        output = self._residual_output(base, residual)
        if return_features:
            return output, shared_ir, predicted_specific_vis, specific_ir
        return output

    def forward(self, image, direction='vis_to_ir', return_features=False):
        if direction == 'vis_to_ir':
            return self.translate_vis_to_ir(image, return_features)
        if direction == 'ir_to_vis':
            return self.translate_ir_to_vis(image, return_features)
        raise ValueError('unknown SHASP direction: {}'.format(direction))


class SemanticStructureDiscriminator(nn.Module):
    """Patch discriminator applied to spatial shared representations."""

    def __init__(self, feature_nc, ndf=64, n_layers=3,
                 norm_layer=nn.InstanceNorm2d):
        super().__init__()
        self.discriminator = networks.NLayerDiscriminator(
            feature_nc, ndf, n_layers=n_layers, norm_layer=norm_layer
        )

    def forward(self, shared_feature):
        return self.discriminator(shared_feature)


def define_shasp_G(vis_nc, ir_nc, ngf=64, norm='instance',
                   use_dropout=False, decoder_blocks=6,
                   specific_dim=256, mapper_blocks=3,
                   tone_hidden_dim=64, thermal_base_gain=0.5,
                   init_type='normal', init_gain=0.02, gpu_ids=None):
    norm_layer = networks.get_norm_layer(norm)
    net = SHASPGenerator(
        vis_nc, ir_nc, ngf, norm_layer, use_dropout, decoder_blocks,
        specific_dim, mapper_blocks, tone_hidden_dim, thermal_base_gain
    )
    net = networks.init_net(net, init_type, init_gain, gpu_ids or [])
    module = net.module if isinstance(net, torch.nn.DataParallel) else net
    module.reset_translation_paths()

    # A from-scratch model starts from a compressed visible base rather than
    # random image noise. Loading a checkpoint replaces these output weights.
    nn.init.zeros_(module.vis_decoder.output_conv.weight)
    nn.init.zeros_(module.vis_decoder.output_conv.bias)
    nn.init.zeros_(module.ir_decoder.output_conv.weight)
    nn.init.zeros_(module.ir_decoder.output_conv.bias)
    return net


def define_structure_D(feature_nc, ndf=64, n_layers=3, norm='instance',
                       init_type='normal', init_gain=0.02, gpu_ids=None):
    norm_layer = networks.get_norm_layer(norm)
    net = SemanticStructureDiscriminator(
        feature_nc, ndf, n_layers, norm_layer
    )
    return networks.init_net(net, init_type, init_gain, gpu_ids or [])
