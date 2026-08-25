"""Network utilities used by SHASP.

This module intentionally contains only normalization, initialization, learning
rate scheduling, GAN loss, and PatchGAN.
"""

import functools

import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler


class Identity(nn.Module):
    def forward(self, tensor):
        return tensor


def get_norm_layer(norm_type='instance'):
    """Return a 2-D normalization-layer constructor."""
    if norm_type == 'batch':
        return functools.partial(
            nn.BatchNorm2d, affine=True, track_running_stats=True
        )
    if norm_type == 'instance':
        return functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False
        )
    if norm_type == 'none':
        return lambda _: Identity()
    raise NotImplementedError(
        'normalization layer [{}] is not supported'.format(norm_type)
    )


def get_scheduler(optimizer, opt):
    """Create the learning-rate scheduler selected by the training options."""
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            return 1.0 - max(
                0, epoch + opt.epoch_count - opt.n_epochs
            ) / float(opt.n_epochs_decay + 1)

        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    if opt.lr_policy == 'step':
        return lr_scheduler.StepLR(
            optimizer,
            step_size=opt.lr_decay_iters,
            gamma=getattr(opt, 'lr_gamma', 0.1),
        )
    if opt.lr_policy == 'plateau':
        return lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5
        )
    if opt.lr_policy == 'cosine':
        return lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=opt.n_epochs, eta_min=0
        )
    raise NotImplementedError(
        'learning-rate policy [{}] is not supported'.format(opt.lr_policy)
    )


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize convolutional and normalization layers."""
    def init_func(module):
        class_name = module.__class__.__name__
        has_weight = hasattr(module, 'weight') and module.weight is not None
        if has_weight and ('Conv' in class_name or 'Linear' in class_name):
            if init_type == 'normal':
                init.normal_(module.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(module.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(module.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(module.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    'initialization [{}] is not supported'.format(init_type)
                )
            if getattr(module, 'bias', None) is not None:
                init.constant_(module.bias.data, 0.0)
        elif 'BatchNorm2d' in class_name:
            init.normal_(module.weight.data, 1.0, init_gain)
            init.constant_(module.bias.data, 0.0)

    print('initialize network with {}'.format(init_type))
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=None):
    """Move a network to the requested device(s), then initialize it."""
    gpu_ids = gpu_ids or []
    if gpu_ids:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA GPU requested, but CUDA is unavailable')
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, init_gain)
    return net


def define_D(input_nc, ndf, netD='basic', n_layers_D=3, norm='instance',
             init_type='normal', init_gain=0.02, gpu_ids=None):
    """Create the image-domain PatchGAN discriminator."""
    norm_layer = get_norm_layer(norm)
    if netD == 'basic':
        n_layers = 3
    elif netD == 'n_layers':
        n_layers = n_layers_D
    else:
        raise NotImplementedError(
            'SHASP supports netD=basic or n_layers, got {}'.format(netD)
        )
    net = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer)
    return init_net(net, init_type, init_gain, gpu_ids)


class GANLoss(nn.Module):
    """GAN objectives operating directly on discriminator logits.

    ``enhance`` is the numerically stable form of the logarithmic adversarial
    objective reported in SHASP equations (7), (9), and (10). For generator
    updates, the usual non-saturating ``-log(D(fake))`` form is used to avoid
    the vanishing gradients of the literal minimax generator objective.
    """

    def __init__(self, gan_mode, target_real_label=1.0,
                 target_fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'enhance':
            self.loss = None
        else:
            raise NotImplementedError(
                'SHASP supports lsgan, vanilla, or enhance, got {}'.format(
                    gan_mode
                )
            )

    def get_target_tensor(self, prediction, target_is_real):
        label = self.real_label if target_is_real else self.fake_label
        return label.expand_as(prediction)

    def forward(self, prediction, target_is_real):
        if self.gan_mode == 'enhance':
            # softplus(-t) = -log(sigmoid(t));
            # softplus(t)  = -log(1 - sigmoid(t)).
            return (
                torch.nn.functional.softplus(-prediction).mean()
                if target_is_real
                else torch.nn.functional.softplus(prediction).mean()
            )
        return self.loss(
            prediction, self.get_target_tensor(prediction, target_is_real)
        )


class NLayerDiscriminator(nn.Module):
    """Fully convolutional PatchGAN discriminator."""

    def __init__(self, input_nc, ndf=64, n_layers=3,
                 norm_layer=nn.BatchNorm2d):
        super().__init__()
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kernel_size, padding = 4, 1
        layers = [
            nn.Conv2d(input_nc, ndf, kernel_size, 2, padding),
            nn.LeakyReLU(0.2, True),
        ]
        multiplier = 1
        for layer_index in range(1, n_layers):
            previous = multiplier
            multiplier = min(2 ** layer_index, 8)
            layers += [
                nn.Conv2d(
                    ndf * previous, ndf * multiplier, kernel_size, 2,
                    padding, bias=use_bias
                ),
                norm_layer(ndf * multiplier),
                nn.LeakyReLU(0.2, True),
            ]

        previous = multiplier
        multiplier = min(2 ** n_layers, 8)
        layers += [
            nn.Conv2d(
                ndf * previous, ndf * multiplier, kernel_size, 1,
                padding, bias=use_bias
            ),
            norm_layer(ndf * multiplier),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * multiplier, 1, kernel_size, 1, padding),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, image_or_feature):
        return self.model(image_or_feature)
