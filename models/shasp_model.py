"""Training wrapper for SHASP."""

import itertools
import os

import torch
import torch.nn.functional as F

from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from . import shasp_networks


class ShaspModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(
            dataset_mode='cross_spectral',
            norm='instance',
            no_dropout=True,
            gan_mode='lsgan',
            input_nc=3,
            output_nc=1,
            crop_size=256,
            load_size=256,
            preprocess='crop' if is_train else 'none',
            display_winsize=256,
        )
        parser.add_argument(
            '--decoder_blocks', type=int, default=6,
            help='residual blocks in each reconstruction decoder'
        )
        parser.add_argument(
            '--specific_dim', type=int, default=256,
            help='dimension of each global modality-specific vector'
        )
        parser.add_argument(
            '--mapper_blocks', type=int, default=3,
            help='residual MLP blocks in each cross-modal vector mapper'
        )
        parser.add_argument(
            '--tone_hidden_dim', type=int, default=64,
            help='hidden width of the IR-specific thermal tone adapter'
        )
        parser.add_argument(
            '--thermal_base_gain', type=float, default=0.5,
            help='initial VIS luminance gain in IR logit rendering'
        )
        if not is_train:
            parser.set_defaults(separate_visual_dirs=True)
        if is_train:
            parser.set_defaults(
                lr=1e-4, batch_size=1, n_epochs=100, n_epochs_decay=30,
                lr_policy='linear', pool_size=0
            )
            parser.add_argument('--lambda_semantic', type=float, default=0.5)
            parser.add_argument('--lambda_cycle', type=float, default=5.0)
            parser.add_argument('--lambda_adversarial', type=float, default=1.0)
            parser.add_argument('--lambda_specific', type=float, default=2.0)
            parser.add_argument('--lambda_paired', type=float, default=20.0)
            parser.add_argument('--lambda_gradient', type=float, default=5.0)
            parser.add_argument('--lambda_intensity', type=float, default=1.0)
            parser.add_argument('--lambda_bright', type=float, default=5.0)
            parser.add_argument(
                '--bright_threshold', type=float, default=0.8,
                help='VIS brightness threshold in [0,1] for correction loss'
            )
            parser.add_argument('--paired_warmup_epochs', type=int, default=5)
            parser.add_argument('--gan_ramp_epochs', type=int, default=5)
            parser.add_argument('--lr_gamma', type=float, default=0.5)
            parser.add_argument(
                '--warmstart_checkpoint', type=str, default='',
                help='optional compatible generator checkpoint or directory'
            )
            parser.add_argument(
                '--global_warmup_epochs', type=int, default=5,
                help='with warm-start, adapt global/fusion paths first'
            )
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        if self.isTrain:
            if not 0.0 <= opt.bright_threshold < 1.0:
                raise ValueError('--bright_threshold must be in [0, 1)')
            if opt.lambda_bright < 0.0:
                raise ValueError('--lambda_bright must be non-negative')

        self.loss_names = [
            'D_X', 'D_Y', 'D_str', 'adv_X', 'adv_Y', 'cycle', 'semantic',
            'specific', 'paired', 'gradient', 'intensity', 'bright'
        ]
        self.visual_names = [
            'real_A', 'fake_B', 'real_B', 'fake_A', 'rec_A', 'rec_B'
        ]
        self.model_names = ['G']
        self.initialized_from_warmstart = bool(
            self.isTrain and opt.warmstart_checkpoint
        )
        if self.isTrain:
            self.model_names += ['D_X', 'D_Y', 'D_str']

        self.netG = shasp_networks.define_shasp_G(
            opt.input_nc, opt.output_nc, opt.ngf, opt.norm,
            not opt.no_dropout, opt.decoder_blocks, opt.specific_dim,
            opt.mapper_blocks, opt.tone_hidden_dim, opt.thermal_base_gain,
            opt.init_type, opt.init_gain, self.gpu_ids
        )

        if self.isTrain:
            if opt.warmstart_checkpoint:
                self._load_warmstart_generator(opt.warmstart_checkpoint)

            self.netD_X = networks.define_D(
                opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm,
                opt.init_type, opt.init_gain, self.gpu_ids
            )
            self.netD_Y = networks.define_D(
                opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm,
                opt.init_type, opt.init_gain, self.gpu_ids
            )
            self.netD_str = shasp_networks.define_structure_D(
                opt.ngf * 4, opt.ndf, opt.n_layers_D, opt.norm,
                opt.init_type, opt.init_gain, self.gpu_ids
            )
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            self.optimizer_D = torch.optim.Adam(
                itertools.chain(
                    self.netD_X.parameters(), self.netD_Y.parameters(),
                    self.netD_str.parameters()
                ),
                lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            self.optimizers += [self.optimizer_G, self.optimizer_D]
        self.current_epoch = 1

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        if not self.isTrain or not self.initialized_from_warmstart:
            return
        global_only = epoch <= self.opt.global_warmup_epochs
        module = (
            self.netG.module
            if isinstance(self.netG, torch.nn.DataParallel) else self.netG
        )
        for parameter in module.parameters():
            parameter.requires_grad = not global_only
        warmup_modules = (
            module.vis_specific_encoder,
            module.ir_specific_encoder,
            module.vis_to_ir_specific_mapper,
            module.ir_to_vis_specific_mapper,
            module.vis_decoder.body[0],
            module.ir_decoder.body[0],
            module.thermal_tone_adapter,
        )
        for warmup_module in warmup_modules:
            for parameter in warmup_module.parameters():
                parameter.requires_grad = True
        if global_only:
            print(
                'SHASP global/fusion-path warmup: epoch {}/{}'
                .format(epoch, self.opt.global_warmup_epochs),
                flush=True
            )
        elif epoch == self.opt.global_warmup_epochs + 1:
            print('SHASP full generator unfrozen', flush=True)

    @staticmethod
    def _clean_checkpoint_state(state):
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        cleaned = {}
        for key, value in state.items():
            if key.startswith('module.'):
                key = key[len('module.'):]
            if key.startswith('netG.'):
                key = key[len('netG.'):]
            cleaned[key] = value
        return cleaned

    @staticmethod
    def _center_3x3_in_5x5(source, target):
        migrated = torch.zeros_like(target)
        migrated[:, :, 1:4, 1:4] = source.to(
            device=target.device, dtype=target.dtype
        )
        return migrated

    def _load_warmstart_generator(self, checkpoint_path):
        if os.path.isdir(checkpoint_path):
            checkpoint_path = os.path.join(
                checkpoint_path, 'latest_net_G.pth'
            )
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                'generator checkpoint not found: {}'.format(checkpoint_path)
            )

        state = torch.load(checkpoint_path, map_location=str(self.device))
        source = self._clean_checkpoint_state(state)
        module = (
            self.netG.module
            if isinstance(self.netG, torch.nn.DataParallel) else self.netG
        )
        target = module.state_dict()
        transferred = set()

        for key, value in source.items():
            if key in target and target[key].shape == value.shape:
                target[key] = value.to(
                    device=target[key].device, dtype=target[key].dtype
                )
                transferred.add(key)

        # Compatibility path for early spatial-specific SHASP checkpoints.
        for domain in ('vis', 'ir'):
            source_prefix = '{}_specific_encoder.encoder.model'.format(domain)
            target_prefix = '{}_specific_encoder.feature_extractor'.format(
                domain
            )
            for source_index, target_index in ((1, 1), (4, 5), (7, 9)):
                for suffix in ('weight', 'bias'):
                    source_key = '{}.{}.{}'.format(
                        source_prefix, source_index, suffix
                    )
                    target_key = '{}.{}.{}'.format(
                        target_prefix, target_index, suffix
                    )
                    if source_key not in source or target_key not in target:
                        continue
                    source_value = source[source_key]
                    target_value = target[target_key]
                    if source_value.shape == target_value.shape:
                        target[target_key] = source_value.to(
                            device=target_value.device,
                            dtype=target_value.dtype
                        )
                        transferred.add(target_key)
                    elif (
                        suffix == 'weight'
                        and source_value.ndim == 4
                        and tuple(source_value.shape[-2:]) == (3, 3)
                        and tuple(target_value.shape[-2:]) == (5, 5)
                        and source_value.shape[:2] == target_value.shape[:2]
                    ):
                        target[target_key] = self._center_3x3_in_5x5(
                            source_value, target_value
                        )
                        transferred.add(target_key)

        module.load_state_dict(target, strict=True)
        transferred_parameters = sum(
            target[key].numel() for key in transferred
        )
        print(
            'initialized SHASP from {}: {} tensors / {:,} parameters '
            'transferred'.format(
                checkpoint_path, len(transferred), transferred_parameters
            ),
            flush=True
        )

    def _adversarial_scale(self):
        warmup = self.opt.paired_warmup_epochs
        if self.current_epoch <= warmup:
            return 0.0
        ramp = max(self.opt.gan_ramp_epochs, 1)
        return min(1.0, (self.current_epoch - warmup) / float(ramp))

    def set_input(self, input):
        self.real_A = input['A'].to(self.device)
        self.real_B = (
            input['B'].to(self.device) if 'B' in input else None
        )
        self.image_paths = input['A_paths']
        if not self.isTrain and self.real_B is None:
            self.visual_names = ['real_A', 'fake_B']

    def forward(self):
        (
            self.fake_B, self.shared_A, self.predicted_specific_B,
            self.source_specific_A
        ) = self.netG(
            self.real_A, direction='vis_to_ir', return_features=True
        )
        if self.real_B is None:
            return
        (
            self.fake_A, self.shared_B, self.predicted_specific_A,
            self.source_specific_B
        ) = self.netG(
            self.real_B, direction='ir_to_vis', return_features=True
        )
        self.rec_A = self.netG(self.fake_B, direction='ir_to_vis')
        self.rec_B = self.netG(self.fake_A, direction='vis_to_ir')

    def _discriminator_loss(self, discriminator, real, fake):
        return 0.5 * (
            self.criterionGAN(discriminator(real), True)
            + self.criterionGAN(discriminator(fake.detach()), False)
        )

    @staticmethod
    def _gradient_loss(fake, real):
        return (
            F.l1_loss(fake[:, :, :, 1:] - fake[:, :, :, :-1],
                      real[:, :, :, 1:] - real[:, :, :, :-1])
            + F.l1_loss(fake[:, :, 1:, :] - fake[:, :, :-1, :],
                        real[:, :, 1:, :] - real[:, :, :-1, :])
        )

    @staticmethod
    def _intensity_loss(fake, real):
        spatial_dims = (2, 3)
        return (
            F.l1_loss(fake.mean(spatial_dims), real.mean(spatial_dims))
            + F.l1_loss(
                fake.std(spatial_dims, unbiased=False),
                real.std(spatial_dims, unbiased=False)
            )
        )

    def _specific_alignment_loss(self, predicted, target):
        target = target.detach()
        if self.opt.pairing == 'paired':
            return F.l1_loss(predicted, target)
        return (
            F.l1_loss(predicted.mean(1), target.mean(1))
            + F.l1_loss(
                predicted.std(1, unbiased=False),
                target.std(1, unbiased=False)
            )
        )

    @staticmethod
    def _visible_gray(visible):
        if visible.size(1) == 1:
            return visible
        return (
            visible[:, 0:1] * 0.299
            + visible[:, 1:2] * 0.587
            + visible[:, 2:3] * 0.114
        )

    def _bright_region_loss(self, fake_ir, real_ir, visible):
        if self.opt.pairing != 'paired' or self.opt.lambda_bright == 0.0:
            return fake_ir.new_tensor(0.0)
        visible_01 = (self._visible_gray(visible) + 1.0) * 0.5
        weights = (
            (visible_01 - self.opt.bright_threshold)
            / (1.0 - self.opt.bright_threshold)
        ).clamp(0.0, 1.0).square()
        weighted_error = (fake_ir - real_ir).abs() * weights
        denominator = weights.sum() * fake_ir.size(1)
        return weighted_error.sum() / denominator.clamp_min(1.0)

    def backward_G(self):
        adversarial_scale = self._adversarial_scale()
        self.loss_adv_X = (
            self.criterionGAN(self.netD_X(self.fake_A), True)
            * self.opt.lambda_adversarial * adversarial_scale
        )
        self.loss_adv_Y = (
            self.criterionGAN(self.netD_Y(self.fake_B), True)
            * self.opt.lambda_adversarial * adversarial_scale
        )
        self.loss_cycle = self.opt.lambda_cycle * adversarial_scale * (
            self.criterionL1(self.rec_A, self.real_A)
            + self.criterionL1(self.rec_B, self.real_B)
        )
        self.loss_semantic = (
            self.criterionGAN(self.netD_str(self.shared_B), True)
            * self.opt.lambda_semantic * adversarial_scale
        )
        self.loss_specific = self.opt.lambda_specific * (
            self._specific_alignment_loss(
                self.predicted_specific_B, self.source_specific_B
            )
            + self._specific_alignment_loss(
                self.predicted_specific_A, self.source_specific_A
            )
        )
        self.loss_paired = (
            self.opt.lambda_paired
            * self.criterionL1(self.fake_B, self.real_B)
        )
        self.loss_gradient = (
            self.opt.lambda_gradient
            * self._gradient_loss(self.fake_B, self.real_B)
        )
        self.loss_intensity = (
            self.opt.lambda_intensity
            * self._intensity_loss(self.fake_B, self.real_B)
        )
        self.loss_bright = self.opt.lambda_bright * self._bright_region_loss(
            self.fake_B, self.real_B, self.real_A
        )
        (
            self.loss_adv_X + self.loss_adv_Y + self.loss_cycle
            + self.loss_semantic + self.loss_specific + self.loss_paired
            + self.loss_gradient + self.loss_intensity + self.loss_bright
        ).backward()

    def backward_D(self):
        adversarial_scale = self._adversarial_scale()
        fake_A = self.fake_A_pool.query(self.fake_A)
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_X = adversarial_scale * self._discriminator_loss(
            self.netD_X, self.real_A, fake_A
        )
        self.loss_D_Y = adversarial_scale * self._discriminator_loss(
            self.netD_Y, self.real_B, fake_B
        )
        self.loss_D_str = (
            0.5 * self.opt.lambda_semantic * adversarial_scale * (
                self.criterionGAN(self.netD_str(self.shared_A.detach()), True)
                + self.criterionGAN(
                    self.netD_str(self.shared_B.detach()), False
                )
            )
        )
        (self.loss_D_X + self.loss_D_Y + self.loss_D_str).backward()

    def optimize_parameters(self):
        self.forward()
        discriminators = [self.netD_X, self.netD_Y, self.netD_str]
        self.set_requires_grad(discriminators, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        if self._adversarial_scale() > 0:
            self.set_requires_grad(discriminators, True)
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()
        else:
            zero = self.fake_A.new_tensor(0.0)
            self.loss_D_X = zero
            self.loss_D_Y = zero
            self.loss_D_str = zero
            self.optimizer_D.zero_grad()
            self.optimizer_D.step()
