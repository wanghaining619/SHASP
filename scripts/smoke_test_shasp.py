"""CPU integration checks for the released SHASP implementation."""

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models import networks
from models.shasp_model import ShaspModel


def options(checkpoint=''):
    return SimpleNamespace(
        gpu_ids=[], isTrain=True, checkpoints_dir='/tmp',
        name='shasp_smoke', preprocess='resize_and_crop', input_nc=3,
        output_nc=1, ngf=8, ndf=8, norm='instance', no_dropout=True,
        decoder_blocks=2, specific_dim=32, mapper_blocks=2,
        tone_hidden_dim=16, thermal_base_gain=0.5,
        init_type='normal', init_gain=0.02, netD='basic', n_layers_D=2,
        pool_size=0, gan_mode='lsgan', lr=1e-4, beta1=0.5,
        lambda_adversarial=1.0, lambda_cycle=5.0,
        lambda_semantic=0.5, lambda_specific=2.0,
        lambda_paired=20.0, lambda_gradient=5.0,
        lambda_intensity=1.0, lambda_bright=5.0,
        bright_threshold=0.8, paired_warmup_epochs=0,
        gan_ramp_epochs=1, direction='AtoB', pairing='paired',
        warmstart_checkpoint=checkpoint, global_warmup_epochs=2,
    )


def check_enhance_gan():
    logits = torch.tensor(
        [-80.0, -2.0, 0.0, 2.0, 80.0], requires_grad=True
    )
    criterion = networks.GANLoss('enhance')
    real_loss = criterion(logits, True)
    fake_loss = criterion(logits, False)
    assert torch.allclose(
        real_loss, torch.nn.functional.softplus(-logits).mean()
    )
    assert torch.allclose(
        fake_loss, torch.nn.functional.softplus(logits).mean()
    )
    (real_loss + fake_loss).backward()
    assert torch.isfinite(logits.grad).all()


def main():
    torch.manual_seed(13)
    model = ShaspModel(options())

    vis = torch.rand(1, 3, 128, 128) * 0.4 + 0.55
    ir = torch.randn(1, 1, 128, 128).clamp(-1.0, 1.0)
    with torch.no_grad():
        fake_ir, shared, predicted_ir, source_vis = model.netG(
            vis, direction='vis_to_ir', return_features=True
        )
        gain, bias = model.netG.thermal_tone_adapter(predicted_ir)
        source_only_ir = model.netG(vis, direction='vis_to_ir')
        visible_gray = model.netG._gray(vis).clamp(-0.999, 0.999)
        expected_initial_ir = torch.tanh(
            0.5 * torch.atanh(visible_gray)
        )

    assert tuple(fake_ir.shape) == (1, 1, 128, 128)
    assert tuple(shared.shape) == (1, 32, 32, 32)
    assert tuple(predicted_ir.shape) == (1, 32)
    assert tuple(source_vis.shape) == (1, 32)
    assert torch.allclose(gain, torch.full_like(gain, 0.5))
    assert torch.equal(bias, torch.zeros_like(bias))
    assert torch.allclose(fake_ir, expected_initial_ir, atol=1e-6)
    assert torch.equal(fake_ir, source_only_ir)
    assert fake_ir.mean() < visible_gray.mean()

    with tempfile.TemporaryDirectory(prefix='shasp_smoke_') as temporary_dir:
        checkpoint = Path(temporary_dir) / 'latest_net_G.pth'
        torch.save(model.netG.state_dict(), checkpoint)
        torch.manual_seed(17)
        warmed = ShaspModel(options(str(checkpoint)))
        source_state = model.netG.state_dict()
        warmed_state = warmed.netG.state_dict()
        assert source_state.keys() == warmed_state.keys()
        assert all(
            torch.equal(source_state[key], warmed_state[key])
            for key in source_state
        )

    warmed.set_epoch(1)
    warmup_parameters = {
        id(parameter)
        for submodule in (
            warmed.netG.vis_specific_encoder,
            warmed.netG.ir_specific_encoder,
            warmed.netG.vis_to_ir_specific_mapper,
            warmed.netG.ir_to_vis_specific_mapper,
            warmed.netG.vis_decoder.body[0],
            warmed.netG.ir_decoder.body[0],
            warmed.netG.thermal_tone_adapter,
        )
        for parameter in submodule.parameters()
    }
    assert all(
        parameter.requires_grad == (id(parameter) in warmup_parameters)
        for parameter in warmed.netG.parameters()
    )
    warmed.set_epoch(3)
    assert all(
        parameter.requires_grad for parameter in warmed.netG.parameters()
    )

    bright_vis = torch.full((1, 3, 16, 16), -1.0)
    bright_vis[:, :, 4:12, 4:12] = 1.0
    bright_fake = torch.ones(1, 1, 16, 16, requires_grad=True)
    gray_target = torch.zeros(1, 1, 16, 16)
    bright_loss = warmed._bright_region_loss(
        bright_fake, gray_target, bright_vis
    )
    assert bright_loss > 0
    bright_loss.backward()
    assert bright_fake.grad[:, :, 4:12, 4:12].abs().sum() > 0
    assert bright_fake.grad[:, :, :4, :].abs().sum() == 0

    warmed.initialized_from_warmstart = False
    warmed.set_input({
        'A': vis,
        'B': ir,
        'A_paths': ['vis.png'],
        'B_paths': ['ir.png'],
    })
    warmed.optimize_parameters()
    expected_shapes = {
        'fake_A': (1, 3, 128, 128),
        'fake_B': (1, 1, 128, 128),
        'rec_A': (1, 3, 128, 128),
        'rec_B': (1, 1, 128, 128),
    }
    for name, shape in expected_shapes.items():
        assert tuple(getattr(warmed, name).shape) == shape
    losses = warmed.get_current_losses()
    assert 'specific' in losses and 'bright' in losses
    assert all(
        torch.isfinite(torch.tensor(value)) for value in losses.values()
    )
    check_enhance_gan()

    print('SHASP smoke test passed')
    print('initial thermal base gain: {:.3f}'.format(gain.item()))
    print(losses)


if __name__ == '__main__':
    main()
