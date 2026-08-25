import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_dir', required=True, type=Path)
    parser.add_argument('--generated_dir', required=True, type=Path)
    parser.add_argument(
        '--generated_tag', default='_fake_B',
        help='suffix removed from generated filenames to recover the source stem'
    )
    parser.add_argument(
        '--distribution_metrics', action='store_true',
        help='also compute clean-fid FID and KID (may download Inception weights)'
    )
    return parser.parse_args()


def image_files(directory):
    return sorted(
        path for path in directory.rglob('*')
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def build_pairs(real_dir, generated_dir, generated_tag):
    real_by_stem = {path.stem: path for path in image_files(real_dir)}
    pairs = []
    for generated in image_files(generated_dir):
        if generated_tag and not generated.stem.endswith(generated_tag):
            continue
        stem = (
            generated.stem[:-len(generated_tag)]
            if generated_tag else generated.stem
        )
        if stem in real_by_stem:
            pairs.append((real_by_stem[stem], generated))
    if not pairs:
        raise RuntimeError(
            'No matched image pairs. Check directories and --generated_tag.'
        )
    return pairs


def load_gray(path, size=None):
    image = Image.open(path).convert('L')
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.float32)


def pixel_metrics(pairs):
    ssim_values, psnr_values = [], []
    for real_path, generated_path in pairs:
        real_image = Image.open(real_path)
        real = load_gray(real_path)
        generated = load_gray(generated_path, real_image.size)
        ssim_values.append(structural_similarity(real, generated, data_range=255))
        psnr_values.append(
            peak_signal_noise_ratio(real, generated, data_range=255)
        )
    return np.asarray(ssim_values), np.asarray(psnr_values)


def distribution_metrics(pairs):
    try:
        from cleanfid import fid
    except ImportError as exc:
        raise RuntimeError(
            'Install clean-fid to use --distribution_metrics'
        ) from exc
    with tempfile.TemporaryDirectory(prefix='shasp_metrics_') as temp_root:
        real_root = Path(temp_root) / 'real'
        fake_root = Path(temp_root) / 'fake'
        real_root.mkdir()
        fake_root.mkdir()
        for index, (real_path, fake_path) in enumerate(pairs):
            os.symlink(real_path.resolve(), real_root / '{:06d}.png'.format(index))
            os.symlink(fake_path.resolve(), fake_root / '{:06d}.png'.format(index))
        fid_value = fid.compute_fid(str(real_root), str(fake_root))
        kid_value = fid.compute_kid(str(real_root), str(fake_root))
    return fid_value, kid_value


def main():
    args = parse_args()
    pairs = build_pairs(
        args.real_dir, args.generated_dir, args.generated_tag
    )
    ssim_values, psnr_values = pixel_metrics(pairs)
    print('matched images: {}'.format(len(pairs)))
    print('SSIM: {:.6f} ± {:.6f}'.format(
        ssim_values.mean(), ssim_values.std()
    ))
    print('PSNR: {:.6f} ± {:.6f} dB'.format(
        psnr_values.mean(), psnr_values.std()
    ))
    if args.distribution_metrics:
        fid_value, kid_value = distribution_metrics(pairs)
        print('FID: {:.6f}'.format(fid_value))
        print('KID: {:.6f}'.format(kid_value))


if __name__ == '__main__':
    main()
