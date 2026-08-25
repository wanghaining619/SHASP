"""Dataset loader for paired or unpaired VIS/IR image collections.

Training requires both domains.  VIS-to-IR inference may contain only a
``testA`` directory; ``testB`` is optional ground truth used for comparison.
"""

import os
import random

from PIL import Image

from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset


class CrossSpectralDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument(
            '--pairing', choices=('paired', 'unpaired'), default='paired',
            help='paired uses matching sorted A/B images and synchronized augmentation'
        )
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.A_paths = sorted(make_dataset(
            os.path.join(opt.dataroot, opt.phase + 'A'), opt.max_dataset_size
        ))
        b_directory = os.path.join(opt.dataroot, opt.phase + 'B')
        self.B_paths = sorted(make_dataset(
            b_directory, opt.max_dataset_size
        )) if os.path.isdir(b_directory) else []
        if not self.A_paths:
            raise RuntimeError(
                'Cross-spectral data requires a non-empty {}A folder under {}'
                .format(opt.phase, opt.dataroot)
            )
        if opt.isTrain and not self.B_paths:
            raise RuntimeError(
                'SHASP training requires a non-empty {}B folder under {}'
                .format(opt.phase, opt.dataroot)
            )
        if (
            opt.pairing == 'paired' and self.B_paths
            and len(self.A_paths) != len(self.B_paths)
        ):
            raise RuntimeError(
                'Paired mode requires equal A/B image counts ({} versus {}). '
                'Use --pairing unpaired for unmatched collections.'
                .format(len(self.A_paths), len(self.B_paths))
            )
        # SHASP's paper convention is fixed: A is RGB VIS and B is IR.
        self.A_nc = opt.input_nc
        self.B_nc = opt.output_nc

    @staticmethod
    def _open(path, channels):
        return Image.open(path).convert('L' if channels == 1 else 'RGB')

    def __getitem__(self, index):
        # In paired mode A and B always use the same sorted-list index.
        # DataLoader may shuffle sample indices between batches during training,
        # but it never breaks the A[index] <-> B[index] correspondence.
        index_A = index % len(self.A_paths)
        if not self.B_paths:
            index_B = None
        elif self.opt.pairing == 'paired':
            index_B = index_A
        elif self.opt.serial_batches:
            index_B = index % len(self.B_paths)
        else:
            index_B = random.randrange(len(self.B_paths))

        A_path = self.A_paths[index_A]
        A_image = self._open(A_path, self.A_nc)
        if index_B is None:
            params_A = get_params(self.opt, A_image.size)
            A = get_transform(
                self.opt, params_A, grayscale=self.A_nc == 1
            )(A_image)
            return {'A': A, 'A_paths': A_path}

        B_path = self.B_paths[index_B]
        B_image = self._open(B_path, self.B_nc)
        if self.opt.pairing == 'paired':
            # Reuse exactly the same random crop/flip parameters. Calling
            # get_transform without shared params would spatially misalign a
            # registered VIS/IR pair even when the file indices match.
            shared_params = get_params(self.opt, A_image.size)
            params_A = shared_params
            params_B = shared_params
        else:
            params_A = get_params(self.opt, A_image.size)
            params_B = get_params(self.opt, B_image.size)
        A = get_transform(
            self.opt, params_A, grayscale=self.A_nc == 1
        )(A_image)
        B = get_transform(
            self.opt, params_B, grayscale=self.B_nc == 1
        )(B_image)
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        if self.opt.pairing == 'paired' or not self.B_paths:
            return len(self.A_paths)
        return max(len(self.A_paths), len(self.B_paths))
