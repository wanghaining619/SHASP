

import torch.utils.data

from .cross_spectral_dataset import CrossSpectralDataset


def _validate_dataset_name(dataset_name):
    if dataset_name.lower() != 'cross_spectral':
        raise ValueError(
            'This repository only provides --dataset_mode cross_spectral '
            '(got {!r})'.format(dataset_name)
        )


def get_option_setter(dataset_name):
    _validate_dataset_name(dataset_name)
    return CrossSpectralDataset.modify_commandline_options


def create_dataset(opt):
    return CustomDatasetDataLoader(opt)


class CustomDatasetDataLoader:
    """Wrap the dataset with a finite PyTorch DataLoader."""

    def __init__(self, opt):
        _validate_dataset_name(opt.dataset_mode)
        self.opt = opt
        self.dataset = CrossSpectralDataset(opt)
        print('dataset [{}] was created'.format(type(self.dataset).__name__))
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads),
        )

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for index, sample in enumerate(self.dataloader):
            if index * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield sample
