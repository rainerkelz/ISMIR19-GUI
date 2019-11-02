import torch
import numpy as np

from clean_reverse import ReversibleModel
from audio_midi_dataset import get_dataset_individually, Spec2MidiDataset, SqueezingDataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler


def get_data_loaders(direction,
                     base_directory,
                     fold_file,
                     instrument_filename,
                     context,
                     audio_options,
                     batch_size):

    print('-' * 30)
    print('getting data loaders:')
    print('direction', direction)
    print('base_directory', base_directory)
    print('fold_file', fold_file)
    print('instrument_filename', instrument_filename)

    clazz = Spec2MidiDataset

    datasets = get_dataset_individually(
        base_directory,
        fold_file,
        instrument_filename,
        context,
        audio_options,
        clazz
    )
    loaders = []
    for dataset in datasets:
        audiofilename = dataset.audiofilename
        midifilename = dataset.midifilename
        dataset = SqueezingDataset(dataset)
        print('len(dataset)', len(dataset))

        sampler = SequentialSampler(dataset)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True
        )
        loaders.append((fold_file, audiofilename, midifilename, loader))

    return loaders


class InvertibleModel(object):
    def __init__(self, max_t):
        self.max_t = max_t

        direction = 'spec2labels'
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = 'cpu'

        self.audio_options = dict(
            spectrogram_type='LogarithmicFilteredSpectrogram',
            filterbank='LogarithmicFilterbank',
            num_channels=1,
            sample_rate=44100,
            frame_size=4096,
            fft_size=4096,
            hop_size=441 * 4,  # 25 fps
            num_bands=24,
            fmin=30,
            fmax=10000.0,
            fref=440.0,
            norm_filters=True,
            unique_filters=True,
            circular_shift=False,
            add=1.
        )
        self.context = dict(
            frame_size=1,
            hop_size=1,
            origin='center'
        )

        print('loading checkpoint')
        # checkpoint = torch.load('./runs/maps_spec2labels_swd/model_state_final.pkl')
        # checkpoint = torch.load('./runs/rk1_model_state_final_safety.pkl', map_location='cpu')
        checkpoint = torch.load('./model_states/model_state_demo.pkl', map_location='cpu')
        self.model = ReversibleModel(
            device=device,
            batch_size=self.max_t,
            depth=5,
            ndim_tot=256,
            ndim_x=144,
            ndim_y=185,
            ndim_z=9,
            clamp=2,
            zeros_noise_scale=3e-2,  # very magic, much hack!
            y_noise_scale=3e-2
        )

        self.model.to(device)
        self.model.load_state_dict(checkpoint)

    def encode(self, x):
        with torch.no_grad():
            z_hat, zy_padding, y_hat = self.model.encode(torch.Tensor(x))
            return z_hat.cpu().numpy(), zy_padding.cpu().numpy(), y_hat.cpu().numpy()

    def decode(self, z_hat, zy_padding, y_hat):
        with torch.no_grad():
            x_inv, x_padding = self.model.decode_padding(
                torch.Tensor(z_hat),
                torch.Tensor(zy_padding),
                torch.Tensor(y_hat)
            )
            return x_inv.cpu().numpy(), x_padding.cpu().numpy()
