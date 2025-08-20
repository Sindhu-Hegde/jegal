import librosa
import torch
import numpy as np
from scipy.io import wavfile

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 


audio_opts = {
    'sample_rate': 16000,
    'n_fft': 512,
    'win_length': 320,
    'hop_length': 160,
    'n_mel': 80,
}


def load_wav(path, fr=0, to=100000000, sample_rate=16000):
    """Loads Audio wav from path at time indices given by fr, to (seconds)"""

    _, wav = wavfile.read(path)

    return wav


def wav2filterbanks(wav, mel_basis=None):
    """
    :param wav: Tensor b x T
    """

    assert len(wav.shape) == 2, 'Need batch of wavs as input'
    device = 'cpu'
    spect = torch.stft(wav,
					   return_complex=True,
					   n_fft=audio_opts['n_fft'],
					   hop_length=audio_opts['hop_length'],
					   win_length=audio_opts['win_length'],
					   window=torch.hann_window(audio_opts['win_length']).to(device),
					   center=True,
					   pad_mode='reflect',
					   normalized=False,
					   onesided=True)  # b x F x T x 2
    spect = torch.view_as_real(spect)
    spect = spect[:, :, :-1, :]

    # ----- Log filterbanks --------------
    # mag spectrogram - # b x F x T
    mag = power_spect = torch.norm(spect, dim=-1)
    phase = torch.atan2(spect[..., 1], spect[..., 0])
    if mel_basis is None:
        # Build a Mel filter
        mel_basis = torch.from_numpy(
            librosa.filters.mel(sr=audio_opts['sample_rate'],
                                n_fft=audio_opts['n_fft'],
                                n_mels=audio_opts['n_mel'],
                                fmin=0,
                                fmax=int(audio_opts['sample_rate'] / 2)))
        mel_basis = mel_basis.float().to(power_spect.device)
    features = torch.log(torch.matmul(mel_basis, power_spect) +
                         1e-20)  # b x F x T
    features = features.permute([0, 2, 1]).contiguous()  # b x T x F
    # -------------------

    return features, mag, phase, mel_basis


def torch_mag_phase_2_np_complex(mag_spect, phase):
    complex_spect_2d = torch.stack(
        [mag_spect * torch.cos(phase), mag_spect * torch.sin(phase)], -1)
    complex_spect_np = complex_spect_2d.cpu().detach().numpy()
    complex_spect_np = complex_spect_np[..., 0] + 1j * complex_spect_np[..., 1]
    return complex_spect_np


def torch_mag_phase_2_complex_as_2d(mag_spect, phase):
    complex_spect_2d = torch.stack(
        [mag_spect * torch.cos(phase), mag_spect * torch.sin(phase)], -1)
    return complex_spect_2d


def torch_phase_from_normalized_complex(spect):
    phase = torch.atan2(spect[..., 1], spect[..., 0])
    return phase


def reconstruct_wav_from_mag_phase(mag, phase):
    spect = torch_mag_phase_2_np_complex(mag, phase)
    wav = np.stack([
        librosa.core.istft(spect[ii],
                           hop_length=audio_opts['hop_length'],
                           win_length=audio_opts['win_length'],
                           center=True) for ii in range(spect.shape[0])
    ])

    return wav
