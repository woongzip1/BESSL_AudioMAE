from matplotlib import pyplot as plt
import torchaudio as ta
import torch
import sys
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import numpy as np
import torchaudio.compliance.kaldi as kaldi
from utils import *

class CustomDataset(Dataset):
    """ PATH = [dir1, dir2 , ...] 

    path_dir_wb=[ "/mnt/hdd/Dataset/FSD50K_16kHz", 
                "/mnt/hdd/Dataset/MUSDB18_HQ_16kHz_mono"],
    path_dir_nb=["/mnt/hdd/Dataset/FSD50K_16kHz_codec",
                 "/mnt/hdd/Dataset/MUSDB18_MP3_8k"],
                 """
    def __init__(self, path_dir_nb, path_dir_wb, seg_len=3, mode="train"):
        assert isinstance(path_dir_nb, list), "PATH must be a list"

        self.seg_len = seg_len
        self.mode = mode
        
        paths_wav_wb = []
        paths_wav_nb = []
        for i in range(len(path_dir_nb)):
            self.path_dir_nb = path_dir_nb[i]
            self.path_dir_wb = path_dir_wb[i]
            paths_wav_wb.extend(get_audio_paths(self.path_dir_wb, file_extensions='.wav'))
            paths_wav_nb.extend(get_audio_paths(self.path_dir_nb, file_extensions='.wav'))

        print(f"LR {len(paths_wav_nb)} and HR {len(paths_wav_wb)} file numbers loaded!")

        if len(paths_wav_wb) != len(paths_wav_nb):
            sys.exit(f"Error: LR {len(paths_wav_nb)} and HR {len(paths_wav_wb)} file numbers are different!")

        self.filenames = [(path_wav_wb, path_wav_nb) for path_wav_wb, path_wav_nb in zip(paths_wav_wb, paths_wav_nb)]
        print(f"{mode}: {len(self.filenames)} files loaded")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        path_wav_wb, path_wav_nb = self.filenames[idx]

        wav_nb, sr_nb = ta.load(path_wav_nb)
        wav_wb, sr_wb = ta.load(path_wav_wb)

        # print(wav_nb.shape)
        wav_wb = wav_wb.view(1, -1)
        wav_nb = wav_nb.view(1, -1)

        if self.seg_len > 0 and self.mode == "train":
            duration = int(self.seg_len * 16000)  # Assuming 16kHz sample rate
            sig_len = wav_wb.shape[-1]

            t_start = np.random.randint(low=0, high=np.max([1, sig_len - duration - 2]), size=1)[0]
            if t_start % 2 == 1:
                t_start -= 1
            t_end = t_start + duration

            wav_nb = wav_nb.repeat(1, t_end // sig_len + 1)[..., t_start:t_end]
            wav_wb = wav_wb.repeat(1, t_end // sig_len + 1)[..., t_start:t_end]

            # wav_nb = self.ensure_length(wav_nb, sr_nb * self.seg_len)
            # wav_wb = self.ensure_length(wav_wb, sr_wb * self.seg_len)

        elif self.mode == "val":
            # min_len = min(wav_wb.shape[-1], wav_nb.shape[-1])
            # wav_nb = self.ensure_length(wav_nb, min_len)
            # wav_wb = self.ensure_length(wav_wb, min_len)
            wav_nb = self.set_maxlen(wav_nb, max_lensec=10.24)
            wav_wb = self.set_maxlen(wav_wb, max_lensec=10.24)

        else:
            sys.exit(f"unsupported mode! (train/val)")

        # Compute fbank using Kaldi
        # fbank_nb = self.compute_fbank(wav_nb, sr_nb)
        fbank_wb = self.compute_fbank(wav_wb, sr_wb)

        # Normalize the fbank
        # fbank_nb = self.normalize_fbank(fbank_nb)
        fbank_wb = self.normalize_fbank(fbank_wb)

        return wav_nb, wav_wb, fbank_wb, get_filename(path_wav_wb)[0]
        # return fbank_wb, get_filename(path_wav_wb)[0]

    @staticmethod
    def ensure_length(wav, target_length):
        if wav.shape[1] < target_length:
            pad_size = target_length - wav.shape[1]
            wav = F.pad(wav, (0, pad_size))
        elif wav.shape[1] > target_length:
            wav = wav[:, :target_length]
        return wav
        
    @staticmethod
    def set_maxlen(wav, max_lensec):
        sr = 16000
        max_len = int(max_lensec * sr)
        if wav.shape[1] > max_len:
            # print(wav.shape, max_len)
            wav = wav[:, :max_len]
        return wav

    @staticmethod
    def compute_fbank(wav, sample_rate, num_mel_bins=128, target_len=1024):
        """Compute the fbank using Kaldi."""
        wav = wav - wav.mean()
        fbank = kaldi.fbank(wav, htk_compat=True, sample_frequency=sample_rate, use_energy=False, 
                            window_type='hanning', num_mel_bins=num_mel_bins, dither=0.0, frame_shift=10)
        
        # Ensure the fbank has the target length
        n_frames = fbank.shape[0]
        p = target_len - n_frames
        if p > 0:
            fbank = F.pad(fbank, (0, 0, 0, p), "constant", 0)
        elif p < 0:
            fbank = fbank[:target_len, :]
        return fbank

    @staticmethod
    def normalize_fbank(fbank):
        """Normalize the fbank based on predefined mean and std."""
        norm_mean = -4.2677393
        norm_std = 4.5689974
        fbank = (fbank - norm_mean) / (norm_std * 2)
        return fbank
    
    def display_fbank(self, bank, minmin=None, maxmax=None, colorbar=False):
        #print(bank.shape, bank.min(), bank.max())
        #plt.figure(figsize=(18, 6))
        plt.figure(figsize=(20, 4))
        plt.imshow(20*bank.T.numpy(), origin='lower', interpolation='nearest', vmax=maxmax, vmin=minmin,  aspect='auto')
        if colorbar: plt.colorbar()
        #S_db = librosa.amplitude_to_db(np.abs(bank.T.numpy()),ref=np.max)
        #S_db = bank.T.numpy()
        #plt.figure()
        #librosa.display.specshow(10*bank.T.numpy())
        #plt.colorbar()
        plt.show()