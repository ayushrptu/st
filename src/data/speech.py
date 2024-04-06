import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import numpy as np
import librosa
from config import DATA_FOLDER
from sphfile import SPHFile
import os
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler


def raw_to_spec(signal, len_=32700):
    if len(signal) < len_:
        return None
    # spectrogram_ = np.abs(librosa.stft(signal.astype(float), n_fft=512))
    spectrogram = np.abs(librosa.stft(signal.astype(float), n_fft=512))
    return np.log(np.swapaxes(spectrogram, 0, 1))


def spec_to_raw(spectrogram, len_=32700):
    spectrogram = np.exp(np.swapaxes(spectrogram, 0, 1))
    signal = librosa.core.spectrum.griffinlim(spectrogram, n_fft=512, length=len_)
    # wavfile.write("audio.wav", 16000, np.array(audio_signal, dtype=np.int16))
    return signal


def read_raw(filename, rate=16000):
    sph = SPHFile(filename)
    # todo: upsample/downsample?
    if sph.format["sample_rate"] != rate:
        return None
    out = raw_to_spec(sph.content)
    if out is not None:
        out = out.astype(np.float32)
    return out


def plot_sp(spectrogram, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    data_log = np.swapaxes(spectrogram, 0, 1)
    ax.imshow(data_log, cmap="Greys", vmin=np.percentile(data_log, 0), vmax=np.percentile(data_log, 99),
              origin="lower")
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')
    plt.show()


# Load TIMIT Data
class SpeechData(Dataset):

    def __init__(self, folder, seq_len=256, batch_size=4, select_class=None, transform=None, **dataloader_kwargs):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.transform = transform
        self.dataloader_kwargs = dataloader_kwargs
        # Load all at once?
        self.fs = [os.path.join(path, f) for (path, n, fns) in os.walk(os.path.join(DATA_FOLDER, folder, "TRAIN"))
                   for f in fns if f.endswith(".WAV")] + \
                  [os.path.join(path, f) for (path, n, fns) in os.walk(os.path.join(DATA_FOLDER, folder, "TEST"))
                   for f in fns if f.endswith(".WAV")]
        self.data = {f: read_raw(f) for f in self.fs}
        self.data = {f: s for f, s in self.data.items() if s is not None}
        if select_class is not None:
            self.data = {f: s for f, s in self.data.items() if (f.split("\\")[-2][0] == "M") == select_class}
        self.fs = list(self.data.keys())
        self.data = list(self.data.values())

        self.male = np.array([1 if f.split("\\")[-2][0] == "M" else 0 for f in self.fs])
        self.person = np.array([f.split("\\")[-2] for f in self.fs])
        self.list_p = np.unique(self.person)
        self.list_m = [f for f in self.list_p if f.startswith("M")]
        self.list_f = [f for f in self.list_p if f.startswith("F")]

        # 438 M, 192 F, 630 T
        tpm = int(np.ceil(len(self.list_m) * 0.1))
        tpf = int(np.ceil(len(self.list_f) * 0.1))
        test_m = self.list_m[-tpm:]
        val_m = self.list_m[-(2 * tpm):-tpm]
        train_m = self.list_m[:-(2 * tpm)]
        test_f = self.list_f[-tpf:]
        val_f = self.list_f[-(2 * tpf):-tpf]
        train_f = self.list_f[:-(2 * tpf)]
        self.train = train_m + train_f
        self.val = val_m + val_f
        self.test = test_m + test_f

        self.anchors = np.insert(np.cumsum([len(d) - seq_len + 1 for d in self.data]), 0, 0)
        pass

    def __getitem__(self, idx):
        anc = np.argmax(self.anchors > idx) - 1
        out = dict()
        out["x"] = self.data[anc][idx - self.anchors[anc]: idx - self.anchors[anc] + self.seq_len]
        out["y"] = out["x"].copy()
        out["didx"] = self.male[anc]
        if self.transform is not None:
            return self.transform(out["x"])
        else:
            return out

    def __len__(self):
        return self.anchors[-1]

    def dataloader(self, **kwargs):
        train_idxs = [x for l in
                      [list(range(self.anchors[i], self.anchors[i + 1])) for i, p in enumerate(self.person) if
                       p in self.train]
                      for x in l
                      ]
        val_idxs = [x for l in
                      [list(range(self.anchors[i], self.anchors[i + 1])) for i, p in enumerate(self.person) if
                       p in self.val]
                      for x in l
                      ]
        test_idxs = [x for l in
                      [list(range(self.anchors[i], self.anchors[i + 1])) for i, p in enumerate(self.person) if
                       p in self.test]
                      for x in l
                      ]
        train_d = DataLoader(self, sampler=SubsetRandomSampler(train_idxs), batch_size=self.batch_size,
                             **kwargs)
        val_d = DataLoader(self, sampler=SubsetRandomSampler(val_idxs), batch_size=self.batch_size,
                           **kwargs)
        test_d = DataLoader(self, sampler=SubsetRandomSampler(test_idxs), batch_size=self.batch_size,
                            **kwargs)
        return train_d, val_d, test_d


if __name__ == "__main__":
    data = SpeechData("TIMIT/TRAIN", "TIMIT/TEST", batch_size=10, seq_len=256).dataloader()[0]
    sp = next(iter(data))
    plot_sp(sp["x"][0])
