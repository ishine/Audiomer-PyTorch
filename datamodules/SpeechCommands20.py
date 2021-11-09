from torchaudio_augmentations import (
    RandomApply,
    ComposeMany,
    RandomResizedCrop,
    PolarityInversion,
    Noise,
    Gain,
    HighLowPass,
    Delay,
    PitchShift,
    Reverb,
)
from functools import partial
import pytorch_lightning as pl
import os
import torch
from torchaudio.datasets import SPEECHCOMMANDS, YESNO
import torchaudio
from torch.utils.data import DataLoader
torchaudio.set_audio_backend("sox_io")


labels = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go',
          'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']


SAMPLE_RATE = 8192
DURATION = 1

train_transform = ComposeMany(
    [
        torchaudio.transforms.Resample(orig_freq=16000, new_freq=SAMPLE_RATE),
        RandomApply([PolarityInversion()], p=0.8),
        RandomApply([Noise()], p=0.01),
        RandomApply([Gain()], p=0.3),
        RandomApply(
            [Reverb(sample_rate=SAMPLE_RATE)], p=0.6
        ),
    ],
    1
)


# def test_transform(x): return x[:, :int(SAMPLE_RATE * DURATION)]
from .pad import make_pad_function
test_transform = ComposeMany(
    [
        torchaudio.transforms.Resample(orig_freq=16000, new_freq=SAMPLE_RATE),
        make_pad_function(int(SAMPLE_RATE * DURATION)),
        lambda x: x[:, :int(SAMPLE_RATE * DURATION)]
    ],
    1
)


def label_to_index(word):
    # Return the position of the word in labels
    return torch.tensor(labels.index(word))


def index_to_label(index):
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return labels[index]


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(
        batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


def collate_fn(batch, dataset, augmentation):
    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number
    tensors, targets = [], []
    if dataset == "train" and augmentation == True:
        t = train_transform
    else:
        t = test_transform

    # Gather in lists, and encode labels as indices
    for waveform, _, label, *_ in batch:
        tensors += [t(waveform).squeeze(1)]
        targets += [label_to_index(label)]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)
    return tensors, targets


class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None, root="./"):
        super().__init__(root, download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.join(self._path, line.strip()) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + \
                load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]


class SpeechCommands20(torch.utils.data.Dataset):
    def __init__(self, subset='train', root="./"):
        super().__init__()
        self.ds = SubsetSC(subset=subset, root=root)
        self.indices = []
        for i in range(len(self.ds)):
            if self.ds[i][2] in labels:
                self.indices.append(i)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.ds[self.indices[index]]


class SpeechCommands20DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, pin_memory=True, num_workers=4, root="./", augmentation=True):
        super().__init__()
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.augmentation = augmentation

        self.train_ds = SpeechCommands20("training", root=root)
        self.test_ds = SpeechCommands20("testing", root=root)
        self.validation_ds = SpeechCommands20("validation", root=root)

    def train_dataloader(self):
        return DataLoader(self.train_ds, self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory, collate_fn=partial(collate_fn, dataset="train", augmentation=self.augmentation))

    def test_dataloader(self):
        return DataLoader(self.test_ds, self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory, collate_fn=partial(collate_fn, dataset="test", augmentation=self.augmentation))

    def val_dataloader(self):
        return DataLoader(self.validation_ds, self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory, collate_fn=partial(collate_fn, dataset="val", augmentation=self.augmentation))


if __name__ == "__main__":
    ds = SpeechCommands20(
        "testing", "../")
    print(ds[0])
