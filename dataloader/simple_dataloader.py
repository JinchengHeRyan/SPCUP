from torch.utils.data import DataLoader
import torch


class SimpleDataLoader(DataLoader):
    def __init__(
        self, dataset, shuffle=False, batch_size=1, num_workers=1, drop_last=False
    ):
        self.dataset = dataset
        super().__init__(
            self.dataset,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=True,
        )

    def collate_fn(self, batch):
        batch = list(zip(*batch))
        (wav_utt, fbank, wav_spec, wav_phase, wav, algorithm_num) = batch
        wav = torch.stack(wav, dim=0)
        wav_spec = torch.stack(wav_spec, dim=0)
        return wav_utt, fbank, wav_spec, wav_phase, wav, algorithm_num
