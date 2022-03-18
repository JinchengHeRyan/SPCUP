import torch
from torch.utils.data import DataLoader
from dataloader.sampler import TrunBatchSampler


class TruncateDataLoader(DataLoader):
    def __init__(
        self,
        dataset,
        trun_range=[80000, 80000],
        step=1,
        shuffle=False,
        batch_size=1,
        num_workers=1,
        drop_last=False,
    ):
        self.dataset = dataset
        self.batch_sampler = TrunBatchSampler(
            self.dataset,
            trun_range=trun_range,
            step=step,
            shuffle=shuffle,
            batch_size=batch_size,
            drop_last=drop_last,
        )
        super().__init__(
            self.dataset,
            collate_fn=self.collate_fn,
            batch_sampler=self.batch_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )

    def collate_fn(self, batch):
        batch = list(zip(*batch))
        (wav_utt, fbank, wav_spec, wav_phase, wav, algorithm_num) = batch
        wav = torch.stack(wav, dim=0)
        wav_spec = torch.stack(wav_spec, dim=0)
        return wav_utt, fbank, wav_spec, wav_phase, wav, algorithm_num
