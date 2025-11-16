from functools import partial

import padertorch as pt
from padertorch.contrib.mk.typing import TSeqLen
from padertorch.ops.sequence.mask import compute_mask
import torch
from torch import Tensor


class Slicer(pt.Module): #padertorch 
    def __init__(
        self,
        sampling_rate: int,
        min_length_in_seconds: float = 1.0,
        max_length_in_seconds: float = 2.0,
        sequence_axis: int = -2,
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.min_length_in_seconds = min_length_in_seconds
        self.max_length_in_seconds = max_length_in_seconds
        self.sequence_axis = sequence_axis

    def forward(
        self,
        x: Tensor,   #Tensor der form [B, T, D] | batch size, time, feature dimension
        sequence_lengths: TSeqLen, #echte länge jeder sequenz, ohne padding
        rng=None,   
    ):
        if sequence_lengths is None: #überprüfen ob sequence_lengths existiert
            raise ValueError("sequence_lengths must be provided")

        bs = x.shape[0] #batch size = anzahl der beispiele
        #zufällige slice längen erzeugen (zwischen min-max length)
        lengths = torch.randint(
            int(self.min_length_in_seconds * self.sampling_rate),
            int(self.max_length_in_seconds * self.sampling_rate),
            (bs,),
            generator=rng,
        ).long()
        lengths = torch.minimum(lengths, torch.tensor(sequence_lengths)-1) #slice längen durch reele länge begrenzen
        max_length = torch.amax(lengths).item() #max_length , alle die kürzer werden gepadet
        #valid stop-Indizes für jedes Beispiel
        stops = torch.cat(list(map(
            partial(torch.randint, size=(1,), generator=rng),
            [max_length]*bs, sequence_lengths,
        ))).long()
        starts = stops - max_length #start position berechnen
        indices = (
            torch.arange(max_length).unsqueeze(0) + starts.unsqueeze(1)
        )
        for _ in range(x.dim() - 2): #an Tensor-dims anpassen damit indizes selbe dim wie tensor haben
            indices = indices.unsqueeze(-1)
        indices = indices.moveaxis(1, self.sequence_axis).to(x.device) #wenn zeit nicht in der 2 dimension liegt wird sie verschoben
        #alle Batch-items auf max_length bringen
        shape = list(x.shape)
        shape[self.sequence_axis] = max_length
        indices = indices.expand(*shape)
        x = x.gather(self.sequence_axis, indices) #extrahiert die ausgewählten Zeit-indizes -> Batch zufälliger audiosegmente: alle gleich lang
        # Padding mask
        mask = compute_mask(
            x, lengths.to(x.device),
            batch_axis=0, sequence_axis=self.sequence_axis,
        )
        x = x * mask
        return x, lengths, indices
