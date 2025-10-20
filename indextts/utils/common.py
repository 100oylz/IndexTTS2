import torch
from typing import Optional
def make_pad_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
        max_len (int, optional): Maximum length. If None, use max value in lengths.

    Returns:
        torch.Tensor: Mask tensor containing indices of padded part (B, 1, T).
    """
    if(max_len is None):
        max_len = lengths.max().item()

    bs = lengths.size(0)
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, max_len)
    mask = seq_range_expand >= lengths.unsqueeze(1)
    return mask.unsqueeze(1)


    