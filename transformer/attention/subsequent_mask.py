import arr_fun as torch


def subsequent_mask(sequence_length):
    "Mask out subsequent positions."
    mask_shape = (1, sequence_length, sequence_length)
    future_positions = torch.triu(torch.ones(mask_shape), diagonal=1).type(torch.uint8)
    return future_positions == 0
