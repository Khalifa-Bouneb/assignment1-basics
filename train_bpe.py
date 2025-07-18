import os

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, 'rb') as f:
        data = f.read()
    # Simulate training BPE and generating a vocabulary
    vocab = {i: bytes(f'token_{i}', 'utf-8') for i in range(vocab_size)}
    merges = [(bytes(f'merge_{i}', 'utf-8'), bytes(f'with_{i}', 'utf-8')) for i in range(vocab_size)]
    return vocab, merges

    
    