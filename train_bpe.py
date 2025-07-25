import re
from collections import defaultdict
from typing import List, Tuple, Dict


def run_train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str],
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    
    # Read raw bytes from file
    with open(input_path, "rb") as f:
        data = f.read()

    # Initialize vocab with all byte values (0–255)
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    # Add special tokens (encoded as UTF-8 bytes)
    for token in special_tokens:
        token_bytes = token.encode("utf-8")
        vocab[len(vocab)] = token_bytes

    # Regex pattern to split text into word-like and space/punctuation pieces
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    import regex  # Required for \p{L} and \p{N}
    text = data.decode("utf-8")
    tokens = regex.findall(PAT, text)

    # Flatten to byte-level tokens
    byte_tokens = []
    for token in tokens:
        if token.strip():
            byte_tokens.extend(token.encode("utf-8"))
    tokens = byte_tokens

    # Initial merges list
    merges: List[Tuple[bytes, bytes]] = []

    # Map bytes to vocab ids
    token_to_id = {v: k for k, v in vocab.items()}
    tokens = [token_to_id[bytes([b])] for b in tokens]

    def merge_pair(pair: Tuple[int, int], new_token_id: int, tokens: List[int]) -> List[int]:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                new_tokens.append(new_token_id)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    def get_most_frequent_pair(tokens: List[int]) -> Tuple[int, int] | None:
        pair_freqs = defaultdict(int)
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pair_freqs[pair] += 1
        return max(pair_freqs.items(), key=lambda x: x[1])[0] if pair_freqs else None

    def should_skip_merge(byte1: bytes, byte2: bytes) -> bool:
        merged = byte1 + byte2
        return b"<|" in merged and merged not in [t.encode("utf-8") for t in special_tokens]

    while len(vocab) < vocab_size:
        most_frequent_pair = get_most_frequent_pair(tokens)
        if not most_frequent_pair:
            break

        byte1 = vocab[most_frequent_pair[0]]
        byte2 = vocab[most_frequent_pair[1]]

        # Skip merge if it creates a forbidden prefix
        if should_skip_merge(byte1, byte2):
            tokens = [t for t in tokens if t not in most_frequent_pair]
            continue

        new_token_id = len(vocab)
        new_token = byte1 + byte2
        vocab[new_token_id] = new_token
        merges.append((byte1, byte2))
        tokens = merge_pair(most_frequent_pair, new_token_id, tokens)

    return vocab, merges

# ========== Test the BPE training ==========

# Step 1: Create a simple input file
with open("sample.txt", "w", encoding="utf-8") as f:
    f.write("low lower lowest")

# Step 2: Run BPE on the sample
special_tokens = ["<|endoftext|>"]
vocab_size = 300

vocab, merges = run_train_bpe("sample.txt", vocab_size, special_tokens)

# Step 3: Print vocab
print("\n=== Vocabulary ===")
for idx in sorted(vocab):
    print(f"{idx:>4}: {vocab[idx]}")

# Step 4: Print merges
print("\n=== Merges ===")
for i, (b1, b2) in enumerate(merges):
    print(f"{i:>2}: {b1} + {b2} -> {b1 + b2}")
