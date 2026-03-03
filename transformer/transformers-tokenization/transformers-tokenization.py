import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]

        vocab_texts = []

        for text in texts:
            vocab_texts.extend(text.lower().split())

        unique_text = sorted(set(vocab_texts))

        vocab = list(dict.fromkeys(special_tokens + unique_text))

        self.id_to_word = vocab
        self.word_to_id = {word: i for i, word in enumerate(vocab)}
        self.vocab_size = len(vocab)
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        tokens = text.lower().split()

        unk_id = self.word_to_id.get(self.unk_token, 1)

        encode_word = [self.word_to_id.get(token, unk_id) for token in tokens]

        return encode_word
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        tokens = [self.id_to_word[idx] for idx in ids if idx < len(self.id_to_word)]

        special = [self.pad_token, self.bos_token, self.eos_token]
        remove_special = [t for t in tokens if t not in special]

        return " ".join(remove_special)