import numpy as np
import tiktoken
class Tokenizer:
    def __init__(self, tokenizer_type: str):
        self.tokenizer_type = tokenizer_type
        if self.tokenizer_type != "char" or self.tokenizer_type != "bpe":
            assert self.tokenizer_type == "char" or self.tokenizer_type == "bpe", 'Tokenizer type must be "char" or "bpe"'
            raise ValueError("Invalid Tokenizer type")
        self.char_set = set(chr(i) for i in range(32, 127))
        self.vocab_size = len(self.char_set)
        self.bpe_tokens = set()

    def encode(self, tokenizer_type: str, text: str):
        if tokenizer_type == "char":
            return self.char_encoding(text)
        elif tokenizer_type == "bpe":
            return self.bpe_encoding(text)

    def decode(self, tokenizer_type: str, encoding: np.array):
        if tokenizer_type == "char":
            return self.char_decoding(encoding)
        elif tokenizer_type == "bpe":
            return self.bpe_decoding(encoding)
            
    def char_encoding(self, text: str):
        return np.array([ord(c) for c in text])
        
    def char_decoding(self, encoding: np.array):
        return "".join([chr(c) for c in encoding])
        
    def bpe_encoding(self, text: str):
        tz = tiktoken.get_encoding("clk100-base")
        return np.array(tz.encode(text))
        
    def bpe_decoding(self, encoding: np.array):
        tz = tiktoken.get_encoding("clk100-base")
        self.bpe_tokens = tz.decode_tokens_bytes(list(encoding))
        return tz.decode(list(encoding))

    def show_bpe_tokens(self):
        print(self.bpe_tokens) if self.bpe_tokens else \
            print("No BPE tokens found. You have either not encoded any text or you have not used the BPE tokenizer.")

    def __str__(self):
        if self.tokenizer_type == "char":
            return f"Tokenizer type: {self.tokenizer_type}: Character Level Tokenizer"
        elif self.tokenizer_type == "bpe":
            return f"Tokenizer type: {self.tokenizer_type}: Byte Pair Encoding Tokenizer"
