import torch
import tiktoken

''' Currently only supports chararcter level ascii tokenizer and openAI's byte pair encoding (BPE) 
tokenizer tiktoken that encodes sequences at sub-word level.
NOTE : Change the vocab size if using custom input dataset that involves language other than English, if using tiktoken'''

class Tokenizer:
    def __init__(self, tokenizer_type: str):
        self.tokenizer_type = tokenizer_type
        if self.tokenizer_type != "char" or self.tokenizer_type != "bpe":
            assert self.tokenizer_type == "char" or self.tokenizer_type == "bpe", 'Tokenizer type must be "char" or "bpe"'
        self.vocab_size = 96 if self.tokenizer_type == "char" else 100277
        self.unk_token = 95
        self.bpe_tokens = set()

    def encode(self, text: str):
        if self.tokenizer_type == "char":
            return self.char_encoding(text)
        elif self.tokenizer_type == "bpe":
            return self.bpe_encoding(text)

    def decode(self, encoding: torch.tensor):
        if self.tokenizer_type == "char":
            return self.char_decoding(encoding)
        elif self.tokenizer_type == "bpe":
            return self.bpe_decoding(encoding)
            
    def char_encoding(self, text: str):
        return torch.tensor([(ord(c)-32 if ord(c) in range(32,127) else self.unk_token) for c in text], dtype=torch.long)
    
    def char_decoding(self, encoding: torch.tensor):
        return "".join([chr(int(c)+32 if c!=self.unk_token else 10) for c in encoding])
        
    def bpe_encoding(self, text: str):
        tz = tiktoken.get_encoding("cl100k_base")
        return torch.tensor(tz.encode(text), dtype=torch.long)
      
    def bpe_decoding(self, encoding: torch.tensor):
        tz = tiktoken.get_encoding("cl100k_base")
        self.bpe_tokens = tz.decode_tokens_bytes(list(encoding))
        return tz.decode(list(encoding))

    def show_bpe_tokens(self):
        assert len(self.bpe_tokens) != 0, \
            "No BPE tokens found. You have either not encoded any text or you have not used the BPE tokenizer."
        print(self.bpe_tokens)

    def __str__(self):
        if self.tokenizer_type == "char":
            return f"Tokenizer type: {self.tokenizer_type}: Character Level Tokenizer"
        elif self.tokenizer_type == "bpe":
            return f"Tokenizer type: {self.tokenizer_type}: Byte Pair Encoding Tokenizer (SubWord)"
