from tokenizer import Tokenizer
from gpt import GPT
import torch
import torch.optim as optim
import json
import mmap
import random
import os

class Model:
    def __init__(self, config_file: str, tokenizer_type: str):
        self.tokenizer = Tokenizer(tokenizer_type)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert os.path.exists(config_file), "Model Configuration file doesn't exist"
        with open(config_file, "r") as configfile:
            self.config = json.load(configfile)
        configfile.close()
        self.vocab_size = self.tokenizer.vocab_size
        self.model = GPT(
            vocab_size=self.vocab_size,
            seq_length=self.config["seq_length"],
            embed_size=self.config["embed_size"],
            num_layers=self.config["num_layers"],
            num_heads=self.config["num_heads"]
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])

    # save model
    def save_checkpoint(self, state, filename="LLM.pth.tar"):
        print("- Saving Checkpoint -")
        torch.save(state, filename)

    # load model
    def load_checkpoint(self, checkpoint):
        print("- Loading Checkpoint -")
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    # train model
    def train(self, train_file: str, val_file: str, load_model=True,
              load_path="LLM.pth.tar", save_path="LLM.pth.tar"):

        # get random data chunks from file
        def get_data_chunk(split: str, train_file: str, val_file: str):
            file = train_file if split == "train" else val_file
            with open(file, "rb") as current_file:
                with mmap.mmap(current_file.fileno(), length=0, access=mmap.ACCESS_READ) as m:
                    start_idx = random.randint(0, len(m) - self.config["seq_length"] * self.config["batch_size"])
                    m.seek(start_idx)
                    context = (m.read(self.config["seq_length"] * self.config["batch_size"] - 1)).decode(
                        "utf-8", errors="ignore").replace("\r",'')
                    context = self.tokenizer.encode(context)
            return context

        # get batches of context and targets sequentially
        def get_batch(split):
            data = get_data_chunk(split, train_file, val_file)
            index = torch.randint(len(data)-self.config["seq_length"], (self.config["batch_size"],))
            context = torch.stack([data[i:i+self.config["seq_length"]] for i in index])
            target = torch.stack([data[i+1:i+self.config["seq_length"] + 1] for i in index])
            context, target = context.to(self.device), target.to(self.device)
            return context, target

        # evaluate loss on train and val splits. Doesn't require torch to track gradients
        @torch.no_grad()
        def eval_loss():
            out = {}
            self.model.eval()
            for split in ['train', 'val']:
                losses = torch.zeros(100)
                for i in range(100):
                    context, target = get_batch(split)
                    _, loss = self.model(context, target)
                    losses[i] = loss.item()
                out[split] = losses.mean()
            self.model.train()
            return out

        # training loop
        if load_model:
            self.load_checkpoint(torch.load(load_path))

        for epoch in range(self.config["epochs"]):
            if not epoch % 100:
                losses = eval_loss()
                print(f'Epoch: {epoch}, Train loss: {losses["train"]}, Val loss: {losses["val"]}')
                checkpoint = {'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}
                self.save_checkpoint(checkpoint, save_path)
            contextb, targetb = get_batch("train")
            logits, loss = self.model(contextb, targetb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

    # sample sequences from the model
    def sample(self, context: str, max_tokens=50, top_k=5, load_model=True, load_path="LLM.pth.tar"):
        if load_model:
            self.load_checkpoint(torch.load(load_path))
        context = self.tokenizer.encode(context).to(self.device)
        context = context.unsqueeze(0)
        generated_seq = self.tokenizer.decode(self.model.sample(context, max_tokens, top_k)[0])
        return generated_seq