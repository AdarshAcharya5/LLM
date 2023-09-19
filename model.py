from tokenizer import Tokenizer
from gpt import GPT
import torch
import torch.optim as optim
import json

class Model:

    def __init__(self, config_file: str, tokenizer_type: str):
        self.tokenizer = Tokenizer(tokenizer_type)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open(config_file, "r") as config_file:
            self.config = json.load(config_file)
        config_file.close()
        self.vocab_size = self.tokenizer.vocab_size
        self.model = GPT(
            vocab_size=self.vocab_size,
            seq_length=self.config["seq_length"],
            embed_size=self.config["embed_size"],
            num_layers=self.config["num_layers"],
            num_heads=self.config["num_heads"]
        ).to(self.device)

    def save_checkpoint(self, state, filename="LLM.pth.tar"):
        print("- Saving Checkpoint -")
        torch.save(state, filename)

    def load_checkpoint(self, optimizer, checkpoint):
        print("- Loading Checkpoint -")
        self.model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    def train(self, data: torch.tensor, load_model=True, load_path="LLM.pth.tar", save_path="LLM.pth.tar"):

        def get_batch(split):
            data = train_data if split == "train" else val_data
            index = torch.randint(len(data)-self.config["seq_length"], (self.config["batch_size"],))
            context = torch.stack([data[i:i+self.config["seq_length"]] for i in index])
            target = torch.stack([data[i+1:i+self.config["seq_length"] + 1] for i in index])
            context, target = context.to(self.device), target.to(self.device)
            return context, target

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

        n = int(0.9 * len(data))
        train_data, val_data = data[:n], data[n:]
        optimizer = optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])
        if load_model:
            print("- Loading Checkpoint -")
            self.load_checkpoint(optimizer, torch.load(load_path))

        for epoch in range(self.config["epochs"]):
            if not epoch % 100:
                losses = eval_loss()
                print(f'Epoch: {epoch}, Train loss: {losses["train"]}, Val loss: {losses["val"]}')
                checkpoint = {'state_dict': self.model.state_dict(), 'optimizer': optimizer.state_dict()}
                self.save_checkpoint(checkpoint, save_path)
            contextb, targetb = get_batch("train")
            logits, loss = self.model(contextb, targetb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    def sample(self, context: str, max_tokens=500):
        context = self.tokenizer.encode(context)
        context = context.unsqueeze(0).to(self.device)
        generated_seq = self.tokenizer.decode(self.model.sample(context, max_tokens)[0])
        return generated_seq




