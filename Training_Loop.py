import mmap
import random
import torch
from gpt import GPT
import json
from tokenizer import Tokenizer

#load dataset and configs
tokenizer = Tokenizer("bpe")
with open("config.json", "r") as config_file:
    config = json.load(config_file)
config_file.close()
with open("poems.txt", "r", encoding="utf-8") as file:
    text = file.read()
    data = (tokenizer.encode(text)).to(torch.long)
    vocab = list(set(data.tolist()))
file.close()
vocab_size = tokenizer.vocab_size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_model = True

def save_checkpoint(state, filename="LLM.pth.tar"):
    print("- Saving Checkpoint -")
    torch.save(state, filename)

def load_checkpoint(checkpoint):
    print("- Loading Checkpoint -")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

'''def get_chunk(split):
    with open("poems.txt", "rb") as file:
        with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        file_size = len(mm)
        start_ix = random.randint(0, (file_size) - config["seq_length"] * config["batch_size"])
        mm.seek(start_ix)
        chunk = mm.read(config["seq_length"] * config["batch_size"] - 1).decode("utf-8", errors="ignore").replace('\r','')
        chunk = tokenizer.encode(chunk)
    return chunk'''


def get_batch(split):
    data = train_data if split == "train" else val_data
    index = torch.randint(len(data) - config["seq_length"], (config["batch_size"],))
    context = torch.stack([data[i:i+config["seq_length"]] for i in index])
    target = torch.stack([data[i+1:i+config["seq_length"]+1] for i in index])
    context, target = context.to(device), target.to(device)
    return context, target

model = GPT(
    vocab_size=vocab_size,
    seq_length=config["seq_length"],
    embed_size=config["embed_size"],
    num_layers=config["num_layers"],
    num_heads=config["num_heads"]
)
model = model.to(device)

@torch.no_grad()
def eval_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(100)
        for i in range(100):
            context, target = get_batch(split)
            _, loss = model(context, target)
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

if load_model:
    load_checkpoint(torch.load("LLM.pth.tar"))
optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
for epoch in range(config["epochs"]):
    if not epoch % 100:
        losses = eval_loss()
        print(f'Epoch: {epoch}, Train loss: {losses["train"]}, Val loss: {losses["val"]}')
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint)
    contextb, targetb = get_batch("train")
    logits, loss = model(contextb, targetb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1,1), dtype=torch.long, device=device)
generated_seq = tokenizer.decode(model.sample(context, max_tokens=1000)[0])
print(generated_seq)