from model import Model
from tokenizer import Tokenizer
from dataloader import Dataloader

model = Model("config.json", "bpe")
tokenizer = Tokenizer("bpe")
dataloader = Dataloader(input_file="poems.txt")
data = dataloader.extract()
data = tokenizer.encode(data)
model.train(data, load_model=False, save_path="LLM.pth.tar")