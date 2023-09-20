from model import Model
from tokenizer import Tokenizer
from dataloader import Dataloader

model = Model("config.json", "bpe")
tokenizer = Tokenizer("bpe")
dataloader = Dataloader(input_file="poems.txt")
data = dataloader.extract()
data = tokenizer.encode(data)
#train_data, val_data = dataloader.get_train_val_split("bpe", data)
#model.train(train_file="train_data.txt", val_file="val_data.txt", load_model=True, save_path="LLM.pth.tar")
print(model.sample("A quick ", 50, load_model=True))