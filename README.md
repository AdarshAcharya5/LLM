# LLM
A Not-So-Large-Yet Language Model
To run : 
1. Make a gpu enabled env/venv.
2. Clone this repo
3. pip install requirements.txt
4. Check the run.py file for general training and sampling example(s).

DataLoader class implements functions to load your own txt file, merge all txt files in a source directory into a single file and split an input data file into train and val files. 
As for the tokenizer, currently the model only supports character level ascii mapping and OpenAI's tiktoken BPE tokenizer, which works on a sub-word level.

Additionally, If you're a stranger and found your way here, I assume you're here because you're interested in language models or all things NLP. Instead of a readme, I think you should 
explore the code yourself, which could help in getting a better understanding of how the model works :). Cheers!
