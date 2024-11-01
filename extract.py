import torch
import numpy as np
import pandas as pd
from esm2 import ESM2
from torch.utils.data import Dataset, DataLoader

def tokenize(cog_list:list):
    cog_vocab = "JAKLBDYVTMNZWUOXCGEFHIPQRS-"
    regular_vocab = ['<pad>', '<mask>', '<cls>', '<eos>','<sep>', '<unk>']
    vocab = regular_vocab + list(cog_vocab)
    input_ids = []
    for cog in cog_list:
        if len(cog) > 1 and "<" not in cog:
            input_ids.append(vocab.index(cog[0]))
        else:
            input_ids.append(vocab.index(cog))
    return torch.tensor(input_ids, dtype=torch.long)

def get_trained_model(chpt_path, device):
    model = ESM2(
        num_layers=6,
        embed_dim=320,
        attention_heads=20,
        vocab_size=33,
        token_dropout=True,
    )
    model.load_state_dict(torch.load(chpt_path, map_location=device))
    return model.to(device)

class OD_Dataset(Dataset):
    def __init__(self, df):
        self.df = df
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        text = self.df['cogs'].values[idx].split(',')
        input_ids = tokenize(text)
        return {'input_ids': input_ids}
    
def extract_cog_embeddings(model, dataloader, device):
    model.eval()
    cog_embeddings = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            embeddings = model(input_ids, repr_layers=[6])['representations'][6].detach().cpu().numpy()
            cog_embeddings.append(embeddings.mean(axis=1))
    cog_embeddings = np.stack(cog_embeddings)
    return cog_embeddings.reshape(-1, 320)


df = pd.read_csv("data/train_data_protein.csv", sep=';')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_trained_model("cog_transformer.pt", device)
dataset = OD_Dataset(df)
dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)
cog_embeddings = extract_cog_embeddings(model, dataloader, device)
np.save("./data/cog_embeddings.npy", cog_embeddings)
