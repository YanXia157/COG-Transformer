import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold
from scipy.stats import pearsonr, spearmanr
import pandas as pd


data_df = pd.read_csv("data/train_data_protein.csv", sep=";")


def load_esm_embeddings(embed_dir):
    from glob import glob
    embeds = {}
    for path in glob(embed_dir + '/*.pt'):
        embeds[path.split('/')[-1].split('.')[0]] = torch.load(path)['mean_representations'][6].detach().cpu().numpy()
    return embeds


def encode_fix_length_vector(cog_list: list):
    cog_vocab = "JAKLBDYVTMNZWUOXCGEFHIPQRS-"
    cog_count_dict = {k:0 for k in cog_vocab}
    for cog in cog_list:
        if len(cog) != 1:
            if cog == "<pad>":
                cog_count_dict["-"] += 1
            else:
                for c in cog:
                    cog_count_dict[c] += 1
        else:
            cog_count_dict[cog] += 1
    return list(cog_count_dict.values())


def get_esm_embedding(protein_ids, embeds):
    feat = []
    for pid in protein_ids:
        if pid in embeds:
            feat.append(embeds[pid])
    if len(feat) == 0:
        return np.zeros_like(list(embeds.values())[0])  # Return a zero vector if no embedding is found
    return np.array(feat).mean(axis=0)


cog_embeddings = np.load("data/cog_embeddings.npy")
esm_embeddings = load_esm_embeddings("data/esm2_8m_embeddings")

X_pretrained = cog_embeddings


X_cog = data_df["cogs"].apply(lambda x: encode_fix_length_vector(x.split(",")))
X_cog = np.array(X_cog.tolist())


X_protein = data_df["proteins"].apply(lambda x: get_esm_embedding(x.split(","), esm_embeddings))
X_protein = np.array(X_protein.tolist())


kf = KFold(n_splits=5, random_state=42, shuffle=True)

all_pretrain_pred = []
all_cog_pred = []
all_protein_pred = []
all_labels = []
all_types = []

for types in ["d_lb", "d_gm2", "d_ph5", "d_urea"]:
    pretrain_pred = []
    cog_pred = []
    protein_pred = []
    labels = []
    y_lb = data_df[types].to_numpy()
    for train_index, test_index in kf.split(X_cog):
        X_cog_train, X_cog_test = X_cog[train_index], X_cog[test_index]
        X_protein_train, X_protein_test = X_protein[train_index], X_protein[test_index]
        X_pretrained_train, X_pretrained_test = X_pretrained[train_index], X_pretrained[test_index]
        X_cog_train = np.stack(X_cog_train)
        X_cog_test = np.stack(X_cog_test)
        X_protein_train = np.stack(X_protein_train)
        X_protein_test = np.stack(X_protein_test)
        assert X_cog.shape[0] == X_protein.shape[0] == X_pretrained.shape[0], "Data shapes do not match!"

        y_train, y_test = y_lb[train_index], y_lb[test_index]
        model_cog = ExtraTreesRegressor(n_estimators=32, random_state=42, n_jobs=-1)
        model_protein = ExtraTreesRegressor(n_estimators=32, random_state=42, n_jobs=-1)
        model_transfer = ExtraTreesRegressor(n_estimators=32, random_state=42, n_jobs=-1)
        model_cog.fit(X_cog_train, y_train)
        model_protein.fit(X_protein_train, y_train)
        model_transfer.fit(X_pretrained_train, y_train)
        y_pred_cog = model_cog.predict(X_cog_test)
        y_pred_protein = model_protein.predict(X_protein_test)
        y_pred_transfer = model_transfer.predict(X_pretrained_test)
        pretrain_pred.extend(y_pred_transfer)
        cog_pred.extend(y_pred_cog)
        protein_pred.extend(y_pred_protein)
        labels.extend(y_test)
    all_pretrain_pred.extend(pretrain_pred)
    all_cog_pred.extend(cog_pred)
    all_protein_pred.extend(protein_pred)
    all_labels.extend(labels)
    all_types.extend([types]*len(labels))

spr = spearmanr(all_protein_pred, all_labels)[0]
pcc = pearsonr(all_protein_pred, all_labels)[0]
print("ESM2-8M")
print(f"Spearman: {spr:.3f}")
print(f"Pearson: {pcc:.3f}")

spr = spearmanr(all_cog_pred, all_labels)[0]
pcc = pearsonr(all_cog_pred, all_labels)[0]
print("COG-OH")
print(f"Spearman: {spr:.3f}")
print(f"Pearson: {pcc:.3f}")

spr = spearmanr(all_pretrain_pred, all_labels)[0]
pcc = pearsonr(all_pretrain_pred, all_labels)[0]
print("COG-Transformer")
print(f"Spearman: {spr:.3f}")
print(f"Pearson: {pcc:.3f}")