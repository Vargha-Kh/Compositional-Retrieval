# generate_embeds.py
import torch
from torch.utils.data import DataLoader
from datasets import QueryDataset, DatabaseDataset
from model import CLIPModel
import pandas as pd
import numpy as np
from tqdm import tqdm
import open_clip

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def collate_fn_query(batch):
    # Remove None samples
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    query_images = torch.stack([item['query_image'] for item in batch])
    query_texts = [item['query_text'] for item in batch]

    return {
        'query_image': query_images,
        'query_text': query_texts,
    }

def collate_fn_database(batch):
    # Remove None samples
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    images = torch.stack(batch)
    return images

def encode_queries(df: pd.DataFrame, image_root) -> np.ndarray:
    model = CLIPModel()
    model.load_state_dict(torch.load('best_weights.pth', map_location=device))
    model = model.to(device)
    model.eval()

    preprocess = model.eval_preprocess
    tokenizer = open_clip.get_tokenizer('ViT-B-16')

    dataset = QueryDataset(df, image_root=image_root, preprocess=preprocess)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn_query
    )

    all_embeddings = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding Queries"):
            if batch is None:
                continue

            query_images = batch['query_image'].to(device)
            query_texts = batch['query_text']

            with torch.cuda.amp.autocast():
                query_texts = tokenizer(query_texts).to(device)

                query_image_features = model.model.encode_image(query_images)
                query_text_features = model.model.encode_text(query_texts)
                query_features = (query_image_features + query_text_features) / 2
                query_features = query_features / query_features.norm(dim=-1, keepdim=True)

            all_embeddings.append(query_features.cpu().numpy())

    all_embeddings = np.vstack(all_embeddings)
    return all_embeddings

def encode_database(df: pd.DataFrame, image_root) -> np.ndarray:
    model = CLIPModel()
    model.load_state_dict(torch.load('best_weights.pth', map_location=device))
    model = model.to(device)
    model.eval()

    preprocess = model.eval_preprocess

    dataset = DatabaseDataset(df, image_root=image_root, preprocess=preprocess)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn_database
    )

    all_embeddings = []

    with torch.no_grad():
        for images in tqdm(dataloader, desc="Encoding Database Images"):
            if images is None:
                continue
            images = images.to(device)
            with torch.cuda.amp.autocast():
                image_features = model.model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            all_embeddings.append(image_features.cpu().numpy())

    all_embeddings = np.vstack(all_embeddings)
    return all_embeddings
