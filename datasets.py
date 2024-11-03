# datasets.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError


class CompositionalDataset(Dataset):
    def __init__(self, dataframe, image_root, preprocess, mode='train'):
        self.dataframe = dataframe
        self.image_root = image_root
        self.preprocess = preprocess
        self.mode = mode

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if idx >= len(self):
            idx = idx % len(self)
        row = self.dataframe.iloc[idx]
        query_image_name = row['query_image'].strip()
        target_image_name = row['target_image'].strip()
        query_text = row['query_text']

        query_image_path = os.path.join(self.image_root, query_image_name)
        target_image_path = os.path.join(self.image_root, target_image_name)

        # Verify file existence
        if not os.path.exists(query_image_path):
            print(f"Query image file does not exist: {query_image_path}")
            return self.__getitem__((idx + 1) % len(self))
        if not os.path.exists(target_image_path):
            print(f"Target image file does not exist: {target_image_path}")
            return self.__getitem__((idx + 1) % len(self))

        # Choose the appropriate preprocess function
        if self.mode == 'train':
            preprocess = self.preprocess['train']
        else:
            preprocess = self.preprocess['eval']

        # Handle exceptions during image loading
        try:
            query_image = preprocess(Image.open(query_image_path).convert('RGB'))
        except (OSError, UnidentifiedImageError) as e:
            print(f"Error opening query image {query_image_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))
        try:
            target_image = preprocess(Image.open(target_image_path).convert('RGB'))
        except (OSError, UnidentifiedImageError) as e:
            print(f"Error opening target image {target_image_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        return {
            'query_image': query_image,
            'query_text': query_text,
            'target_image': target_image
        }


class QueryDataset(Dataset):
    def __init__(self, dataframe, image_root, preprocess):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_root = image_root
        self.preprocess = preprocess

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        query_image_name = row['query_image'].strip()
        query_text = row['query_text']

        query_image_path = os.path.join(self.image_root, query_image_name)

        # Handle exceptions during image loading
        try:
            query_image = self.preprocess(Image.open(query_image_path).convert('RGB'))
        except (OSError, UnidentifiedImageError) as e:
            print(f"Error opening query image {query_image_path}: {e}")
            # Return None to skip this sample
            return None

        return {
            'query_image': query_image,
            'query_text': query_text,
        }


class DatabaseDataset(Dataset):
    def __init__(self, dataframe, image_root, preprocess):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_root = image_root
        self.preprocess = preprocess

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_name = self.dataframe.iloc[idx]['target_image'].strip()
        image_path = os.path.join(self.image_root, image_name)

        # Handle exceptions during image loading
        try:
            image = self.preprocess(Image.open(image_path).convert('RGB'))
        except (OSError, UnidentifiedImageError) as e:
            print(f"Error opening image {image_path}: {e}")
            # Return None to skip this sample
            return None

        return image
