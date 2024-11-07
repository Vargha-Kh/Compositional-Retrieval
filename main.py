# main.py
import pandas as pd
from torch.utils.data import DataLoader
from datasets import CompositionalDataset
from model import CLIPModel
from train import train_model
from sklearn.model_selection import train_test_split
import torch

def main():
    # Load your dataset CSV
    df = pd.read_csv('./dataset/data.csv')

    # Split into training and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

    # Initialize model
    model = CLIPModel()

    # Create preprocessing dictionary
    preprocess = {
        'train': model.train_preprocess,
        'eval': model.eval_preprocess
    }

    # Create datasets
    image_root = './dataset/images'  # Replace with your image directory
    train_dataset = CompositionalDataset(train_df, image_root=image_root, preprocess=preprocess, mode='train')
    val_dataset = CompositionalDataset(val_df, image_root=image_root, preprocess=preprocess, mode='eval')

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=3, shuffle=False, num_workers=4, pin_memory=True)

    # Train the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 25
    model = train_model(model, train_dataloader, val_dataloader, num_epochs=num_epochs, device=device)

    # Save the trained model
    torch.save(model.state_dict(), 'weights.pth')

if __name__ == '__main__':
    main()
