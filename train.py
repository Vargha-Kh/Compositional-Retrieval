# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from model import CLIPModel
import open_clip
from tqdm import tqdm
import csv
import os


def train_model(model, train_dataloader, val_dataloader, num_epochs, device, csv_log_path='training_log.csv'):
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer('ViT-B-16')  # Adjust tokenizer based on your backbone if needed

    optimizer = optim.AdamW(model.parameters(), lr=5e-6, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # For mixed-precision training
    scaler = torch.cuda.amp.GradScaler()

    best_val_loss = float('inf')
    temperature = 0.07  # Temperature hyperparameter for scaling

    # Prepare CSV logger
    csv_columns = ['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy']
    if not os.path.exists(csv_log_path):
        with open(csv_log_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # Initialize tqdm progress bar for training
        train_progress = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training", leave=False)

        for batch in train_progress:
            optimizer.zero_grad()

            query_images = batch['query_image'].to(device)
            target_images = batch['target_image'].to(device)
            query_texts = batch['query_text']

            # Tokenize texts
            query_texts = tokenizer(query_texts).to(device)

            with torch.cuda.amp.autocast():
                # Forward pass
                query_image_features = model.model.encode_image(query_images)
                query_text_features = model.model.encode_text(query_texts)
                query_features = (query_image_features + query_text_features) / 2
                query_features = query_features / query_features.norm(dim=-1, keepdim=True)
                target_image_features = model.model.encode_image(target_images)
                target_image_features = target_image_features / target_image_features.norm(dim=-1, keepdim=True)

                # Compute similarity logits with temperature scaling
                logits_per_query = (query_features @ target_image_features.T) / temperature
                logits_per_target = (target_image_features @ query_features.T) / temperature

                # Labels
                labels = torch.arange(len(query_features)).to(device)

                # Compute loss in both directions
                loss_i = criterion(logits_per_query, labels)
                loss_t = criterion(logits_per_target, labels)
                loss = (loss_i + loss_t) / 2

            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Update training loss and accuracy
            train_loss += loss.item()

            # Compute accuracy (query to target)
            _, preds = torch.max(logits_per_query, dim=1)
            train_correct += torch.sum(preds == labels).item()
            train_total += labels.size(0)

            # Calculate running averages
            avg_loss = train_loss / (train_total / labels.size(0))
            avg_accuracy = train_correct / train_total

            # Update tqdm progress bar with running metrics
            train_progress.set_postfix({
                'Loss': f"{avg_loss:.4f}",
                'Acc': f"{avg_accuracy * 100:.2f}%"
            })

        avg_train_loss = train_loss / len(train_dataloader)
        train_accuracy = train_correct / train_total

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        # Initialize tqdm progress bar for validation
        val_progress = tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation", leave=False)

        with torch.no_grad():
            for batch in val_progress:
                query_images = batch['query_image'].to(device)
                target_images = batch['target_image'].to(device)
                query_texts = batch['query_text']

                # Tokenize texts
                query_texts = tokenizer(query_texts).to(device)

                # Forward pass
                query_image_features = model.model.encode_image(query_images)
                query_text_features = model.model.encode_text(query_texts)
                query_features = (query_image_features + query_text_features) / 2
                query_features = query_features / query_features.norm(dim=-1, keepdim=True)
                target_image_features = model.model.encode_image(target_images)
                target_image_features = target_image_features / target_image_features.norm(dim=-1, keepdim=True)

                # Compute similarity logits with temperature scaling
                logits_per_query = (query_features @ target_image_features.T) / temperature
                logits_per_target = (target_image_features @ query_features.T) / temperature

                # Labels
                labels = torch.arange(len(query_features)).to(device)

                # Compute loss in both directions
                loss_i = criterion(logits_per_query, labels)
                loss_t = criterion(logits_per_target, labels)
                loss = (loss_i + loss_t) / 2
                val_loss += loss.item()

                # Compute accuracy (query to target)
                _, preds = torch.max(logits_per_query, dim=1)
                val_correct += torch.sum(preds == labels).item()
                val_total += labels.size(0)

                # Calculate running averages
                avg_val_loss = val_loss / (val_total / labels.size(0))
                avg_val_accuracy = val_correct / val_total

                # Update tqdm progress bar with running metrics
                val_progress.set_postfix({
                    'Loss': f"{avg_val_loss:.4f}",
                    'Acc': f"{avg_val_accuracy * 100:.2f}%"
                })

        avg_val_loss = val_loss / len(val_dataloader)
        val_accuracy = val_correct / val_total

        # Adjust learning rate based on validation loss
        scheduler.step(avg_val_loss)

        # Print epoch summary
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy * 100:.2f}%")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy * 100:.2f}%")

        # Log metrics to CSV
        with open(csv_log_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writerow({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_accuracy': train_accuracy * 100,
                'val_loss': avg_val_loss,
                'val_accuracy': val_accuracy * 100
            })

        # Save the model with the best validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_weights.pth')
            print("Best model saved.")

    # Load the best model before returning
    model.load_state_dict(torch.load('best_weights.pth'))
    return model
