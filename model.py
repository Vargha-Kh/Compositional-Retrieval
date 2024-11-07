# model.py
import torch
import torch.nn as nn
import open_clip
from preprocessing import get_train_preprocess, get_eval_preprocess
import torch.nn.functional as F

class CLIPModel(nn.Module):
    def __init__(self, model_name='ViT-B-32', pretrained='openai', image_size=512, dropout_prob=0.65):
        super(CLIPModel, self).__init__()
        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.image_size = image_size
        self.dropout_prob = dropout_prob
        # Use custom preprocessing functions
        self.train_preprocess = get_train_preprocess(self.image_size)
        self.eval_preprocess = get_eval_preprocess(self.image_size)
        # Adjust positional embeddings
        self._resize_positional_embeddings()

    def _resize_positional_embeddings(self):
        # Calculate the number of patches
        patch_size = self.model.visual.conv1.kernel_size[0]  # Usually 32 or 16
        grid_size = self.image_size // patch_size
        expected_tokens = grid_size ** 2 + 1  # +1 for class token

        # Get current positional embeddings
        current_positional_embeddings = self.model.visual.positional_embedding  # Shape: (current_tokens, dim)

        if expected_tokens != current_positional_embeddings.shape[0]:
            # Interpolate positional embeddings
            print("Resizing positional embeddings...")
            class_pos = current_positional_embeddings[0:1]
            grid_pos = current_positional_embeddings[1:]
            grid_size_current = int(grid_pos.shape[0] ** 0.5)
            grid_pos = grid_pos.reshape(1, grid_size_current, grid_size_current, -1).permute(0, 3, 1, 2)

            grid_pos = torch.nn.functional.interpolate(
                grid_pos,
                size=(grid_size, grid_size),
                mode='bilinear',
                align_corners=False
            )

            grid_pos = grid_pos.permute(0, 2, 3, 1).reshape(1, grid_size * grid_size, -1)
            new_positional_embeddings = torch.cat([class_pos.unsqueeze(0), grid_pos], dim=1)
            self.model.visual.positional_embedding = nn.Parameter(new_positional_embeddings.squeeze(0))

    def forward(self, image, text):
        image_features = self.model.encode_image(image)
        text_features = self.model.encode_text(text)
        if self.training:
            image_features = F.dropout(image_features, p=self.dropout_prob, training=True)
            text_features = F.dropout(text_features, p=self.dropout_prob, training=True)
        return image_features, text_features
