# evaluate.py
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from generate_embeds import encode_queries, encode_database

def load_data(csv_file):
    """
    Load the data from a CSV file.

    Args:
        csv_file (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame with columns:
            - query_image: str
            - query_text: str
            - target_image: str
    """
    df = pd.read_csv(csv_file)
    return df

def calculate_top1_accuracy(predictions, ground_truth_images, database_df):
    """
    Calculate Top-1 accuracy and collect predicted image filenames.

    Args:
        predictions (np.ndarray): Array of predicted indices.
        ground_truth_images (list): List of ground truth target image filenames.
        database_df (pd.DataFrame): DataFrame containing the database images.

    Returns:
        float: Top-1 accuracy as a percentage.
        list: List of predicted image filenames.
    """
    correct = 0
    predicted_images = []
    for pred_idx, true_image in zip(predictions, ground_truth_images):
        pred_image = database_df.iloc[pred_idx]['target_image']
        predicted_images.append(pred_image)
        if pred_image == true_image:
            correct += 1
    accuracy = correct / len(ground_truth_images) * 100
    return accuracy, predicted_images

if __name__ == '__main__':
    # Paths to the CSV files
    test_csv = 'sample evaluation/data.csv'         # Replace with your test CSV file
    database_csv = 'dataset/data.csv'               # Replace with your database CSV file

    # Step 1: Load data
    test_df = load_data(test_csv)
    database_df = load_data(database_csv)
    database_image_path = './dataset/images'
    testset_image_path = './sample evaluation/images'

    # Extract necessary columns for embeddings
    query_df = test_df[['query_image', 'query_text']]
    database_images_df = database_df[['target_image']]

    # Step 2: Generate embeddings
    print("Generating database embeddings...")
    database_embeddings = encode_database(database_images_df, database_image_path)

    print("Generating query embeddings...")
    query_embeddings = encode_queries(query_df, testset_image_path)

    # Step 3: Calculate cosine similarity
    print("Calculating cosine similarities...")
    similarities = cosine_similarity(query_embeddings, database_embeddings)

    # Step 4: Get top-1 predictions
    print("Retrieving top-1 predictions...")
    predictions = np.argmax(similarities, axis=1)

    # Step 5: Calculate accuracy and collect predicted image filenames
    print("Calculating Top-1 accuracy...")
    ground_truth_images = test_df['target_image'].tolist()
    accuracy, predicted_images = calculate_top1_accuracy(predictions, ground_truth_images, database_df)

    print(f"Top-1 Retrieval Accuracy: {accuracy:.2f}%")

    # Print predictions along with ground truth
    print("\nPredictions:")
    for idx, (query_image, query_text, pred_image, true_image) in enumerate(zip(
            test_df['query_image'], test_df['query_text'], predicted_images, ground_truth_images)):
        print(f"Query {idx+1}:")
        print(f"  Query Image: {query_image}")
        print(f"  Query Text: {query_text}")
        print(f"  Predicted Image: {pred_image}")
        print(f"  Ground Truth Image: {true_image}")
        print(f"  Correct: {'Yes' if pred_image == true_image else 'No'}\n")
