import os
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy as np
import time
from tqdm import tqdm
import logging
import warnings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("indexing.log"),
        logging.StreamHandler()
    ]
)

warnings.filterwarnings("ignore", category=FutureWarning,
                        message=r".*clean_up_tokenization_spaces*.")


def load_data(file_path):
    logging.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    return df['Questions'].tolist(), df['Paragraphs'].tolist()


def load_model(model_name, device):
    logging.info(f"Loading model {model_name} on device {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    return tokenizer, model


def vectorize_text(text, tokenizer, model, device):
    tokens = tokenizer(text, return_tensors='pt', truncation=True,
                       padding=True, max_length=512).to(device)
    with torch.no_grad():
        embeddings = model(**tokens).last_hidden_state.mean(dim=1).squeeze()
    return embeddings.cpu().numpy()


def batch_vectorize_texts(texts, tokenizer, model, device, batch_size=16):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        tokens = tokenizer(batch, return_tensors='pt', truncation=True,
                           padding=True, max_length=512).to(device)
        with torch.no_grad():
            batch_embeddings = model(
                **tokens).last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)


def load_metadata(metadata_path):
    if os.path.exists(metadata_path):
        logging.info(f"Loading metadata from {metadata_path}")
        with open(metadata_path, 'r') as f:
            return json.load(f)
    logging.info("No metadata file found; proceeding without it.")
    return None


def save_metadata(metadata_path, indexed_count, model_name, tokenizer_name):
    metadata = {
        'indexed_count': indexed_count,
        'model_name': model_name,
        'tokenizer_name': tokenizer_name
    }
    logging.info(f"Saving metadata to {metadata_path}")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)


def create_index(vectorized_texts, embedding_dim, index_path):
    logging.info(
        f"Creating a new index with embedding dimension {embedding_dim}")
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(vectorized_texts)
    faiss.write_index(index, index_path)
    logging.info(f"New index saved to {index_path}")


def modify_index(index, new_vectorized_texts, index_path):
    logging.info("Adding new data to the existing index.")
    index.add(new_vectorized_texts)
    faiss.write_index(index, index_path)
    logging.info(f"Updated index saved to {index_path}")


def check_for_rebuild(metadata, model_name, tokenizer_name):
    return metadata is None or metadata['model_name'] != model_name or metadata['tokenizer_name'] != tokenizer_name


def create_or_modify_index(questions, paragraphs, tokenizer, model, index_path, metadata_path, model_name, tokenizer_name, device):
    metadata = load_metadata(metadata_path)
    index_exists = os.path.exists(index_path)
    current_count = len(questions)

    if metadata:
        indexed_count = metadata['indexed_count']
        rebuild_index = check_for_rebuild(
            metadata, model_name, tokenizer_name) or (current_count < indexed_count)
    else:
        rebuild_index = True
        indexed_count = 0

    if index_exists and not rebuild_index:
        logging.info("Loading existing index.")
        index = faiss.read_index(index_path)
    else:
        if rebuild_index:
            logging.info(
                "Model/tokenizer has changed, or dataset size has decreased. Rebuilding the index...")
        index = None

    if index is None:
        concatenated_texts = [q + " " + p for q,
                              p in zip(questions, paragraphs)]
        vectorized_texts = batch_vectorize_texts(
            concatenated_texts, tokenizer, model, device)
        embedding_dim = vectorized_texts.shape[1]

        create_index(vectorized_texts, embedding_dim, index_path)
        logging.info("New index created with all data.")
        save_metadata(metadata_path, current_count, model_name, tokenizer_name)
        logging.info("Metadata updated successfully.")

    else:
        if current_count > indexed_count:
            new_questions = questions[indexed_count:]
            new_paragraphs = paragraphs[indexed_count:]
            concatenated_texts = [q + " " + p for q,
                                  p in zip(new_questions, new_paragraphs)]

            vectorized_texts = batch_vectorize_texts(
                concatenated_texts, tokenizer, model, device)
            modify_index(index, vectorized_texts, index_path)
            logging.info("Only new values added to the existing index.")
            save_metadata(metadata_path, current_count,
                          model_name, tokenizer_name)
            logging.info("Metadata updated successfully.")
        else:
            logging.info("Index is already up to date.")


def main():
    data_file = os.path.join(
        'data', 'Paragraphs_Questions_Answers.csv')
    index_file = os.path.join(
        'index', 'Paragraphs_Questions_Answers_index.faiss')
    metadata_file = os.path.join(
        'index', 'Paragraphs_Questions_Answers_index_metadata.json')

    questions, paragraphs = load_data(data_file)
    model_name = 'sentence-transformers/all-MiniLM-L12-v2'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f'Using device: {device}')

    tokenizer, model = load_model(model_name, device)

    start = time.time()
    create_or_modify_index(questions, paragraphs, tokenizer, model,
                           index_file, metadata_file, model_name, model_name, device)
    end = time.time()

    logging.info(f'Indexing took {end - start:.2f} seconds.')


if __name__ == "__main__":
    main()
