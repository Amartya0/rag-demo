import os
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy as np
import time
from tqdm import tqdm  # For progress bar

import warnings
warnings.filterwarnings("ignore", category=FutureWarning,
                        message=r".*clean_up_tokenization_spaces*.")


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df['Questions'].tolist(), df['Paragraphs'].tolist()


def load_model(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    return tokenizer, model


def vectorize_text(text, tokenizer, model, device):
    tokens = tokenizer(text, return_tensors='pt', truncation=True,
                       padding=True, max_length=512).to(device)
    with torch.no_grad():
        embeddings = model(**tokens).last_hidden_state.mean(dim=1).squeeze()
    return embeddings.cpu().numpy()


def load_metadata(metadata_path):
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return None


def save_metadata(metadata_path, indexed_count, model_name, tokenizer_name):
    metadata = {
        'indexed_count': indexed_count,
        'model_name': model_name,
        'tokenizer_name': tokenizer_name
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)


def create_index(vectorized_texts, embedding_dim, index_path):
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(vectorized_texts)
    faiss.write_index(index, index_path)


def modify_index(index, new_vectorized_texts, index_path):
    index.add(new_vectorized_texts)
    faiss.write_index(index, index_path)


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
        index = faiss.read_index(index_path)
    else:
        if rebuild_index:
            print(
                "Model, tokenizer has changed, or dataset size has decreased. Rebuilding the index...")
        index = None

    # Handle case where new data has been added
    if index is None:
        concatenated_texts = [q + " " + p for q,
                              p in zip(questions, paragraphs)]

        vectorized_texts = np.array(
            [vectorize_text(text, tokenizer, model, device) for text in tqdm(
                concatenated_texts, desc="Indexing Progress")]
        )
        embedding_dim = vectorized_texts.shape[1]

        create_index(vectorized_texts, embedding_dim, index_path)
        print("New index created with all data.")
        save_metadata(metadata_path, current_count,
                      model_name, tokenizer_name)
        print("Metadata updated successfully.")

    else:
        if current_count > indexed_count:
            new_questions = questions[indexed_count:]
            new_paragraphs = paragraphs[indexed_count:]
            concatenated_texts = [q + " " + p for q,
                                  p in zip(new_questions, new_paragraphs)]

            vectorized_texts = np.array(
                [vectorize_text(text, tokenizer, model, device) for text in tqdm(
                    concatenated_texts, desc="Indexing Progress")]
            )
            embedding_dim = vectorized_texts.shape[1]
            modify_index(index, vectorized_texts, index_path)
            print("Only new values added to the existing index.")

            save_metadata(metadata_path, current_count,
                          model_name, tokenizer_name)
            print("Metadata updated successfully.")
        else:
            print("Index is already up to date.")


def main():
    data_file = os.path.join('data', 'Paragraphs_Questions.csv')
    index_file = os.path.join('index', 'Paragraphs_Questions_index.faiss')
    metadata_file = os.path.join(
        'index', 'Paragraphs_Questions_index_metadata.json')

    questions, paragraphs = load_data(data_file)
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    tokenizer, model = load_model(model_name, device)

    start = time.time()
    create_or_modify_index(questions, paragraphs, tokenizer, model,
                           index_file, metadata_file, model_name, model_name, device)
    end = time.time()

    print(f'Indexing took {end - start:.2f} seconds.')


if __name__ == "__main__":
    main()
