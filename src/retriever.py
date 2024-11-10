import os
import torch
import faiss
import json
import logging
from index_builder import load_data, load_model, vectorize_text

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("retriever.log"),
        logging.StreamHandler()
    ]
)


def load_metadata(metadata_file):
    if not os.path.exists(metadata_file):
        logging.error(
            f"Metadata file not found at {metadata_file}. Please run the index_builder script first.")
        raise FileNotFoundError(
            f"Metadata file not found at {metadata_file}. Please run the index_builder script first.")

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    logging.info("Metadata loaded successfully.")
    return metadata


def sanity_check(metadata, model, tokenizer, paragraphs):
    if metadata['indexed_count'] != len(paragraphs) or metadata['model_name'] != model or metadata['tokenizer_name'] != tokenizer:
        logging.error(
            "Mismatch between the number of paragraphs in the dataset and the metadata. Please rebuild the index.")
        raise ValueError(
            "Mismatch between the number of paragraphs in the dataset and the metadata. Please rebuild the index.")
    logging.info("Sanity check passed.")


def retrieve_relevant_paragraphs(query, index, tokenizer, model, paragraphs, device, top_k=5):
    logging.info("Vectorizing query for retrieval.")
    query_embedding = vectorize_text(
        query, tokenizer, model, device).reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)

    unique_paragraphs = set()  # Ensure unique paragraphs
    for i in indices[0]:
        unique_paragraphs.add(paragraphs[i])

    logging.info(f"Retrieved {len(unique_paragraphs)} unique paragraphs.")
    return list(unique_paragraphs)  # Convert back to list


def main():
    data_file = os.path.join(
        'data', 'test_Paragraphs_Questions_Answers.csv')
    index_file = os.path.join(
        'index', 'test_Paragraphs_Questions_Answers_index.faiss')
    metadata_file = os.path.join(
        'index', 'test_Paragraphs_Questions_Answers_index_metadata.json')

    if not os.path.exists(index_file):
        logging.error(
            f"Index file not found at {index_file}. Please run the index_builder script first.")
        raise FileNotFoundError(
            f"Index file not found at {index_file}. Please run the index_builder script first.")

    logging.info("Loading data and metadata.")
    questions, paragraphs = load_data(data_file)
    metadata = load_metadata(metadata_file)

    model_name = 'sentence-transformers/all-MiniLM-L12-v2'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f'Using device: {device}')

    tokenizer, model = load_model(model_name, device)

    # Perform sanity check
    sanity_check(metadata, model_name, model_name, paragraphs)

    # Load the existing index for querying
    index = faiss.read_index(index_file)
    logging.info("Index loaded successfully.")

    # Example query
    query = "Why is the heat loss calculation often done by month?"
    logging.info(f"Retrieving relevant paragraphs for query: '{query}'")
    relevant_paragraphs = retrieve_relevant_paragraphs(
        query, index, tokenizer, model, paragraphs, device)

    logging.info("Relevant paragraphs retrieved:")
    for paragraph in relevant_paragraphs:
        print(paragraph, end='\n\n')


if __name__ == "__main__":
    main()
