import os
import torch
import faiss
import json
from index_builder import load_data, load_model, vectorize_text


def load_metadata(metadata_file):
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(
            f"Metadata file not found at {metadata_file}. Please run the index_builder script first.")

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    return metadata


def sanity_check(metadata, model, tokenizer, paragraphs):
    # Check if the number of paragraphs matches the metadata count
    if metadata['indexed_count'] != len(paragraphs) or metadata['model_name'] != model or metadata['tokenizer_name'] != tokenizer:
        raise ValueError(
            "Mismatch between the number of paragraphs in the dataset and the metadata. Please rebuild the index.")


def retrieve_relevant_paragraphs(query, index, tokenizer, model, paragraphs, device, top_k=5):
    query_embedding = vectorize_text(
        query, tokenizer, model, device).reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)

    unique_paragraphs = set()  # Use a set to ensure uniqueness
    for i in indices[0]:
        unique_paragraphs.add(paragraphs[i])

    return list(unique_paragraphs)  # Convert back to a list


def main():
    data_file = os.path.join(
        'data', 'test_Paragraphs_Questions_Answers_Grades.csv')
    index_file = os.path.join(
        'index', 'test_Paragraphs_Questions_Answers_Grades_index.faiss')
    metadata_file = os.path.join(
        'index', 'test_Paragraphs_Questions_Answers_Grades_index_metadata.json')

    if not os.path.exists(index_file):
        raise FileNotFoundError(
            f"Index file not found at {index_file}. Please run the index_builder script first.")

    # Load data and metadata
    questions, paragraphs = load_data(data_file)
    metadata = load_metadata(metadata_file)

    model_name = 'sentence-transformers/all-MiniLM-L12-v2'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    tokenizer, model = load_model(model_name, device)

    # Perform sanity check
    sanity_check(metadata, model_name, model_name, paragraphs)

    # Load the existing index for querying
    index = faiss.read_index(index_file)

    # Example query
    query = "When did humanity's ecological footprint surpass Earth's capacity?"
    relevant_paragraphs = retrieve_relevant_paragraphs(
        query, index, tokenizer, model, paragraphs, device)

    print("Relevant paragraphs:")
    for paragraph in relevant_paragraphs:
        print(paragraph)


if __name__ == "__main__":
    main()
