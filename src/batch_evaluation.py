import os
import pandas as pd
import torch
import logging
from tqdm import tqdm
from index_builder import load_model
from answer_generator import load_generation_model, generate_answer
from retriever import retrieve_relevant_paragraphs
from evaluation import evaluate_with_bleu, evaluate_with_semantic_similarity, compute_weighted_score
import faiss

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def evaluate_answers(data_file, index_file, model_name, gen_model_name, device):
    """ Evaluate answers from a CSV file and store scores in a new column """
    # Load data
    logging.info("Loading data...")
    df = pd.read_csv(data_file)

    # Load the retrieval model
    tokenizer, retrieval_model = load_model(model_name, device)

    # Load FAISS index
    logging.info(f"Loading FAISS index from {index_file}...")
    index = faiss.read_index(index_file)

    # Load the generation model
    gen_tokenizer, gen_model = load_generation_model(gen_model_name, device)

    # Prepare a new column for scores
    df['score'] = 0.0

    # Evaluate each question-answer pair with a progress bar
    logging.info("Starting evaluation of answers...")
    for i in tqdm(range(len(df)), desc="Evaluating Answers"):
        question = df.loc[i, 'Questions']
        test_answer = df.loc[i, 'Answers']

        # Retrieve relevant paragraphs
        relevant_paragraphs = retrieve_relevant_paragraphs(
            question, index, tokenizer, retrieval_model, df['Paragraphs'].tolist(), device)

        # Generate an answer based on the retrieved paragraphs
        generated_answer = generate_answer(
            question, relevant_paragraphs, gen_tokenizer, gen_model, device)

        # Evaluate the test answer against the generated answer
        bleu_score = evaluate_with_bleu(generated_answer, test_answer)
        semantic_similarity = evaluate_with_semantic_similarity(
            generated_answer, test_answer, retrieval_model, tokenizer, device)

        # Compute the weighted score
        weighted_score = compute_weighted_score(
            bleu_score, semantic_similarity)

        # Store the weighted score in the DataFrame
        df.loc[i, 'score'] = weighted_score

    # Save the updated DataFrame to the same CSV file
    df.to_csv(data_file, index=False)
    logging.info(f"Evaluation completed. Scores saved to {data_file}.")


def main():
    """ Main function to run the evaluation process """
    logging.info("Starting the evaluation script...")

    # Define file paths and model names
    data_file = os.path.join(
        'data', 'test_Paragraphs_Questions_Answers_Grades.csv')
    index_file = os.path.join(
        'index', 'test_Paragraphs_Questions_Answers_Grades_index.faiss')
    retrieval_model_name = 'sentence-transformers/all-MiniLM-L12-v2'
    generation_model_name = 'facebook/bart-large'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Run the evaluation
    evaluate_answers(data_file, index_file, retrieval_model_name,
                     generation_model_name, device)


if __name__ == "__main__":
    main()
