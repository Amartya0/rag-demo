import os
import pandas as pd
import torch
import logging
from tqdm import tqdm
# Import batch_vectorize_texts
from index_builder import load_model, batch_vectorize_texts
from answer_generator import load_generation_model, generate_answer
from retriever import retrieve_relevant_paragraphs
from evaluation import evaluate_with_bleu, evaluate_with_semantic_similarity, compute_weighted_score
import faiss

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def evaluate_answers(data_file, index_file, model_name, gen_model_name, device, output_file):
    """ Evaluate answers from a CSV file and store scores in a separate CSV file """
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

    # Create a list to store evaluation results
    evaluation_results = []

    # Evaluate each question-answer pair with a progress bar
    logging.info("Starting evaluation of answers...")
    for i in tqdm(range(len(df)), desc="Evaluating Answers"):
        paragraph = df.loc[i, 'Paragraphs']
        question = df.loc[i, 'Questions']
        test_answer = df.loc[i, 'Answers']

        # Retrieve relevant paragraphs
        relevant_paragraphs = retrieve_relevant_paragraphs(
            question, index, tokenizer, retrieval_model, df['Paragraphs'].tolist(), device)

        # Use batch_vectorize_texts to get embeddings of the retrieved paragraphs
        paragraph_embeddings = batch_vectorize_texts(
            relevant_paragraphs, tokenizer, retrieval_model, device)

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

        # Store the evaluation results for this row
        evaluation_results.append({
            'Paragraph': paragraph,
            'Question': question,
            'Answer': test_answer,
            'Generated_Answer': generated_answer,
            'BLEU_Score': bleu_score,
            'Semantic_Similarity': semantic_similarity,
            'Weighted_Score': weighted_score
        })

        # Every 10 evaluations, save the results to the output file
        if (i + 1) % 10 == 0:
            logging.info(f"Saving results after processing {i + 1} rows.")
            # Append the results to the CSV file
            results_df = pd.DataFrame(evaluation_results)
            if not os.path.exists(output_file):
                results_df.to_csv(output_file, index=False)
            else:
                results_df.to_csv(output_file, mode='a',
                                  header=False, index=False)

            # Clear the evaluation results list to prepare for the next batch
            evaluation_results = []

    # If there are any remaining results after the loop ends (less than 10 evaluations)
    if evaluation_results:
        logging.info("Saving remaining results after processing all rows.")
        results_df = pd.DataFrame(evaluation_results)
        results_df.to_csv(output_file, mode='a',
                          header=not os.path.exists(output_file), index=False)

    logging.info(f"Evaluation completed. Results saved to {output_file}.")


def main():
    """ Main function to run the evaluation process """
    logging.info("Starting the evaluation script...")

    # Define file paths and model names
    data_file = os.path.join(
        'data', 'Paragraphs_Questions_Answers.csv')
    index_file = os.path.join(
        'index', 'Paragraphs_Questions_Answers_index.faiss')
    output_file = os.path.join(
        'output', 'Grades.csv')  # New file to store the evaluation results
    retrieval_model_name = 'sentence-transformers/all-MiniLM-L12-v2'
    generation_model_name = 'facebook/bart-large'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Run the evaluation and store results in a new CSV file
    evaluate_answers(data_file, index_file, retrieval_model_name,
                     generation_model_name, device, output_file)


if __name__ == "__main__":
    main()
