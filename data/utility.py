"""
A utility script to generate a test dataset from the main dataset
"""
import pandas as pd
import os
import random


def test_data_generator(main_data_file, test_paragraph_question_answer_file, n=100, append_to_existing=False):
    """
    Generate a test dataset from the main dataset
    :param main_data_file: str, path to the main dataset
    :param test_paragraph_file: str, path to the test paragraph dataset
    :param test_question_file: str, path to the test question dataset
    :param n: int, number of samples to generate
    :param append_to_existing: bool, append to existing test dataset

    """

    if append_to_existing:
        pass

    else:
        main_df = pd.read_csv(main_data_file)
        test_paragraphs = []
        test_questions = []
        test_answers = []

        for _ in range(n):
            k = random.randint(0, len(main_df))
            text = main_df.iloc[k].Paragraphs
            for _ in range(3):
                test_paragraphs.append(text)

            test_questions.append(main_df.iloc[k].Question1)
            test_questions.append(main_df.iloc[k].Question2)
            test_questions.append(main_df.iloc[k].Question3)
            test_answers.append(main_df.iloc[k].Answer1)
            test_answers.append(main_df.iloc[k].Answer2)
            test_answers.append(main_df.iloc[k].Answer3)

        test_df = pd.DataFrame(
            {'Paragraphs': test_paragraphs, 'Questions': test_questions, 'Answers': test_answers})
        test_df.to_csv(test_paragraph_question_answer_file, index=False)


def get_a_question(data_file, index=0):
    """
    Get a question from the dataset
    :param data_file: str, path to the dataset
    :param index: int, index of the question
    :return: str, the question
    """
    df = pd.read_csv(data_file)
    if index >= len(df):
        return None
    return df.iloc[index].Questions


def get_a_paragraph(data_file, index=0):
    """
    Get a paragraph from the dataset
    :param data_file: str, path to the dataset
    :param index: int, index of the paragraph
    :return: str, the paragraph
    """
    df = pd.read_csv(data_file)
    if index >= len(df):
        return None
    return df.iloc[index].Paragraphs


def main():
    # Load data
    main_data_file = os.path.join(
        'data', 'QA_dataset.csv')

    whole_df = pd.read_csv(main_data_file)
    paragraphs = []
    questions = []
    answers = []

    # Process the rows in the input CSV
    for i in range(len(whole_df)):
        for _ in range(3):
            paragraphs.append(whole_df.iloc[i].Paragraphs)
        questions.append(whole_df.iloc[i].Question1)
        questions.append(whole_df.iloc[i].Question2)
        questions.append(whole_df.iloc[i].Question3)
        answers.append(whole_df.iloc[i].Answer1)
        answers.append(whole_df.iloc[i].Answer2)
        answers.append(whole_df.iloc[i].Answer3)

        # Append the data to fresh_df
        temp_df = pd.DataFrame(
            {'Paragraphs': paragraphs, 'Questions': questions, 'Answers': answers})

        # After every 100 rows, save the current state of fresh_df to a CSV
        if (i + 1) % 100 == 0:
            print(f"Saving after {i + 1} iterations.")
            temp_df.to_csv(
                'data/Paragraphs_Questions_Answers.csv', index=False)

    # Final save to ensure all data is saved after processing
    temp_df.to_csv('data/Paragraphs_Questions_Answers.csv', index=False)

    # test_paragraph_question_answer_file = os.path.join(
    #     'data', 'test_Paragraphs_Questions_Answers.csv')

    # n = 10
    # test_data_generator(
    #     main_data_file, test_paragraph_question_answer_file, n, append_to_existing=False)

    # k = random.randint(0, n)
    # print(k)
    # print(get_a_question(test_data_file, k))
    # print(get_a_paragraph(test_data_file, k))

    '''
    245
    How do fully segregated busways differ from priority bus lanes in traffic management?
    The integration of public transport priorities in traffic management is vital, especially considering the challenges posed by the slower speeds and frequent stops of buses. Strategies like dedicated lanes and segregated busways help counteract the negative impacts of mixed traffic, encouraging increased bus usage. While priority bus lanes provide a basic solution, they come with limitations, such as hindering roadside access and susceptibility to intrusion. On the other hand, fully segregated busways, featuring central lanes and controlled pedestrian crossings at stations, offer a comprehensive approach. This model has demonstrated success in cities like Bogotá and Curitiba, showcasing efficiency improvements, emission reductions, and better air quality. By effectively managing and sustaining public transport operations, cities can achieve a balance between congestion reduction, air quality enhancement, and commuter convenience.
    Fully segregated busways offer a comprehensive solution with central lanes and controlled crossings, unlike priority lanes that have limitations and can impede roadside access.
    '''


if __name__ == '__main__':
    main()
