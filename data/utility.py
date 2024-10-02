"""
A utility script to generate a test dataset from the main dataset
"""
import pandas as pd
import os
import random


def test_data_generator(main_data_file, test_data_file, n=100, append_to_existing=False):
    """
    Generate a test dataset from the main dataset
    :param main_data_file: str, path to the main dataset
    :param test_data_file: str, path to the test dataset
    :param n: int, number of samples to take from the main dataset
    :param append_to_existing: bool, whether to append the test dataset to the existing test dataset
    :return: None
    """
    if append_to_existing:
        test_df = pd.read_csv(test_data_file)
        main_df = pd.read_csv(main_data_file)
        test_df = pd.concat([test_df, main_df.sample(n=n)])
    else:
        main_df = pd.read_csv(main_data_file)
        test_df = main_df.sample(n=n)

    test_df.to_csv(test_data_file, index=False)


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
        'data', 'Paragraphs_Questions_Answers_Grades.csv')
    test_data_file = os.path.join(
        'data', 'test_Paragraphs_Questions_Answers_Grades.csv')

    n = 100
    test_data_generator(main_data_file, test_data_file,
                        n=n, append_to_existing=False)
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
