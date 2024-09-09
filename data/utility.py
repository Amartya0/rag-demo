"""
A utility script to generate a test dataset from the main dataset
"""
import pandas as pd
import os


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


def main():
    # Load data
    main_data_file = os.path.join('data', 'Paragraphs_Questions.csv')
    test_data_file = os.path.join('data', 'test_Paragraphs_Questions.csv')

    # test_data_generator(main_data_file, test_data_file, n=100, append_to_existing=False)
    # print(get_a_question(test_data_file, 5))


if __name__ == '__main__':
    main()
