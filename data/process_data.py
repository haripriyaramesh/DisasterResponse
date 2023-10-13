import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load and merge message and category data from CSV files.

    Args:
        messages_filepath (str): Path to the CSV file containing messages.
        categories_filepath (str): Path to the CSV file containing message categories.

    Returns:
        pd.DataFrame: A merged DataFrame containing messages and categories.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on="id", how="left")
    return df

def clean_data(df):
    """
    Clean and preprocess the DataFrame.

    Args:
        df (pd.DataFrame): The merged DataFrame containing messages and categories.

    Returns:
        pd.DataFrame: A cleaned DataFrame.
    """
    # Split categories into separate columns
    categories = df.categories.str.split(";", expand=True)
    
    # Extract category column names
    row = categories.iloc[0]
    category_colnames = list(map(lambda s: s[:-2], row))
    
    # Rename the category columns
    categories.columns = category_colnames

    # Convert category values to binary (0 or 1)
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1]).astype(int)
    
    # Drop the original categories column
    df = df.drop('categories', axis=1)
    
    # Concatenate the cleaned categories DataFrame
    df = pd.concat([df, categories], axis=1)
    
    # Drop duplicates
    df = df.drop_duplicates()

    return df

def save_data(df, database_filename):
    """
    Save the DataFrame to a SQLite database.

    Args:
        df (pd.DataFrame): The cleaned DataFrame.
        database_filename (str): Path to the SQLite database file.
    """
    db_path = 'sqlite:///' + database_filename
    engine = create_engine(db_path)
    df.to_sql('MessageCategories', engine, index=False)  

def main():
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to the database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

if __name__ == '__main__':
    main()
