import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories datasets and merge them.
    
    Args:
        messages_filepath (str): Filepath for the messages CSV file.
        categories_filepath (str): Filepath for the categories CSV file.
        
    Returns:
        pd.DataFrame: Merged dataset.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on="id")
    return df

def clean_data(df):
    """
    Clean the merged dataset.
    
    Args:
        df (pd.DataFrame): Merged dataset.
        
    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    # Split categories into separate columns
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]  # First row for column names
    category_colnames = row.apply(lambda x: x.split('-')[0])
    categories.columns = category_colnames
    
    # Convert category values to binary (0 or 1)
    for column in categories:
        categories[column] = categories[column].str.split('-').str[1]
        categories[column] = categories[column].astype(int)
        categories[column] = categories[column].apply(lambda x: 1 if x > 1 else x)
    
    # Drop the original categories column and concatenate the new categories dataframe
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)
    
    # Remove duplicates
    df = df.drop_duplicates()
    return df

def save_data(df, database_filepath):
    """
    Save the clean dataset into an SQLite database.
    
    Args:
        df (pd.DataFrame): Cleaned dataset.
        database_filepath (str): Filepath for the SQLite database.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')

def main():
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        print(f"Loading data...\n    MESSAGES: {messages_filepath}\n    CATEGORIES: {categories_filepath}")
        df = load_data(messages_filepath, categories_filepath)
        
        print("Cleaning data...")
        df = clean_data(df)
        
        print(f"Saving data...\n    DATABASE: {database_filepath}")
        save_data(df, database_filepath)
        
        print("Cleaned data saved to database!")
    else:
        print("Please provide the filepaths of the messages and categories "
              "datasets as the first and second argument respectively, as well as "
              "the filepath of the database to save the cleaned data to as the third argument. \n\nExample: python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db")

if __name__ == '__main__':
    main()
