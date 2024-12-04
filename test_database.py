import pandas as pd
from sqlalchemy import create_engine

# Path to your database file
database_filepath = r"C:\Users\CMRom\Downloads\DisasterResponse\data\DisasterResponse.db"

# Load the database
engine = create_engine(f'sqlite:///{database_filepath}')
df = pd.read_sql_table('Messages', engine)

# Display the first few rows
print(df.head())
print(f"\nThe database contains {df.shape[0]} rows and {df.shape[1]} columns.")
