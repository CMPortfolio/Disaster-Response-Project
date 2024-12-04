import sys
import pandas as pd
import pickle
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

def load_data(database_filepath):
    """
    Load data from SQLite database.
    
    Args:
        database_filepath (str): Filepath for the SQLite database.
        
    Returns:
        X (pd.Series): Messages (features).
        Y (pd.DataFrame): Categories (labels).
        category_names (list): List of category names.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message']
    Y = df.iloc[:, 4:]  # Assuming the first 4 columns are not categories
    category_names = Y.columns
    return X, Y, category_names

def build_model():
    """
    Build a machine learning pipeline with GridSearchCV.
    
    Returns:
        GridSearchCV: Grid search model object.
    """
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),  # Use default tokenization here
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__min_samples_split': [2, 3],
    }
    
    # Use single-threaded GridSearchCV to avoid parallelization issues
    model = GridSearchCV(pipeline, param_grid=parameters, cv=3, verbose=2, n_jobs=1)
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model performance and print classification report.
    
    Args:
        model: Trained model.
        X_test (pd.Series): Test features.
        Y_test (pd.DataFrame): Test labels.
        category_names (list): List of category names.
    """
    Y_pred = model.predict(X_test)
    for i, col in enumerate(category_names):
        print(f"Category: {col}")
        print(classification_report(Y_test.iloc[:, i], Y_pred[:, i]))

def save_model(model, model_filepath):
    """
    Save the model as a pickle file.
    
    Args:
        model: Trained model.
        model_filepath (str): Filepath for the pickle file.
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print(f"Loading data...\n    DATABASE: {database_filepath}")
        X, Y, category_names = load_data(database_filepath)
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        print("Building model...")
        model = build_model()
        
        print("Training model...")
        print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")  # Debug data shape
        model.fit(X_train, Y_train)
        
        print("Evaluating model...")
        evaluate_model(model, X_test, Y_test, category_names)
        
        print(f"Saving model...\n    MODEL: {model_filepath}")
        save_model(model, model_filepath)
        
        print("Trained model saved!")
    else:
        print("Please provide the filepath of the SQLite database as the first argument and "
              "the filepath of the pickle file to save the model as the second argument. \n\nExample: python train_classifier.py ../data/DisasterResponse.db classifier.pkl")

if __name__ == '__main__':
    main()
