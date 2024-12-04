
# Disaster Response Pipeline

## Project Summary
This project is a machine learning-based web application that helps classify disaster-related messages into multiple categories. These classifications enable efficient routing of messages to appropriate disaster response teams, such as medical aid, water supply, and shelter. The app processes real disaster data provided by Appen (formerly Figure 8) and demonstrates end-to-end data engineering, machine learning, and deployment.

The project includes:
- ETL Pipeline: For extracting, transforming, and loading data into a SQLite database.
- Machine Learning Pipeline: For building and training a multi-output classification model.
- Web Application: For visualizing data and predicting message classifications.

---

## Directory Structure
DisasterResponse/
- app/
  - templates/
    - master.html (Main page of the web app)
    - go.html (Classification result page)
  - run.py (Flask app to serve predictions and visualizations)
- data/
  - disaster_categories.csv (Categories dataset)
  - disaster_messages.csv (Messages dataset)
  - process_data.py (ETL script for data cleaning and storage)
  - DisasterResponse.db (SQLite database of cleaned data)
- models/
  - train_classifier.py (Machine learning script for training and saving the model)
  - classifier.pkl (Trained model saved as a pickle file)
- README.md (Project documentation)
- requirements.txt (Python dependencies)

---

## How to Run the Project

### 1. Install Dependencies
Ensure you have Python installed. Use pip to install the required dependencies:
```
pip install -r requirements.txt
```

### 2. Run the ETL Pipeline
Execute the process_data.py script to clean the data and save it into a SQLite database:
```
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```

### 3. Train the Machine Learning Model
Run the train_classifier.py script to train the model and save it as a pickle file:
```
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```

### 4. Launch the Web App
Start the Flask web application by running:
```
python app/run.py
```
The app will be available at http://localhost:3001.

---

## Features

### ETL Pipeline
- Merges datasets (disaster_messages.csv and disaster_categories.csv).
- Cleans the data:
  - Splits categories into separate binary columns.
  - Converts values to binary (0 or 1).
  - Removes duplicates.
- Saves the cleaned data to a SQLite database.

### Machine Learning Pipeline
- Processes text data using Natural Language Processing (NLP):
  - Tokenizes text.
  - Converts to lowercase and removes punctuation.
  - Lemmatizes words.
- Vectorizes text using TF-IDF and applies multi-output classification.
- Optimizes parameters with GridSearchCV.
- Outputs precision, recall, and F1-score for each classification category.

### Web Application
- Message Classification: Users can input a disaster-related message to see its classification across 36 categories.
- Data Visualizations:
  - Distribution of message genres (Pie Chart).
  - Distribution of messages across categories (Horizontal Bar Chart).

---

## Example Screenshots
**Home Page**
![image](https://github.com/user-attachments/assets/fddcd079-2bad-492e-a41f-639a27943642)
![image](https://github.com/user-attachments/assets/532ced22-1489-4c81-bac2-7594ef8e650c)


**Message Classification**
![image](https://github.com/user-attachments/assets/4b451ada-cec1-4efc-b3ae-83d7d8c294b1)
![image](https://github.com/user-attachments/assets/34dc7c56-f12d-406e-b546-7a315d98ddc1)

---

## Improvements
- Added data visualizations for better understanding of the dataset.
- Improved model training using GridSearchCV.
- Handled dataset imbalance by balancing precision and recall during evaluation.

---

## Model and Dataset Discussion
- The dataset is imbalanced, with some categories (e.g., "water") having very few examples. To address this:
  - Evaluated both precision and recall for multi-output classification.
  - Emphasized recall for critical categories to reduce false negatives in disaster responses.
- Future improvements:
  - Oversampling or undersampling techniques to handle imbalanced categories.
  - Further feature engineering to capture additional message contexts.

---

## Key Challenges
- Cleaning and processing the categories data required custom transformations.
- The model had to handle multi-output classification across 36 categories efficiently.
- Deploying the web app required integrating both the machine learning model and interactive visualizations.

---

## Dependencies
- Python 3.7+
- Flask
- Plotly
- Pandas
- SQLAlchemy
- NLTK
- Scikit-learn

---
