import json
import plotly
import pandas as pd
from flask import Flask, render_template, request
from sqlalchemy import create_engine
import pickle
import os

app = Flask(__name__)

# Set base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load data
database_path = os.path.join(BASE_DIR, '../data/DisasterResponse.db')
engine = create_engine(f'sqlite:///{database_path}')
df = pd.read_sql_table('DisasterResponse', engine)

# Load model
model_path = os.path.join(BASE_DIR, '../models/classifier.pkl')
model = pickle.load(open(model_path, "rb"))

@app.route('/')
@app.route('/index')
def index():
    """
    Renders the main page with visualizations.
    """
    # Data for genre distribution (Pie chart)
    genre_counts = df['genre'].value_counts()
    genre_names = list(genre_counts.index)

    # Data for category distribution (Horizontal bar chart)
    category_counts = df.iloc[:, 4:].sum().sort_values(ascending=False)
    category_names = list(category_counts.index)

    # Define graphs
    graphs = [
        # Genre Distribution Pie Chart
        {
            'data': [
                {
                    'type': 'pie',
                    'labels': genre_names,
                    'values': genre_counts,
                    'hoverinfo': 'label+percent',
                    'textinfo': 'value'
                }
            ],
            'layout': {
                'title': 'Distribution of Message Genres',
                'height': 400
            }
        },
        # Category Distribution Horizontal Bar Chart
        {
            'data': [
                {
                    'type': 'bar',
                    'x': category_counts.values,
                    'y': category_counts.index,
                    'orientation': 'h',
                    'marker': {
                        'color': 'rgba(76, 175, 80, 0.6)',
                        'line': {'color': 'rgba(76, 175, 80, 1.0)', 'width': 1}
                    }
                }
            ],
            'layout': {
                'title': 'Distribution of Message Categories',
                'xaxis': {'title': 'Count', 'automargin': True},
                'yaxis': {'title': 'Category', 'automargin': True},
                'height': 700,
                'margin': {'l': 150, 'r': 20, 't': 50, 'b': 50}
            }
        }
    ]

    # Encode Plotly graphs in JSON
    graph_json = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # Render the homepage
    return render_template('master.html', graphJSON=graph_json)


@app.route('/go')
def go():
    """
    Renders the classification result for a user-input message.
    """
    # Get the user query from the form
    query = request.args.get('query', '')

    # Predict classifications
    classification_results = model.predict([query])[0]

    # Convert results to dictionary
    categories = df.columns[4:]  # Assuming first 4 columns are not categories
    classification_dict = dict(zip(categories, classification_results))

    return render_template(
        'go.html',
        query=query,
        classification_result=classification_dict
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001, debug=True)
