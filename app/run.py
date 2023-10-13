import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Pie
import plotly.graph_objs as gro
import plotly.figure_factory as ff
from sklearn.externals import joblib
import plotly.express as px
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('MessageCategories', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    show_buttons = True

    print('in Index function')
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

        # To display the category distribution

    category_counts = df.sum().drop(['id', 'message', 'genre'], inplace = False)
    category_names = category_counts.index.tolist()
    category_values = category_counts.astype(int).tolist()

    #Category corr heatmap

    category_frequencies = category_counts.sort_values(ascending=False)

    # Select the top 10 most frequent categories
    top_10_categories = category_frequencies.head(10).index

    # Create a subset of the DataFrame with the selected columns
    subset_df = df[top_10_categories]

    # Calculate the correlation matrix
    correlation_matrix = subset_df.corr()

    # Create a correlation heatmap using Plotly
    trace = gro.Heatmap(z=correlation_matrix.values,
                    x=top_10_categories,
                    y=top_10_categories,
                    colorscale='Viridis')

    layout = gro.Layout(title="Correlation Heatmap of Top 10 Categories")

    fig = gro.Figure(data=[trace], layout=layout)

    # Serialize the figure as a JSON string
    figure_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Pie(
                    labels=genre_names,
                    values=genre_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Genres'
            }
        },

        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_values
                )
            ],
            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [trace],  # Include the heatmap trace here
            'layout': {
                'title': "Correlation Heatmap of Top 10 Categories"
            }
        }

    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)


    # category_values = category_counts.tolist()
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON, show_buttons=show_buttons)

# web page that handles user query and displays model results
@app.route('/go')
def go():

    show_buttons = False
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results,
        show_buttons=show_buttons
    )



def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()