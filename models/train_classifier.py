import sys
import nltk
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer, f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
import sqlite3
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import pickle

def load_data(database_filepath):
    """
    Load data from a SQLite database and return features, labels, and target names.

    Args:
        database_filepath (str): Path to the SQLite database file.

    Returns:
        tuple: A tuple of features (X), labels (Y), and target names.
    """
    db_path = 'sqlite:///' + database_filepath
    engine = create_engine(db_path)
    df = pd.read_sql_table('MessageCategories', engine)

    X = df.message.values
    Y = df[['related', 'request', 'offer',
    'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
       'security', 'military', 'child_alone', 'water', 'food', 'shelter',
       'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report']].values # List of target labels
    target_names = ['related', 'request', 'offer','aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
       'security', 'military', 'child_alone', 'water', 'food', 'shelter',
       'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report']  # List of target names

    return X, Y, target_names

def tokenize(text):
    """
    Tokenize and clean text data.

    Args:
        text (str): Input text.

    Returns:
        list: List of cleaned tokens.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    """
    Build a machine learning model and perform hyperparameter tuning.

    Returns:
        GridSearchCV: A GridSearchCV object containing the machine learning pipeline.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    #To help with run times, the below code for Gridsearch params have been commented and 
    # the best estimators post gridsearch have been listed in parameters

    # parameters = {
    #     'clf__estimator__n_estimators': [50, 100],
    #     'clf__estimator__max_depth': [None, 10]
    # }

    parameters = {
        'clf__estimator__n_estimators': [100], 
        'clf__estimator__max_depth': [None]
    }


    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the trained model and print classification reports for each category.

    Args:
        model: Trained machine learning model.
        X_test: Testing features.
        Y_test: True labels.
        category_names: List of category names.
    """
    y_pred = model.predict(X_test)

    for i, column_name in enumerate(category_names):
        true_labels = Y_test[:, i]
        predicted_labels = y_pred[:, i]
        # Calculate and print a classification report for the current category
        report = classification_report(true_labels, predicted_labels)
        print(f"Category: {column_name}")
        print(report)

def save_model(model, model_filepath):
    """
    Save the trained model to a pickle file.

    Args:
        model: Trained machine learning model.
        model_filepath (str): Path to the pickle file for saving the model.
    """
    # Serialize and save the model to a pickle file
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)

        print("\nBest Parameters:", model.best_params_)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()
