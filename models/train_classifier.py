import sys
import time
import string
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine


from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline

import nltk
nltk.download(['punkt', 'stopwords', 'wordnet'])
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def load_data(database_filepath):
    """
    Loads data from the SQLite database, reading from table `cleaned_data`
    :param database_filepath: The path of SQLite database to read
    :return: Message, categories classification of messages, list of categories
    """
    engine = create_engine('sqlite:///%s' % database_filepath)
    df = pd.read_sql_table(con=engine, table_name='cleaned_data')

    df.fillna(value=0, inplace=True)
    categories = df.columns.tolist()
    # remove message
    categories.remove('id')
    categories.remove('message')
    categories.remove('original')
    categories.remove('genre')

    X = df[['message']]
    y = df[categories]

    return X, y, categories


def is_stop_punc(token):
    """
    Returns True if the word is a punctuation or a stop word.
    :param token: String token to analyse
    :return: True if stop word or punctuation
    """
    if token in string.punctuation:
        return True
    if token in stopwords.words("english"):
        return True
    return False


def tokenize(text):
    """
    Function to tokenise the text and perform stemming
    https://towardsdatascience.com/stemming-lemmatization-what-ba782b7c0bd8
    https://stackoverflow.com/questions/10554052/what-are-the-major-differences-and-benefits-of-porter-and-lancaster-stemming-alg
    :param text: Text to analyse
    :return: List of tokens
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model(tokeniser, estimator):
    """
    Prepares a new instance of the Pipeline
    :param tokeniser: Tokeniser to use
    :param estimator: Estimator to use e.g RandomForest
    :return: Pipeline with transforms for text analysis
    """

    classifier = Pipeline([('vect', CountVectorizer(tokenizer=tokeniser)),
                           ('tf_idf', TfidfTransformer()),
                           ('multi_class', MultiOutputClassifier(estimator=estimator, n_jobs=-1)
                            )]
                          )

    return classifier


def display_results(y_test, y_pred):
    """
    Compares the predicted and the test score and returns the metrics
    :param y_test: Test data
    :param y_pred: Predicted data
    :return: A dictionary with label, Confusion matrix and Accuracy
    """
    labels = np.unique(y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)
    accuracy = (y_pred == y_test).mean()

    return {'Labels': labels,
            'Confusion Matrix': confusion_mat,
            'Accuracy': accuracy}


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Takes test data and prints the metrics for each group
    :param model: The model to evaluate
    :param X_test: Test input data
    :param Y_test: Test output data
    :param category_names: The expected category name
    :return: None
    """
    y_pred = model.predict(X_test.iloc[:, 0].values)

    # display results
    indv_results = {}
    for i, classification in enumerate(Y_test.columns):
        print(classification)
        indv_results[classification] = display_results(Y_test.values[:, i], y_pred[:, i])
        print(indv_results[classification])

    indv_report = {}
    for i, classification in enumerate(Y_test.columns):
        print(classification)
        indv_report[classification] = classification_report(Y_test.values[:, i], y_pred[:, i])
        print(indv_report[classification])


def save_model(model, model_filepath):
    """
    Saves the model as a file
    :param model: Model to be saved
    :param model_filepath: Path of the destination file
    :return: None
    """
    joblib.dump(model, model_filepath)


def grid_search(model, params, X_train, Y_train, n_jobs=None):
    """
    Performs a grid search and returns the optimised model.
    :param model: The model to optimise
    :param params: The dictionary or parameters
    :param X_train: Train dataset
    :param Y_train:
    :return: Optimised model
    """
    cv = GridSearchCV(model, params, n_jobs=n_jobs, cv=5, verbose=50, error_score=0, refit=True)
    cv.fit(X_train, Y_train)

    # Return the refitted model
    return cv.best_estimator_


def main():
    if len(sys.argv) == 3:

        database_filepath, model_filepath = sys.argv[1:]

        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        # --------------

        print('Building rnd forest model using lemmatisation...')
        model = build_model(tokenize,
                            RandomForestClassifier(n_estimators=10))

        print('Training model...')
        model.fit(X_train.iloc[:, 0].values, Y_train.values)

        print('Saving model...\n    MODEL: {}'.format('rnd_frst_lem.pkl'))
        save_model(model, 'rnd_frst_lem.pkl')

        print('Evaluating model...')
        t_start = time.perf_counter()
        evaluate_model(model, X_test, Y_test, category_names)
        t_stop = time.perf_counter()
        print("Elapsed time: %.1f [min]" % ((t_stop - t_start) / 60))

        # --------------

        print('Doing GridSearch...')
        t_start = time.perf_counter()
        parameters = {'multi_class__estimator__n_estimators': [20, 50],  # 30
                      'multi_class__estimator__criterion': ['gini', 'entropy'],  # Entropy
                      'multi_class__estimator__min_samples_split': [2, 4, 10],  # 2
                      'multi_class__estimator__min_samples_leaf': [2, 4, 10],  # 4
                      }
        model = grid_search(model, parameters, X_train.iloc[:, 0].values, Y_train.values)
        t_stop = time.perf_counter()
        print("Elapsed time: %.1f [min]" % ((t_stop - t_start) / 60))

        print('Evaluating Optimised model...')
        t_start = time.perf_counter()
        evaluate_model(model, X_test, Y_test, category_names)
        t_stop = time.perf_counter()
        print("Elapsed time: %.1f [min]" % ((t_stop - t_start) / 60))

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
