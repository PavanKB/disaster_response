import sys
import string
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

import nltk
nltk.download(['punkt', 'stopwords', 'wordnet'])
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer


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


def tokenize(text, stem_or_lem='stem'):
    """
    Function to tokenise the text and perform stemming or lemmatisation
    https://stackoverflow.com/questions/10554052/what-are-the-major-differences-and-benefits-of-porter-and-lancaster-stemming-alg
    :param text: Text to analyse
    :param stem_or_lem: Option to stem or lemmatise
    :return: List of tokens
    >>> tokenize("This is a test for the tokeniser to see how effective it is.")
    """
    text = text.lower()

    tokens = nltk.word_tokenize(text)
    tokens_clean = [token for token in tokens if not is_stop_punc(token)]

    if stem_or_lem.lower() == 'stem':
        stemmer = SnowballStemmer('english')
        result = [stemmer.stem(token) for token in tokens_clean]
    elif stem_or_lem.lower() == 'lem':
        lemmatiser = WordNetLemmatizer()
        result = [lemmatiser.lemmatize(token) for token in tokens_clean]
    else:
        raise ValueError('stem or lem. Could not understand \'%s\' ?' % stem_or_lem)

    return result


def build_model():
    """
    Prepares a new instance of the Pipeline
    :return: Pipeline with transforms for text analysis
    """
    classifier = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tf_idf', TfidfTransformer()),
        ('multi_class', MultiOutputClassifier(estimator=RandomForestClassifier(n_estimators=20)))
    ])
    
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
    Takes test data and return the metrics for ech group
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


def main():
    if len(sys.argv) == 3:

        database_filepath, model_filepath = sys.argv[1:]

        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train.iloc[:, 0].values, Y_train.values)

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
