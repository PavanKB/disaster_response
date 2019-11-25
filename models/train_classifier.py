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
    """
    if token in string.punctuation:
        return True
    if token in stopwords.words("english"):
        return True
    return False


def tokenize(text, stem_or_lem='stem'):
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
    classifier = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tf_idf', TfidfTransformer()),
        ('multi_class', MultiOutputClassifier(estimator=RandomForestClassifier(n_estimators=10)))
    ])
    
    return classifier


def display_results(y_test, y_pred):
    labels = np.unique(y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)
    accuracy = (y_pred == y_test).mean()

    return {'Labels': labels,
            'Confusion Matrix': confusion_mat,
            'Accuracy': accuracy}


def evaluate_model(model, X_test, Y_test, category_names):
    
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