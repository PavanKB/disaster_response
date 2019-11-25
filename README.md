# Disaster Response Pipeline Project

## Introduction
This code base contain the pipeline to process messages encountered during disaster
and classify them using `RandomForest` into one of the 36 categories that will define 
the response or help to be sent.

## Pipeline
The messages are processed using the following:
1. Tokenise (`Lemmatisation`)
1. `CountVectoriser`
1. `TF-IDF`
1. `MultiOutputClassifier` using `RandomForest`

The pipeline was further optimised by hyper parameter tuning using `GridSearch`
The best parameters were : **TO ADD**

## Files
The project is organised as follows:
1. **data**
Contains the csv file of messages and their categories classification as csv files.
It also has `process_data.py` which reads the csv file and prepares the data
for the model.  

1. **model**
Contains the `train_classifier.py` that has the logic to setup, train and evaluate 
the model using data from `data` folder

1. **app**
This is the flask app to interact with the model. (by Udacity)


## How to run this:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
