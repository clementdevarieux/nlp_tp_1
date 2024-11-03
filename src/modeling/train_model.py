import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import joblib
from nltk.corpus import stopwords
from nltk.stem.snowball import FrenchStemmer
import nltk
import random
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier


nltk.download('stopwords')

def stemmed_words(doc):
    stemmer = FrenchStemmer()
    analyzer = CountVectorizer().build_analyzer()
    return (stemmer.stem(w) for w in analyzer(doc))

def train(input_filename, model_dump_filename):
    data = pd.read_csv(input_filename)
    X = data['video_name']
    y = data['is_comic']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(analyzer='char', ngram_range=(1, 6))),
        ('classifier', RandomForestClassifier())
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

    joblib.dump(pipeline, model_dump_filename)