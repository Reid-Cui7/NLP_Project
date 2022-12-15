from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from icecream import ic
from sklearn.metrics import accuracy_score


TRAIN_CORPUS = 'dataset/train_after_analysis.csv'
STOP_WORDS = 'dataset/stopwords.txt'
WORDS_COLUMN = 'words_keep'

STOP_WORDS_SIZE = 300
WORDS_LONG_TAIL_BEGIN = 10000
WORDS_SIZE = WORDS_LONG_TAIL_BEGIN - STOP_WORDS_SIZE

content = pd.read_csv(TRAIN_CORPUS)
corpus = content[WORDS_COLUMN].values

stop_words = open(STOP_WORDS, encoding="utf-8").read().split()[:STOP_WORDS_SIZE]

tfidf = TfidfVectorizer(max_features=WORDS_SIZE, stop_words=stop_words)
text_vectors = tfidf.fit_transform(corpus)
print(text_vectors.shape)

targets = content['label']

x_train, x_test, y_train, y_test = train_test_split(text_vectors, targets, test_size=0.2, random_state=0)

rf = RandomForestClassifier()
rf.fit(x_train, y_train)
accuracy = accuracy_score(rf.predict(x_test), y_test)
ic(accuracy)
