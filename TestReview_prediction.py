import re, nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('wordnet') 
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle
!pip install UMAP #Installs UMAP library
import umap
import plotly.graph_objs as go
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import joblib

# Reading dataset as dataframe
df = pd.read_csv("/content/SpotifyReviews.csv", encoding = "ISO-8859-1") # You can also use "utf-8"
pd.set_option('display.max_colwidth', None) # Setting this so we can see the full content of cells
pd.set_option('display.max_columns', None) # to make sure we can see all the columns in output window
# Cleaning Tweets
def cleaner(review):
    soup = BeautifulSoup(review, 'lxml') # removing HTML entities such as ‘&amp’,’&quot’,'&gt'; lxml is the html parser and shoulp be installed using 'pip install lxml'
    souped = soup.get_text()
    re1 = re.sub(r"(@|http://|https://|www|\\x)\S*", " ", souped) # substituting @mentions, urls, etc with whitespace
    re2 = re.sub("[^A-Za-z]+"," ", re1) # substituting any non-alphabetic character that repeats one or more times with whitespace

    """
    For more info on regular expressions visit -
    https://docs.python.org/3/howto/regex.html
    """

    tokens = nltk.word_tokenize(re2)
    lower_case = [t.lower() for t in tokens]

    stop_words = set(stopwords.words('english'))
    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))

    wordnet_lemmatizer = WordNetLemmatizer()
    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in filtered_result]
    return lemmas

df['cleaned_review'] = df.Review.apply(cleaner)
df = df[df['cleaned_review'].map(len) > 0] # removing rows with cleaned reviews of length 0
print("Printing top 5 rows of dataframe showing original and cleaned reviews....")
print(df[['Review','cleaned_review']].head())
df.drop(['Review'], axis=1, inplace=True)
# Saving cleaned tweets to csv
df.to_csv('cleaned_data.csv', index=False)
df['cleaned_review'] = [" ".join(row) for row in df['cleaned_review'].values] # joining tokens to create strings. TfidfVectorizer does not accept tokens as input
data = df['cleaned_review']
Y= df['Recommend'].map({'Yes':1,'No':0})
#Y = df.Recommend # target column
print('Out of 100%, nearly {}% belongs to positive class'.format(round(sum(Y/len(Y)*100))))

tfidf = TfidfVectorizer(min_df=.00084998, ngram_range=(1,3)) # min_df=.00084998 means that each ngram (unigram, bigram, & trigram) must be present in at least 30 documents for it to be considered as a token (200000*.00015=30). This is a clever way of feature engineering
tfidf.fit(data) # learn vocabulary of entire data
data_tfidf = tfidf.transform(data) # creating tfidf values
pd.DataFrame(pd.Series(tfidf.get_feature_names_out())).to_csv('vocabulary.csv', header=False, index=False)
print("Shape of tfidf matrix: ", data_tfidf.shape)


# Implementing Support Vector Classifier
svc_clf = LinearSVC() # kernel = 'linear' and C = 1

# Running cross-validation
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1) # 10-fold cross-validation
scores=[]
iteration = 0
for train_index, test_index in kf.split(data_tfidf, Y):
    iteration += 1
    print("Iteration ", iteration)
    X_train, Y_train = data_tfidf[train_index], Y.iloc[train_index]
    X_test, Y_test = data_tfidf[test_index], Y.iloc[test_index]

    svc_clf.fit(X_train, Y_train) # Fitting SVC
    Y_pred = svc_clf.predict(X_test)
    score = metrics.accuracy_score(Y_test, Y_pred) # Calculating accuracy
    print("Cross-validation accuracy: ", score)
    scores.append(score) # appending cross-validation accuracy for each iteration
svc_mean_accuracy = np.mean(scores)
print("Mean cross-validation accuracy: ", svc_mean_accuracy, " \n")

# Implementing Naive Bayes Classifier
nbc_clf = MultinomialNB()

# Running cross-validation
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1) # 10-fold cross-validation
scores=[]
iteration = 0
for train_index, test_index in kf.split(data_tfidf, Y):
    iteration += 1
    print("Iteration ", iteration)
    X_train, Y_train = data_tfidf[train_index], Y.iloc[train_index]
    X_test, Y_test = data_tfidf[test_index], Y.iloc[test_index]
    nbc_clf.fit(X_train, Y_train) # Fitting NBC
    Y_pred = nbc_clf.predict(X_test)
    score = metrics.accuracy_score(Y_test, Y_pred) # Calculating accuracy
    print("Cross-validation accuracy: ", score)
    scores.append(score) # appending cross-validation accuracy for each iteration
nbc_mean_accuracy = np.mean(scores)
print("Mean cross-validation accuracy: ", nbc_mean_accuracy)

if svc_mean_accuracy > nbc_mean_accuracy:
  clf = LinearSVC().fit(data_tfidf, Y)
  joblib.dump(clf, 'svc.sav')
else:
  clf = MultinomialNB().fit(data_tfidf, Y)
  joblib.dump(clf, 'nbc.sav')

####################################################
## Retraining the selected model for production
X = data_tfidf
svc = LinearSVC() 
svc.fit(X, Y)
### Saving and loading the model
with open('model.pkl', 'wb') as f:
    pickle.dump((svc, tfidf), f)
with open('model.pkl', 'rb') as f:
    model, vect = pickle.load(f)
### Testing the model
test_data = pd.read_csv("/content/TestReviews.csv")
print(test_data)
test_data['cleaned_review'] = test_data['Review'].apply(cleaner)
test_data= test_data[test_data['cleaned_review'].map(len) > 0] # removing rows with cleaned reviews of length 0
print("Printing top 5 rows of dataframe showing original and cleaned reviews....")
print(test_data[['Review','cleaned_review']].head())
test_data.drop(['Review'], axis=1, inplace=True)
# Saving cleaned tweets to csv
test_data.to_csv('cleaned_data2.csv', index=False)
test_data['cleaned_review'] = [" ".join(row) for row in test_data['cleaned_review'].values] # joining tokens to create strings. TfidfVectorizer does not accept tokens as input
x = vect.transform(test_data['cleaned_review'])
pred = model.predict(x)
test_data['predictions'] = pred
test_data.head()
test_data.to_csv("/content/TestReviews.csv", index = False)

## n_neighbor = 150 min_dis = 0.4
u = umap.UMAP(n_components=2, n_neighbors=150, min_dist=0.4)
x_umap = u.fit_transform(X)
data_ = [go.Scatter(x=x_umap[:,0], y=x_umap[:,1], mode='markers',
                    marker = dict(color=test_data['predictions'], colorscale='Rainbow', opacity=0.5), #df['Recommend']
                                text=[f'cleaned_review: {a}<br>Class: {b}' for a,b in list(zip(test_data['cleaned_review'],test_data['predictions']))],hoverinfo='text')
        ]

layout = go.Layout(title = 'UMAP Dimensionality Reduction', width = 1000, height = 1000,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data_, layout=layout)
fig.show()


## n_neighbor = 200 min_dis = 0.6
u = umap.UMAP(n_components=2, n_neighbors=200, min_dist=0.6)
x_umap = u.fit_transform(X)
data_ = [go.Scatter(x=x_umap[:,0], y=x_umap[:,1], mode='markers',
                    marker = dict(color=test_data['predictions'], colorscale='Rainbow', opacity=0.5),
                                text=[f'cleaned_review: {a}<br>Class: {b}' for a,b in list(zip(test_data['cleaned_review'],test_data['predictions']))],hoverinfo='text')
        ]

layout = go.Layout(title = 'UMAP Dimensionality Reduction', width = 1000, height = 1000,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data_, layout=layout)
fig.show()
