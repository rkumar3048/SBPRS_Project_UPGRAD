# **Capstone Project - Sentiment Based Product Recommendation System** 

# Suppress Warnings

import warnings
warnings.filterwarnings('ignore')

# Importing Libraries
import pandas as pd
import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.display.max_rows = 4000

import re
import string
import math

from sklearn import linear_model, metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV

from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

# import other required libraries
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

#Read the data for building the recommendation system
df_product = pd.read_csv("./data/sample30.csv")

### **EDA - Exploring the Data and Preparing it | Data Cleaning and Pre-Processing**

# Taking a look at the missing values - Percentage
(df_product.isnull().sum(axis=0)/len(df_product)*100).sort_values(ascending=False)

#Data overivew
print('rows: ', df_product.shape[0])
print('columns: ', df_product.shape[1])
print('\nfeatures: ', df_product.columns.to_list())
print('\nmissing vlues: ', df_product.isnull().values.sum())
print('\nUnique values: \n', df_product.nunique())

# Dropping the columns with high missing values and the columns not helpful in making recommndation system
df_product.drop(columns=['reviews_userProvince','reviews_userCity','reviews_didPurchase','reviews_doRecommend','manufacturer','reviews_date','brand','categories'],inplace=True)

# Listing the available Features now
df_product.columns

# Total null values count
df_product.isnull().sum().sum()

# Glacing the missing values after dropping the columns in previous step
for cols in df_product.columns:
	# count number of rows with missing values
	n_miss = df_product[cols].isnull().sum()
	perc = n_miss / df_product.shape[0] * 100
	print('col: %s, Missing: %d (%.1f%%)' % (cols, n_miss, perc))


# """**Handling Missing Values**"""

# imputing the missing 'review_username' as 'Unknown'
df_product['reviews_username'] = df_product['reviews_username'].fillna('Unknown')

#Verifying the column - 'review_username'
df_product['reviews_username'].isna().sum()

df_product['reviews_username'].value_counts(ascending=False)

# imputing the missing values in the column 'review_title' by comparing the existing title values with the review text of that NaN column
for ind_nan_title in df_product[df_product['reviews_title'].isna()]['reviews_text'].index:
  for ind_prod in df_product.index:
    s1=df_product[df_product['reviews_title'].isna()]['reviews_text'][ind_nan_title]
    s2=str(df_product['reviews_title'][ind_prod])
    if(s2 in s1):
      df_product.at[ind_nan_title, 'reviews_title'] = s2
      break

# Againg reviewing the missing count
df_product.isnull().sum()

df_product.info()

# Checking the row with 'review_title' is NaN
df_product[df_product['reviews_title'].isna()]

df_product[df_product['reviews_title'].str.contains("Enjoy", case=True)==True]['reviews_title'].value_counts()

#imputing Review Title with most relevant Title from the existing Titles used till now
df_product['reviews_title'].fillna('Enjoyed',inplace=True)

# Checking the NaN value in 'user_sentiment' column
df_product[df_product['user_sentiment'].isna()]

# Since Review Title/ Rating/ Review Text all are Pointing to - 'Positive Sentiment' : Assigning 'Positive' to this null value
df_product['user_sentiment'].fillna('Positive',inplace=True)

df_product.isnull().sum().sum()

df_product.isnull().sum()

# """**Completed Missing Value Handling**"""

df_product['reviews_username'] = df_product['reviews_username'].str.lower()

# Reviwing Data Again
print('rows: ', df_product.shape[0])
print('columns: ', df_product.shape[1])
print('\nfeatures: ', df_product.columns.to_list())
print('\nmissing vlues: ', df_product.isnull().values.sum())
print('\nUnique values: \n', df_product.nunique())

df_product['reviews_username'].value_counts(ascending=False)

# Checking User Sentiment Ratio
df_product['user_sentiment'].value_counts()/len(df_product)*100

# Checking Rating Distribution
df_product['reviews_rating'].value_counts()/len(df_product)*100

df_product.describe()

# Encoding User Sentiments : Positive : 1 & Negative : 0
df_product.user_sentiment.replace(('Positive', 'Negative'), (1, 0), inplace=True)


### **Text Processsing**

# create column user_review : merging columns - 'reviews_title' and 'reviews_text'

# Creating a derived feature - 'user_review'
df_product['user_review'] = df_product['reviews_title'] + ' ' + df_product['reviews_text']

# Dropping the base feature - 'reviews_title' & 'reviews_text'
df_product.drop(columns=['reviews_title','reviews_text'],inplace=True)

# Checking shape
df_product.shape

# List The features in use now
df_product.columns

# View informtion of the dataframe
df_product.info()

# Verify the 'user_review' columns
df_product['user_review'].value_counts()[:10]

df_product[df_product['user_review']=='Five Stars good']

df_product[df_product['user_review']=='Good Good']

# **convert text to lower case**

df_product['user_review'] = df_product['user_review'].str.lower()

# **Expand Contractions**

contractions_dict = {"ain't": "are not","'s":" is","aren't": "are not", "he’d" : "he would", "i've": "i have", "you can't": "you cannot", "they 're": "they are", "'re": "are", "'ve": "have", "n't": "not", "'re": "are", "i'm": "i am", "let's": "let us" }

# Regular expression for finding contractions
contractions_re=re.compile('(%s)' % '|'.join(contractions_dict.keys()))
def expand_contractions(text,contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, text)

# Expanding Contractions in the user reviews
df_product['user_review']=df_product['user_review'].apply(lambda x:expand_contractions(x))

# **Remove words and numbers containing digits**

df_product['user_review'] = df_product['user_review'].apply(lambda x: re.sub('W*dw*','',x))

# **Remove Punctuations**

#remove punctuation
df_product['user_review'] = df_product['user_review'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '' , x))

# **Remove Stop Words**


#remove stopwords
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in stop_words])

df_product['user_review'] = df_product['user_review'].apply(lambda x: remove_stopwords(x))


#Lemmatization : to reduce the words into their lemma by matching them with a language dictionary.
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

df_product['user_review'] = df_product['user_review'].apply(lambda text: lemmatize_words(text))

# **Remove Extra Space**

df_product['user_review'] = df_product['user_review'].apply(lambda text: re.sub(' +', ' ', text))

# **Reviewing Duplicate Records and Deleting It**

#Take count of duplicate record
df_product.duplicated().sum()

# Drop Duplicates
df_product.drop_duplicates(inplace=True)

# Reviwing Data Again
print('rows: ', df_product.shape[0])
print('columns: ', df_product.shape[1])
print('\nfeatures: ', df_product.columns.to_list())
print('\nmissing vlues: ', df_product.isnull().values.sum())
print('\nUnique values: \n', df_product.nunique())

### **Feature Extraction**

from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

review_text = df_product['user_review']
y = df_product['user_sentiment']

review_text.count()

# Using the n-gram tfidf vectorizer

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 3) )
    #,max_features=100000 ) 
word_vectorizer.fit(review_text)

train_word_features = word_vectorizer.transform(review_text)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(2, 6),
    max_features=50000)

char_vectorizer.fit(review_text)

train_char_features = char_vectorizer.transform(review_text)

train_features = hstack([train_char_features, train_word_features])

print(type(train_features))

import time 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
seed = 71

X_train, X_test, y_train, y_test = train_test_split(train_features, y, test_size=0.3, random_state=seed)
print('X_train', X_train.shape)
print('y_train', y_train.shape)
print('X_test', X_test.shape)
print('y_test', y_test.shape)

print(type(X_train))
print(type(X_test))

### **Model Building**

# **Handling Class Imballance**

from imblearn.over_sampling import SMOTE 
smt = SMOTE(random_state = 2) 
X_train_bl, y_train_bl = smt.fit_sample(X_train,y_train)

print("Balanced X Train :",X_train_bl.shape)
print("Balanced Y Train :",y_train_bl.shape)


# **Model: Logistic Regression for Sentiment Based Recommendation **

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score

# Using C = 77426 from the above Hyper Parameter Analysis
init_time = time.time()
logit = LogisticRegression(C=77426, multi_class = 'ovr')
logit.fit(X_train_bl, y_train_bl)
pred_logit = logit.predict(X_test)

time_taken = time.time() - init_time
print('Time Taken: {:.2f} seconds'.format(time_taken))

print(type(pred_logit))

print("Logistic Regression Accuracy", accuracy_score(pred_logit, y_test))
print(classification_report(pred_logit, y_test))
print(confusion_matrix(pred_logit, y_test))


### **Building the Recommendation System**
# Checking Our DataFrame
df_product.head()

df_grouped = df_product.groupby(['id','reviews_username'])

df_grouped['reviews_rating'].max().reset_index().head()

df_grouped['reviews_rating'].max().reset_index().isnull().sum()

df_reco_sys = df_grouped['reviews_rating'].max().reset_index()

df_reco_sys.head()

# """**User-Based Recommendation Systems**"""

df_reco_sys.shape

df_reco_sys.nunique()

# Test and Train split of the dataset.
from sklearn.model_selection import train_test_split
train, test = train_test_split(df_reco_sys, test_size=0.30, random_state=31)

print(train.shape)
print(test.shape)

# Pivot the train ratings' dataset into matrix format in which columns are item id(or product id) and the rows are username.
df_pivot = train.pivot(
    index=['reviews_username'],
    columns='id',
    values='reviews_rating'
).fillna(0)

df_pivot.head(3)

df_pivot.shape

# """**Create Dummy Train**"""

# Copy the train dataset into dummy_train
dummy_train = train.copy()

# The product not rated by user is marked as 1 for prediction. 
dummy_train['reviews_rating'] = dummy_train['reviews_rating'].apply(lambda x: 0 if x>=1 else 1)

# Convert the dummy train dataset into matrix format.
dummy_train = dummy_train.pivot(
    index='reviews_username',
    columns='id',
    values='reviews_rating'
).fillna(1)

dummy_train.head()

dummy_train.shape

# """**User Similarity Matrix - Using Adjusted Cosine Similarity**"""

# Create train pivot with NaN values for Adjusted Cosine Similarity
df_pivot = train.pivot(
    index='reviews_username',
    columns='id',
    values='reviews_rating'
)

df_pivot.head()

# """**Normalizing the Review Rating**"""

mean = np.nanmean(df_pivot, axis=1)
df_subtracted = (df_pivot.T-mean).T

df_subtracted.head()

# """**Finding the Cosine Similarity**"""

from sklearn.metrics.pairwise import pairwise_distances

# Creating the User Similarity Matrix using pairwise_distance function.
user_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
user_correlation[np.isnan(user_correlation)] = 0
print(user_correlation)

user_correlation.shape

# """**Prediction User-User**"""

user_correlation[user_correlation<0]=0
user_correlation

user_correlation.shape

df_pivot.shape

# Get user predicted rating by doing dot product between pivot and user_correlation
user_predicted_ratings = np.dot(user_correlation, df_pivot.fillna(0))
user_predicted_ratings

user_predicted_ratings.shape

# """Since we are interested only in the products not rated by the user, we will ignore the movies rated by the user by making it zero. """

# Calculate the user final Rating
user_final_rating = np.multiply(user_predicted_ratings,dummy_train)
user_final_rating.head()

# """**Evaluation User-User**"""

# Find out the common users of test and train dataset.
test.reviews_username.isin(train.reviews_username)
common = test[test.reviews_username.isin(train.reviews_username)]
common.shape

common.head()

# convert into the user-product matrix.
common_user_based_matrix = common.pivot_table(index='reviews_username', columns='id', values='reviews_rating')

common_user_based_matrix.head(5)

# Convert the user_correlation matrix into dataframe.
user_correlation_df = pd.DataFrame(user_correlation)

user_correlation_df['reviews_username'] = df_subtracted.index
user_correlation_df.set_index('reviews_username',inplace=True)
user_correlation_df.head()

user_correlation_df.shape

list_name = common.reviews_username.to_list()

user_correlation_df.columns = df_subtracted.index.tolist()

user_correlation_df_1 =  user_correlation_df[user_correlation_df.index.isin(list_name)]

user_correlation_df_1.shape

user_correlation_df_2 = user_correlation_df_1.T[user_correlation_df_1.T.index.isin(list_name)]

user_correlation_df_3 = user_correlation_df_2.T

user_correlation_df_3[user_correlation_df_3<0]=0

common_user_predicted_ratings = np.dot(user_correlation_df_3, common_user_based_matrix.fillna(0))
common_user_predicted_ratings

common_user_predicted_ratings.shape

common_user_based_matrix.shape

#Create Dummy Test

dummy_test = common.copy()

dummy_test['reviews_rating'] = dummy_test['reviews_rating'].apply(lambda x: 1 if x>=1 else 0)

dummy_test = dummy_test.pivot_table(index='reviews_username', columns='id', values='reviews_rating').fillna(0)

dummy_test.shape

common_user_predicted_ratings = np.multiply(common_user_predicted_ratings,dummy_test)

common_user_predicted_ratings.head(2)

#Normalize rating before calculatinng RMSE

# Normalization using MinMax Scaler
from sklearn.preprocessing import MinMaxScaler
from numpy import *

X  = common_user_predicted_ratings.copy() 
X = X[X>0]

scaler = MinMaxScaler(feature_range=(1, 5))
print(scaler.fit(X))
y = (scaler.transform(X))

print(y)

common_ = common.pivot_table(index='reviews_username', columns='id', values='reviews_rating')

common_.shape

common_.head(2)

# Finding total non-NaN value
total_non_nan = np.count_nonzero(~np.isnan(y))

# Calculating RMSE(Root Mean Square Error) for the User Based Correlation
rmse = (sum(sum((common_ - y )**2))/total_non_nan)**0.5
print(rmse)


# RMSE of 2.5 is a decent model for the prediction


### **Prediction Using the Recommendation Based System** 
#   - **Recommendation of Top 20 Products to a Specified User**


# Take the user ID as input.
user_input = str(input("Enter your user name "))
print(user_input)

user_final_rating.head(2)

d = user_final_rating.loc[user_input].sort_values(ascending=False)[0:20]
print("Top 20 Recommendations for the user '",user_input,"' are :- \n")
print(d)

from IPython.display import HTML
print('Top 20 Recommended Product Names : ')
print('==================================\n')

prod_reco = pd.merge(d,df_product,left_on='id',right_on='id', how = 'left')

HTML(pd.DataFrame(prod_reco['name'].unique()).rename(columns= {0:"Top_20_Recommendation"}).to_html(index=False))

# """### **Fine-Tuning the Recommendation System and Recommendation of Top 5 Products**"""

prod_reco['user_review'].count()

# Feature Extraction


rev_word_features = word_vectorizer.transform(prod_reco['user_review'])

rev_char_features = char_vectorizer.transform(prod_reco['user_review'])

from scipy.sparse import hstack

rec_prod_rev_features = hstack([rev_char_features, rev_word_features])

rec_prod_rev_features.shape

prod_reco['pred_sentiment'] = logit.predict(rec_prod_rev_features)

prod_reco.groupby(by='name')['pred_sentiment'].sum()/prod_reco.groupby(by='name')['pred_sentiment'].count()

final_reco_sys = pd.DataFrame(prod_reco.groupby(by='name')['pred_sentiment'].sum()/prod_reco.groupby(by='name')['pred_sentiment'].count())

final_reco = final_reco_sys.reset_index().sort_values(by='pred_sentiment',ascending=False).head()

print('Top 5 Recommended Product Names : ')
print('==================================\n')


HTML(final_reco.to_html(index=False))

# """### **Deployment : Creation of Model and Pickle File**"""

# Create Model Class

class model_recommendation:

  def predict(self, username):
    d=user_final_rating.loc[username].sort_values(ascending=False)[0:20]
    prod_reco = pd.merge(d,df_product,left_on='id',right_on='id', how = 'left')
    # Feature Extraction
    rev_word_features = word_vectorizer.transform(prod_reco['user_review'])
    rev_char_features = char_vectorizer.transform(prod_reco['user_review'])
    
    rec_prod_rev_features = hstack([rev_char_features, rev_word_features])
    
    #Final Tuning Based on sentiments to Best 5 Recommendations
    prod_reco['pred_sentiment'] = logit.predict(rec_prod_rev_features)
    final_reco_sys = pd.DataFrame(prod_reco.groupby(by='name')['pred_sentiment'].sum()/prod_reco.groupby(by='name')['pred_sentiment'].count())
    final_reco = final_reco_sys.reset_index().sort_values(by='pred_sentiment',ascending=False).head()
    print(final_reco)

reco_model_obj = model_recommendation()

print(user_input)

# Test Prediction
reco_model_obj.predict(user_input)



