# Machine Learning Code Challenge, Level 2

## Description
Using the Yahoo! Question-Answer Dataset, build a document classification model and serve it.


## Requirements

- Use this [link](https://drive.google.com/open?id=1BHICkntwHlD_KaaG2_0n6obV9fi_TqBv) to get the data
- Update this `Readme` with instructions to train, evaluate and serve the model(s). 
- For serving, you may choose any technology you want, but there has to be an `http endpoint` that can be hit with data and returns predictions.
- Please include justification for any model architecture chosen in the `Readme`.
- Include any exploratory data analysis done before building the model (in a jupyter notebook or any other format) if applicable.
- You can use any library you believe is relevant to the completion of the task.
- We encourage the use of `python`, if you do not use `python`, please reach out to us.

## Review notes

When reviewing the code, we will pay attention to:

- Code quality, maintainability and readability. At Dialogue, you'll be working in a team, so it's important for us to see how easy it is for others to work with your code.
- Model architecture choice. This exercice is *not* about winning a Kaggle competition or beating SOTA. But you should be able to explain your choice of architecture.
- Model serving. At Dialogue we value vestatility and customer orientation. It is a nice touch to show that your model isn't just a theoretical exercice in a notebook, but something that other developers can use.
- Evaluation method and metrics used. It's not just about the code, it's also about the choice of metrics.
- Model performance. Again, it's OK to not reach SOTA, but it's interesting to learn why you think you're getting the performance you're getting, and what can be done to improve it if you had more time.
- Data exploration. What insights can you point to about the dataset you're working with.
- Code testability and best software engineering pratices. At Dialogue, your code will eventually be running in production. Show us that you understand and follow the best practices.

The whole exercice should not take more than a couple of hours - we don't need to write a production-ready high-performant solution. Remember that this code challenge is mainly to understand how you code and have a good topic of conversation about the choices you made.





# Section 1 of 3
""" ***** Model Deployment ***** 
The trained model is now live and accessible over the web. The model is deployed as a
REST API using Flask RESTful

To access the web application and query the trained model, please format your input data as the 
link below and submitt it through http:

curl -X GET http://hosseinmhz.pythonanywhere.com/ -d title="title here" -d content="content here" -d answer="and answer here"

As the response, you will receive a JSON object formatted as:
{"Prediction": "Class ID", "Category": "Class Label"}

"""

import pandas as pd<br />
import numpy as np<br/>
import matplotlib.pyplot as plt<br/>

from sklearn.feature_extraction.text import TfidfVectorizer<br/>
from sklearn.naive_bayes import MultinomialNB\n
from sklearn.svm import SVC\n
from sklearn.linear_model import SGDClassifier\n
from sklearn.neural_network import MLPClassifier\n
from sklearn.model_selection import GridSearchCV\n
from sklearn.ensemble import RandomForestClassifier\n
from sklearn.feature_selection import chi2, SelectKBest\n
from sklearn.pipeline import Pipeline\n
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix\n

# Section 2 of 3
""" ***** Data preparation, data analysis and prepresoosing and model training  *****"""

### Step 0: Data preparation 
columns_header = ["label", "title", "content", "answer"] # column names for the dataframe
trainset = pd.read_csv("./yahoo_answers_csv/train.csv", header = None)
trainset.columns = columns_header #setting the columns names
testset = pd.read_csv("./yahoo_answers_csv/test.csv", header = None)
testset.columns = columns_header #setting the columns names

### Removing the rows with missing values in the column Answer
trainset = trainset[trainset['answer'].notna()]
testset = testset[testset['answer'].notna()]

### Creating a new column (feature) by combining two filed: Title and Answer
""" I am curious to see if combination of the two fileds leads to a better performance of the model """
trainset['title_answer'] = trainset['title'] + trainset['answer']
testset['title_answer'] = testset['title'] + testset['answer']


### **** Data is balanced - train: 140,000 of each class, test: 6,000 of each class

## Step 1: Feature extraction and data wrangling
""" There are different approaches for feature extraction in NLP. But based on my experience 
###TFIDF should work just as fine as others (such as countvectorizer) """

tfidf_vectorizer = TfidfVectorizer(max_df = 0.5, min_df = 10, stop_words = "english", 
                            lowercase = True, ngram_range = (1,2), # considering both 1 grams and 2 grams
                            max_features = 5000, # it will keep top 5000 terms based on their term frequency
                            token_pattern= u'(?ui)\\b\\w*[a-z]+\\w*\\b') # ignore the terms that includes only number
train_tfidf_vec = tfidf_vectorizer.fit_transform(trainset['title_answer'])
train_features = tfidf_vectorizer.get_feature_names()

test_tfidf_vec = tfidf_vectorizer.transform(testset['title_answer'])
test_features = tfidf_vectorizer.get_feature_names()


## Step 2: Exploring 4 classifiers 
#To get insight of the trainability of the data and choose the best model

# SVM
classification(SGDClassifier(), train_tfidf_vec, trainset['label'], test_tfidf_vec, testset['label'])

# NaiveBayes
classification(MultinomialNB(alpha = 0.01, fit_prior = True), train_tfidf_vec, trainset['label'], test_tfidf_vec, testset['label'])

###test on single text
test_x = tfidf_vectorizer.transform(["Why do women get PMS"])
model = MultinomialNB()
model.fit(train_tfidf_vec, trainset['label'])
model.predict(test_x)

# Random Forest
classification(RandomForestClassifier(max_depth=10, random_state=0), train_tfidf_vec, trainset['label'], test_tfidf_vec, testset['label'])

# NN
model_NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,), random_state=1,
                         activation = 'relu', max_iter=50)
classification(model_NN, train_tfidf_vec, trainset['label'], test_tfidf_vec, testset['label'])

## Result of models
#4 classifier were trained over 3 different combinations of the data (using top 5000 tf values):
#(i) title only, (ii) answer only, (iii) title and answer 
#Overall, all the models have the best perfromance when both title and answer are considered,
#and lowest performance when answer only (from 5 to 10 percent improve when considered both fileds).
#Random Forest has the lowest performance with (acc: 27%, f1: 28%) in title only 
#growing up to 39% for oth accuracy and F1.
#The other 3 classifier have very similair performance around 60% in title only and 65% in title and answer together.
#Overall SVM and NN do slightly better than NB, but SVM takes secodns to train, but NN around 10 minutes
#So, I decided to continue the analysis with NB
#- I tried NB over whole data (not limited to top 5000), I get accuracy and f1 of 70% and 69% respectively.   
"""

## Step 3: Continue feature engineering - Dimensionality reduction - 
#generate the TFIDF matrix again, this time with all the phrases inlcuded
""" I am going to use chi2 score to find the most relevant terms that help boost the classifer """
tfidf_vectorizer = TfidfVectorizer(max_df = 0.5, min_df = 10, stop_words = "english", 
                            lowercase = True, ngram_range = (1,2),
                            token_pattern= u'(?ui)\\b\\w*[a-z]+\\w*\\b')
train_tfidf_vec = tfidf_vectorizer.fit_transform(trainset['title_answer'])
train_features = tfidf_vectorizer.get_feature_names()

test_tfidf_vec = tfidf_vectorizer.transform(testset['title_answer'])
test_features = tfidf_vectorizer.get_feature_names()

## Sensitivity analysis: 
#try k = 5000, 10000, 25000, 50000, and 160000 (based on chi2 cutoff threshold - statistical signficance)
ch2_selection = SelectKBest(chi2, k=25000)
train_chi2_selected = ch2_selection.fit_transform(train_tfidf_vec, trainset['label'])
test_chi2_selected = ch2_selection.transform(test_tfidf_vec)
classification(MultinomialNB(), train_chi2_selected, trainset['label'], test_chi2_selected, testset['label'])

## Result
#k = 25,000 is gives us the best results: 
#only 1% decrease in accuracy and 2% decrease in f1, comparing with the model that
#includes total dataset with 637006 terms (features)
#compared with other models, NB is still the best
"""

## Step 4: hyperparameters tuning

model = Pipeline([('clf', MultinomialNB())])
#model = MultinomialNB()
model.fit(train_chi2_selected, trainset['label'])
prediction = model.predict(test_chi2_selected)
np.mean(predicted ==  testset['label'])

parameters = {'clf__fit_prior': (True, False), 'clf__alpha': (2.0, 1.0, 0.1, 0.01, 0.001)}
model_hp_tuning = GridSearchCV(model, parameters, cv=5, n_jobs=-1)
model_hp_tuning = model_hp_tuning.fit(train_chi2_selected, trainset['label'])
model_hp_tuning.best_score_ # best score is 69%
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, model_hp_tuning.best_params_[param_name]))
# best parameters are: clf__alpha: 0.01, and clf__fit_prior: True

"""
## ******* Final configuration *********
#Data: 
#input = title + answer
#TFIDF = with 2 grams
#TFIDF_vector = top 25,000 terms based on chi2 score
#Classifier:
#MultinomialNB(alpha = 0.01, fit_prior = True)
"""

# Section 3 of 3: Future work
""" ***** If I had more time I would do the following items:  *****
- Include the results from dimensonality reduction from the chi2 analysis into the online version
- For dimensonality reduction, PCA could be investigated as well
- Better handling the numbers in the corpus at TFIDF vectorizing 
- Include stemming (such as Porter stemming) at the time of vectorizing 
- Parameter tuning for TFIDFvectorizing
- Working with NN would probably gives slightly better performance if it is trained well. 
  It takes time to train a NN model, buth it's worth investigating. 
"""



## extra methods
#To measure the performance of a classifier using Accuracy and F1 Score 
def performance_metrics(actual, prediction):
    accuracy = accuracy_score(actual, prediction)
    precision = precision_score(actual, prediction, average = 'macro') #Since the data is well-balances I use Macro to calculate the mean of performance metrics
    recall = recall_score(actual, prediction, average = 'macro')
    f1 = f1_score(actual, prediction, average = 'macro')
    # I conider accuracy and f1 to measure the performance.  
    return accuracy, f1 

#To run a classifier model with the passing arguemnts (data input for train and test) 
def classification(model, train_X, train_Y, test_X, test_Y):
    model.fit(train_X, train_Y)
    prediction = model.predict(test_X)
    #model.predict_proba(test_X)
    print(performance_metrics(test_Y, prediction))
    
    
##Calculations of chi2 to find the cutoff threshold
chi2score_ = chi2(train_tfidf_vec, trainset['label'])[0]
train_tfidf_chi2_vec = zip(train_features, chi2score_)
train_tfidf_chi2_vec = sorted(train_tfidf_chi2_vec, key=lambda x:x[1])
top_train_tfidf_chi2 = list(zip(*train_tfidf_chi2_vec[-159006:]))
#159006

labels = top_train_tfidf_chi2[0]
plt.figure(figsize=(15,10))
x = range(len(top_train_tfidf_chi2[1]))
plt.barh(x,top_train_tfidf_chi2[1], align='center', alpha=0.2)
plt.plot(top_train_tfidf_chi2[1], x, '-o', markersize=5, alpha=0.8)
plt.yticks(x, labels)
plt.xlabel('$\chi^2$')
