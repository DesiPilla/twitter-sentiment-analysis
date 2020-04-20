# Purpose

This is the final project for the course Mathematical Techniques in Data Science (MATH 637) at the University of Delaware. The course instructor is Dr. Dominique Guillot. The team members associated with this project are Olivia Mwangi, Desiderio Pilla, and Akshaya Ramesh, who are all students in the Master of Science in Data Science (MSDS) program at UD.

# Problem Statement

The goal of this project is to create a machine learning model that will be able to classify tweets from Twitter based on their sentiment. This scope of this problem has been reduced to classifying a tweet as either "negative", "neutral", or "positive". Once a model has been created, trained, and validated, we intend to use it on real-time and historical tweets to collect aggregated sentiment scores for varying topics and subjects to create time series charts.

# Overall Methodology

First, we will use `CountVectorizer` to create a dictionary where the keys are every word in the training set and the values are the frequencies of each word.
```
bow = CountVectorizer(lowercase=True, 
                      strip_accents='ascii',
                      stop_words='english')
bow.fit(X)
```
Next, we will convert the tweet text data (which is a 1D array) to an $n*w$ matrix, where *n* is the number of tweets in the training data and *w* is the number of words in the vocabulary dictionary. Each value in the matrix represents the number of times a given word appears in a given tweet. This will result in a very sparse dataset.
```
bow_matrix = bow.transform(X)
```
After this, we will use `TfidfTransformer` to calculate the term-frequency times inverse document-frequency (tf-idf) value for each word in the training set. 
 * term-frequency is the number of times the word appears in a tweet
 * inverse document-frequency is the number of texts that contain the word

Hence, a word's tf-idf score increases proportionally to the number of times it appears in the text, and is offset by the total number of tweets that contain the word. This is used to determine the importance of a word in a given tweet. The more times the word appears in the tweet, the more important it must be to the sentiment of the tweet. However, if many tweets contain this word, it must not be as important in differentiating between sentiments.

The `fit()` method will learn the idf vector, which is the total number of tweets that contain each word. The `transform()` method will compute the tf vector and calculate the tf-idf matrix.
```
tfidf_transformer = TfidfTransformer()
tfdif.fit(bow_matrix)
messages_tfidf = tfidf_transformer.transform(bow_matrix)
```
Finally, we will use `MultinomialNB` to train the tf-idf vectors with a Naive-Bayes classifier. This will be the final model that is used to classify tweet sentiment.
```
model = MultinomialNB().fit(messages_tdidf)
```
---


## Streamlined Methodology

Notice in the previous methodology that the three estimators (`CountVectorizer`, `TfidfTransformer`, and `MultinomialNB`) are used almost identically. Each estimator is instantiated, fit with the most recent estimator, and then transformed (except for the final estimator, which need only be trained). This process can be streamlined by using the `Pipeline` object. This object will conduct a series of fits and transforms. The code below will replicate the desired exuctions:
 * fit a `CountVectorizer` to the training set and transform it on the training data
 * fit a `TdifdTransformer` to the previous estimator and transform it on the previous estimator
 * fit a `MultinomialNB` to the previous estimator

```
pipeline = Pipeline([
        ('bow', CountVectorizer(lowercase=True, 
                                strip_accents='ascii',
                                stop_words='english')),
        ('tfidf', TfidfTransformer()),
        ('classifier', MultinomialNB()),
        ])
```
The first element of each tuple passed into the `Pipeline` is the name of the estimator. The second element of each tuple is the estimator itself, instantiated with any non-changing arguments.

---


## Cross Validation

The benefit of using the `Pipeline` approach is that it allows us to apply cross validation on all three of our estimators at once. Using `GridSearchCV`, we are able to create models that cycle through a given set of hyperparamters. 

Tthe `parameters` dictionary contains the set of hyperparameters to loop through. In our case, the hyperparameters of our model are the arguments that are passed into each estimator. The parameters dictionary is created using the following format:
```
parameters = {estimatorName__argumentName : [list_of_hyperparameters]}

parameters = {'bow__ngram_range': [(1, 1), (1, 2)],
              'tfidf__norm' : ['l1', 'l2'],
              'classifier__fit_prior' : [True, False],
              'classifier__alpha' : np.arange(0.1, 1.5, 16)
               }
```
We can then find and compute the best hyperparameters for our model by cycling through these argyments and using 10-fold cross validation. Using the optimal hyperparameters, we can then train out final model on the full training set.
```
grid = GridSearchCV(pipeline, cv=10, param_grid=parameters, verbose=1)
grid.fit(X, y)
```

---

## Testing Multiple Classifiers

One last improvement we can make is to add a `CLFSwitcher` function that will allow us to use different classifier methods in our `Pipeline` process. This object needs to extend the `fit()`, `predict()`, `predict_proba()`, and `score()` methods of the classifier to be used. To implement this change, we need to make slight adjustments to our `Pipeline` process and parameters list.
```
pipeline = Pipeline([
        ('bow', CountVectorizer(lowercase=True, 
                                strip_accents='ascii',
                                stop_words='english')),
        ('tfidf', TfidfTransformer()),
        ('clf', CLFSwitcher()),           # Change this estimator to the CLFSwitcher           
        ])
```

In our parameters list, we create multiple dictionaries -- one for each classifier. For two classifiers, the parameters list would look like
```
 parameters = [
            {'clf__estimator': [MultinomialNB()],
             'bow__ngram_range': [(1, 1), (1, 2)],
             'tfidf__norm' : ['l1', 'l2'],
             'clf__estimator__fit_prior' : [True, False],
             'clf__estimator__alpha' : np.arange(0.1, 1.5, 16)
            },
            {'clf__estimator': [SGDClassifier(random_state=637, n_jobs=-1)],
             'bow__ngram_range': [(1, 1), (1, 2)],
             'tfidf__norm' : ['l1', 'l2'],
             'clf__estimator__loss': ['hinge', 'log', 'modified_huber'],
             'clf__estimator__penalty': ['l1', 'l2', 'elasticnet']
            },
```
For this project, we will be cross validating with the following classifiers:
 * Support Vector Machines
 * Naive Bayes
 * Stochastic Gradient Descent
 * K-Nearest Neighbor
 * Random Forests
 * Perceptron
 * Logistic Regression

By cross validating among each of these classifiers (as well as cross validating among the hyperparameters of each estimator) we will be able to acheive an the best model possible for our training data.
</br>
