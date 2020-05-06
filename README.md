# Purpose

This is the final project for the course Mathematical Techniques in Data Science (MATH 637) at the University of Delaware. The course instructor is Dr. Dominique Guillot. The team members associated with this project are Olivia Mwangi, Desiderio Pilla, and Akshaya Ramesh, who are all students in the Master of Science in Data Science (MSDS) program at UD.

# Background

On Twitter, users all over the world can express their opinion on any topic in a matter of seconds. By searching keywords or hashtags, anyone can scroll through a feed of content relating to any interest they like. The rise of social media has seen an uptick in polarizing content on the web, as it has become extraordinarily easy to share one's views. 

An important field of data science is sentiment analysis, which uses natural language processing to extract certain emotions of feelings from text. The main goal of this project is to build and train a model that can detect the sentiment of a tweet. For simplicity, this model limit its scope to label all tweets into one of these three categories:
 * negative
 * neutral
 * positive

Though there are many more specific sentiments that would need to be learned in order to obtain a clearer picture of a tweet's intentions, we will limit our model to learning these three classifications. However, a perfect model is nearly impossible to create. The reason is because sentiment is highly subjective. Anywhere from 10% - 30% of tweets would cause a debate amongst humans as to which category they fall into. To illustrate this, consider the following two tweets:

> I love the world!
>
> I hate the world.

These two tweets clearly fall into the positive and negative, respectively, sentiment categories. However, consider this tweet:

> I am indifferent about the world.

This tweet lies somewhere in between the previous two. Yet, rather than labeling it as "neutral", one could argue that this in fact elicits  a negative sentiment, as it is sad for someone to feel indifferent about the world. In any case, this classification is much less trivial to assign. Another non-trivial tweet is:

> The S&P 500 was down 300 points on Thursday.

This would come as bad news for investors, but happy news for short-sellers. But broadly, this tweet doesn't seem to have any emotion, but rather only delivers a fact. Lastly, consider one more tweet:

> I love candy, but it has too much sugar in it.

This person begins by saying something positive, but then ends it negatively. Overall, it is unclear whether this is a net-positive or net-negative tweet. It is also not as bland as the third example tweet; this sentence exhibits both positive and negative sentiments in one. The point of these examples is to show that sentiment analysis is not an exact science. Furthermore, not all sentiments are as easy to distinguish. A "neutral" label is much more subjective and poorly-defined than a "positive" or "negative" label.


# Problem Statement

In this project, we aim to create a machine learning model that is able to accurately classify the sentiment of a tweet. "Accuracy" for this model will be defined not only by its ability to correctly classify individual tweets, but also its ability to correctly classify large quantities of tweets to create an *aggregated sentiment score*. 

Though the accuracy of the model on individual tweets will be limited by the subjective nature of this application, we believe it is reasonable to achieve very accurate aggregate scores. Once a model has been created, trained, and validated, we intend to use it on real-time and historical tweets to collect aggregated sentiment scores for varying topics. These can be used to compare real-time responses and attitudes towards different subjects.

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

Furthermore, the `TfidfVectorizer` is a module which is equivalent to `Countvectorizer` followed by `TfidfTransformer`. This will convert raw text into a tf-idf matrix. The new Pipeline would look like

```
pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(lowercase=True, 
                                strip_accents='ascii',
                                stop_words='english')),
        ('classifier', MultinomialNB()),
        ])
```

---


## Cross Validation

The benefit of using the `Pipeline` approach is that it allows us to apply cross validation on all three of our estimators at once. Using `GridSearchCV`, we are able to create models that cycle through a given set of hyperparameters. 

The `parameters` dictionary contains the set of hyperparameters to loop through. In our case, the hyperparameters of our model are the arguments that are passed into each estimator. The parameters dictionary is created using the following format:
```
parameters = {estimatorName__argumentName : [list_of_hyperparameters]}

parameters = {'tfidf__ngram_range': [(1, 1), (1, 2)],
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
        ('tfidf', TfidfVectorizer(lowercase=True, 
                                strip_accents='ascii',
                                stop_words='english')),
        ('clf', CLFSwitcher()),           # Change this estimator to the CLFSwitcher           
        ])
```

In our parameters list, we create multiple dictionaries -- one for each classifier. For two classifiers, the parameters list would look like
```
 parameters = [
            {'clf__estimator': [MultinomialNB()],
             'tfidf__ngram_range': [(1, 1), (1, 2)],
             'tfidf__norm' : ['l1', 'l2'],
             'clf__estimator__fit_prior' : [True, False],
             'clf__estimator__alpha' : np.arange(0.1, 1.5, 16)
            },
            {'clf__estimator': [SGDClassifier(random_state=637, n_jobs=-1)],
             'tfidf__ngram_range': [(1, 1), (1, 2)],
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
