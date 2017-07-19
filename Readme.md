## Supervised Latent Dirichlet Allocation

*Aim* : 

To use topics derived from latent dirichlet allocation to understand if a review is talking about a Movie or a Book.

*Running Instructions*:



*Datasets* : 

We used the dataset from this [link](http://jmcauley.ucsd.edu/data/amazon/links.html), which contains product reviews and metadata from Amazon, including 142.8 million reviews spanning May 1996 - July 2014. We only needed the reviews for Books and Movies_TV datasets to be able to create a model that can learn to differentiate between documents talking about one or the other topic.

*read_json.py* : 

This script takes the top K samples from both Books and Movies json files and extracts the reviews along with their label as a dataframe. We use this script to create the training dataset of 20k reviews (10 k for each label) and the testing dataset of 10 k reviews ( 5k for each label)

*train.py* : 

The train.py file first develops features from the review text and then uses Latent Dirichlet Allocation to extract topics from the Bags of words matrix. The topics are then used as features to train the model. We test several models, including Logistic Regression, Random Forests, Gradient Boost and Adaboost.

*test.py* :

This file takes as input either text written in the terminal or a test file, and outputs the class of the texts. In case of text written in the terminal, it prints out the predicted class, while in case of a text file, it outputs the predictive accuracy of the model and a confusion matrix of the results.

*text preprocessing*

We have made use of tokenizing, punctuation and stop word removal followed by stemming of words to their roots.

*Models used* : 

We test several models, including Logistic Regression, Random Forests, Gradient Boost and Adaboost, out of which Logistic Regression and Gradient Boost give the best and almost similar results. We can also use an ensemble model if getting a couple of decimal points higher accuracy is worth the extra computational cost.

*Time taken by program*

*Results* :

which accuracy metrics did we use and why

*Latent Dirichlet Allocation*:

Latent Dirichlet Alllocation is a probabilistic topic model with a corresponding generative process. A topic is a distribution over a fixed vocabulary that the documents are expected to be generated out of. We chose to use this technique since it offers an efficient and low loss dimensionality reduction as compared to using bag of word counts or tfidf frequencie over the entire document vocabulary. 

*Scaling with larger corpus*

The LDA function in gensim offers the possibility to run the model online where the model is updated in iterations running on chunks of the dataset, which also allows us to account for topic drifts. An even larger speed up can be obtained by running Distributed LDA function over different clusters.

*Latent Dirichlet Allocation Topics* :

We chose to map the content to n = 10 topics, which were defined by the lda model as:

| Topic | Composition                              |
| ----- | ---------------------------------------- |
| 0     | 0.050*"book" + 0.026*"read" + 0.016*"charact" + 0.015*"stori" + 0.011*"one" + 0.010*"like" + 0.009*"time" + 0.008*"seri" + 0.007*"end" + 0.007*"would"' |
| 1     | 0.020*"film" + 0.019*"jesu" + 0.013*"movi" + 0.011*"christ" + 0.009*"gibson" + 0.009*"see" + 0.008*"passion" + 0.008*"peopl" + 0.007*"one" + 0.006*"god" |
| 2     | 0.048*"movi" + 0.023*"film" + 0.015*"one" + 0.013*"watch" + 0.012*"great" + 0.012*"good" + 0.011*"like" + 0.009*"time" + 0.009*"well" + 0.008*"see" |
| 3     | 0.014*"mysteri" + 0.012*"christi" + 0.011*"murder" + 0.009*"emma" + 0.006*"mr" + 0.006*"novel" + 0.006*"agatha" + 0.006*"one" + 0.005*"poirot" + 0.005*"8217" |
| 4     | 0.023*"war" + 0.009*"world" + 0.008*"histori" + 0.007*"german" + 0.006*"american" + 0.006*"one" + 0.006*"british" + 0.005*"flashman" + 0.005*"u" + 0.004*"boat" |
| 5     | 0.059*"quot" + 0.015*"novel" + 0.011*"clanci" + 0.009*"hemingway" + 0.008*"jane" + 0.008*"war" + 0.007*"robert" + 0.007*"one" + 0.007*"jack" + 0.006*"john" |
| 6     | 0.011*"life" + 0.010*"circu" + 0.008*"stori" + 0.008*"man" + 0.007*"jacob" + 0.007*"love" + 0.007*"old" + 0.006*"one" + 0.005*"charact" + 0.005*"year" |
| 7     | 0.017*"ann" + 0.014*"mari" + 0.013*"famili" + 0.012*"henri" + 0.012*"histor" + 0.012*"boleyn" + 0.010*"life" + 0.010*"histori" + 0.010*"king" + 0.009*"sister" |
| 8     | 0.028*"movi" + 0.017*"34" + 0.014*"one" + 0.014*"love" + 0.013*"watch" + 0.013*"great" + 0.011*"time" + 0.010*"good" + 0.009*"like" + 0.009*"dvd" |
| 9     | 0.022*"film" + 0.011*"movi" + 0.009*"version" + 0.008*"dvd" + 0.008*"one" + 0.006*"scene" + 0.006*"like" + 0.005*"releas" + 0.005*"get" + 0.005*"make" |

As we can see, some of these topics consist of specifically book oriented words, others only words that we use when talking about a movie.