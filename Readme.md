## Supervised Latent Dirichlet Allocation 

### Aim 

To use topics derived from latent dirichlet allocation to understand if a review is talking about a Movie or a Book. 

### Running Instructions

run *test.py* in the terminal. 

You will be prompted to enter 0/1 based on whether your input is text or a file. If you choose

- `0`: you are then prmpted to enter text into the terminal 


-  `1`: you are promted to enter the address of the test csv file, which has a column of reviews and another of labels. 

If you'd like to run the model on a new test dataset, you can create a new random test_set:

 run `read_json.py path_to_json_files number_of_lines output_filename` in the terminal.

The program will randomly sample *number_of_lines*/2 from each of Movies and Books file, to give you a newly populated text dataset.

### Dataset

 We used the dataset from this [link](http://jmcauley.ucsd.edu/data/amazon/links.html), which contains product reviews and metadata from Amazon, including 142.8 million reviews spanning May 1996 - July 2014. We only needed the reviews for Books and Movies_TV datasets to be able to create a model that can learn to differentiate between documents talking about one or the other topic. 

### Scripts

*read_json.py* : 

This script takes the top K samples from both Books and Movies json files and extracts the reviews along with their label as a dataframe. We use this script to create the training dataset of 20k reviews (10 k for each label) and the testing dataset of 10 k reviews ( 5k for each label) 

*train.py* : 

The train.py file first develops features from the review text and then uses Latent Dirichlet Allocation to extract topics from the Bags of words matrix. The topics are then used as features to train the model. We test several models, including Logistic Regression, Random Forests, Gradient Boost and Adaboost. 

*test.py* : 

This file takes as input either text written in the terminal or a test file, and outputs the class of the texts. In case of text written in the terminal, it prints out the predicted class, while in case of a text file, it outputs the predictive accuracy of the model and a confusion matrix of the results.

### Methodology

We have made use of tokenizing, punctuation and stop word removal followed by stemming of words to their roots, followed by converting the document to a bag of words and then applying latent dirichlet model from the [gensim](https://radimrehurek.com/gensim/index.html) package that specializes in topic modelling.

*Models used* : 

We test several models, including Logistic Regression, Random Forests, Gradient Boost and Adaboost, out of which Logistic Regression and Gradient Boost give the best and almost similar results. We can also use an ensemble model if getting a couple of decimal points higher accuracy is worth the extra computational cost.

### Results 

We use roc_auc as a metric so as to be able to get the performance of the classifier irrespective of the threshold of cutoff between the two classes, but rather on its ability to rank patterns belonging to either class. A reliable and valid AUC estimate can be interpreted as the probability that the classifier will assign a higher score to a randomly chosen positive example than to a randomly chosen negative example from the sample. The final performance of the model on the test set gives us: 

```python
roc_auc_score
97.2 %

classification accuracy
92 % 
```

### Latent Dirichlet Allocation Topics

We chose to map the content to n = 15 topics, which were defined by the lda model as:

```python
(0, u'0.036*"circu" + 0.023*"life" + 0.022*"jacob" + 0.020*"stori" + 0.016*"love" + 0.014*"old" + 0.013*"eleph" + 0.013*"anim" + 0.011*"man" + 0.010*"water"')
(1, u'0.021*"seri" + 0.020*"godzilla" + 0.013*"episod" + 0.012*"u" + 0.010*"german" + 0.010*"war" + 0.010*"monster" + 0.009*"boat" + 0.007*"mothra" + 0.007*"show"')
(2, u'0.032*"book" + 0.019*"charact" + 0.019*"seri" + 0.014*"martin" + 0.009*"stori" + 0.009*"get" + 0.008*"one" + 0.008*"end" + 0.007*"like" + 0.007*"next"')
(3, u'0.066*"book" + 0.042*"read" + 0.019*"stori" + 0.015*"charact" + 0.015*"one" + 0.014*"like" + 0.012*"time" + 0.011*"good" + 0.011*"enjoy" + 0.011*"love"')
(4, u'0.029*"34" + 0.014*"novel" + 0.012*"mar" + 0.012*"clanci" + 0.010*"quot" + 0.009*"ship" + 0.008*"jack" + 0.007*"polit" + 0.007*"charact" + 0.006*"ryan"')
(5, u'0.029*"war" + 0.010*"world" + 0.008*"american" + 0.005*"one" + 0.005*"histori" + 0.005*"forc" + 0.005*"militari" + 0.004*"battl" + 0.004*"fight" + 0.004*"action"')
(6, u'0.010*"song" + 0.008*"one" + 0.008*"girl" + 0.008*"music" + 0.008*"get" + 0.008*"like" + 0.006*"love" + 0.006*"time" + 0.006*"best" + 0.006*"video"')
(7, u'0.034*"dvd" + 0.026*"version" + 0.016*"film" + 0.013*"releas" + 0.012*"edit" + 0.011*"ray" + 0.010*"1" + 0.010*"blu" + 0.010*"origin" + 0.009*"qualiti"')
(8, u'0.015*"basebal" + 0.012*"play" + 0.010*"boy" + 0.010*"film" + 0.008*"river" + 0.008*"team" + 0.007*"stand" + 0.006*"king" + 0.006*"snake" + 0.006*"game"')
(9, u'0.060*"movi" + 0.024*"film" + 0.023*"quot" + 0.017*"one" + 0.014*"watch" + 0.014*"great" + 0.013*"good" + 0.013*"like" + 0.010*"time" + 0.009*"love"')
(10, u'0.009*"flashman" + 0.007*"encount" + 0.006*"alien" + 0.006*"fraser" + 0.005*"maclean" + 0.005*"vincent" + 0.005*"close" + 0.005*"genet" + 0.005*"human" + 0.004*"one"')
(11, u'0.016*"christma" + 0.011*"leon" + 0.009*"year" + 0.009*"famili" + 0.008*"killer" + 0.008*"girl" + 0.007*"old" + 0.007*"rudolph" + 0.006*"santa" + 0.006*"young"')
(12, u'0.023*"ann" + 0.021*"mari" + 0.018*"henri" + 0.016*"boleyn" + 0.015*"hemingway" + 0.015*"histor" + 0.014*"king" + 0.014*"famili" + 0.012*"sister" + 0.012*"novel"')
(13, u'0.009*"one" + 0.008*"charact" + 0.005*"make" + 0.005*"work" + 0.005*"even" + 0.005*"life" + 0.005*"much" + 0.004*"seem" + 0.004*"well" + 0.004*"way"')
(14, u'0.031*"film" + 0.018*"movi" + 0.018*"jesu" + 0.010*"christ" + 0.009*"see" + 0.008*"gibson" + 0.007*"passion" + 0.007*"one" + 0.007*"peopl" + 0.006*"would"')
```

As we can see, some of these topics like topics [0, 2, 3, 4..] consist of specifically book oriented words like *stori, book, read, charact, novel, etc* and others made of words that we use when talking about a movie like *film, movi, see, dvd..etc*.

### Latent Dirichlet Allocation 

Latent Dirichlet Alllocation is a probabilistic topic model with a corresponding generative process. A topic is a distribution over a fixed vocabulary that the documents are expected to be generated out of. We chose to use this technique since it offers an efficient and low loss dimensionality reduction as compared to using bag of word counts or tfidf frequencie over the entire document vocabulary. 

### Scaling with larger corpus 

Given that the slowest and hence the rate determining step of the script is the LDA transformation, it is here that we can make the biggest different in speed. The LDA function in gensim offers the possibility to run the model online where the model is updated in iterations running on chunks of the dataset, which also allows us to account for topic drifts. An even larger speed up can be obtained by running Distributed LDA function over different clusters.