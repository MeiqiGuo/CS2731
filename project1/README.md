# meg168-project1
Project 1 repository created for meg168

## README

### Environment and modules

Python 3.6.5

import nltk

nltk.download('punkt')

nltk.download('stopwords')

import sklearn

### Run experiments

Run below code for step 2:
>python main.py data.csv 2

Run below code for step 4:
>python main.py data.csv 4

Run below code for step 5: (You may need to wait for 15 minutes until it finishes.)
>python main.py data.csv 5

You may also directly load the result *acc_step5.pkl* if you don't want to run it again.

Notice that *data.cvs* is the file name of the given dataset in cvs format.

### References

[SciKit](https://scikit-learn.org/stable/index.html) Documentation


## REPORT

### Basic Statistic of Dataset

Total number of instances: 1043.

Count of each toxic level: {1: 829, 2: 172, 3: 35, 4: 7}. We find that labels are unbalanced.

### Step 1:
We need to randomize the total dataset because comments in the initial dataset are ordered by articles. There are two ways to randomize: 

- Randomize comments without considering information about articles.
- Randomize comments by considering information about articles. More specifically, we split randomly comments in one article into cross-validation folds and make sure each fold has balanced comments from one article. 

I chose to use the first way because it is closer to the real test data. The second way makes sure that each train data covers almost all the articles of comments in test data during cross-validation. However, it is not true in real test data. Once the model is trained by all training data, it could be used to predict new comments from new articles. In this case, the training dataset doesn't cover articles of comments in real test dataset. So the second split way overestimates the performance of the trained model, while the first way to randomize estimates the model performance accurately. That's why I chose the first way.

### Step 2:

There is no need to distinguish capitalization such as "The" and "the", so I convert all uppercase to lowercase. 

Total vocabulary size in all comments is 8569.

I don't want to delete all punctuations  because some of them such as "!", "?" may show the toxic level.

However, stopwords such as "the", "of" have no sense to be a feature so I decided to delete them from the vocabulary. I used the intrinsic nltk english stopwords and some punctuations set [".", ",", "'s", "'", ")", "(", "-", ":"] to build my stopwords set for this task.

I filtered tokens which only appear once because they could be misspelled, or they are rare words. It would be useless to keep them in features. 

After filtering tokens by stopwords and frequency >= 2, the vocabulary size becomes 3789. After adding "UNK" to the vocabulary, the total size becomes 3790.

I trained a logistic regression multi-class classifier where the features are just the vocabulary set and feature values are frequencies of the vocabulary appearing in the comment. The average accuracy of this classifier is 0.76. The majority-vote baseline gives us an accuracy of 0.79 since the dataset is quite unbalanced. Our logistic regression classifier is slightly weaker than the majority-vote baseline. I think it is because the logistic regression classifier is overfitted with 3790 features and only 1043 training data. I verified this hypothesis by adding regularized penalty to the logistic regression and the accuracy increases as expected. 

For better understanding the classifier performance, I report the precision, recall, f1-score for each class, shown as below. The classifier predicts well on the toxic level 1 class but poorly on level 3 or level 4. It is reasonable because there are much more comments in level 1 than level 3 or 4 in dataset. It could sometimes predict comments in level 2 but both precision and recall are low.

|             | precision |  recall | f1-score |  support |
|------------ |-----------|---------|----------|----------|
|         1   |   0.81    |   0.92  |   0.86   |   170    |
|         2   |   0.12    |   0.07  |   0.09   |    28    |
|         3   |   0.00    |   0.00  |   0.00   |     9    |
|         4   |   0.00    |   0.00  |   0.00   |     2    |
| avg / total |   0.68    |   0.76  |   0.71   |   209    |

I chose "accuracy" as the metric to evaluate model performance for this task. Alternative metric could be f1-score. However, the micro average f1-score is exactly same as the accuracy for multi-class problem. Macro average f1-score is not a good choice because the training data is quite unbalanced and we don't want to weight classes equally when computing the average. "Accuracy" is a well-suited metric to evaluate the global performance for this task.

### Step 3:
As the last step shows, the logistic regression classifier was overfitted because of its huge number of features. Feature reduction is a key issue for this task. I tried to make a simple improvement by reducing the vocabulary size. I filtered all tokens which have less than 2 characters such as most punctuations and single letters because there are quite meaningless in the original vocabulary. The filtered vocabulary size for new model is 3682. This improvement filtered nearly 100 tokens out of the vocabulary. The average accuracy of the new model is 0.77 by cross-validation.

### Step 4:
I used Bootstrap Test for comparing the new model and the old one. I generated 100 samples for the Bootstrap test. The significant test shows that the new model is better than old model by 0.0086 in accuracy with p-value equal to 0.04. The p-value was sufficiently small, less than the standard thresholds of 0.05, then we might reject the null hypothesis and agree that 0.0086 was a sufficiently surprising difference and the improved model is really better than the initial one. 

### Step 5:
The above experiments show that one main issue is that the classifier is overfitted because of large feature set (3000+) and relatively small training dataset (1000+). Feature reduction may solve this problem. I have tried some improvements by preprocessing the vocabulary set such as setting more strict filters. However, it doesn't make a significant improvement. I would like to explore some efficient feature reduction methods during this step.

Feature reduction can be done in two different ways[1]:

- By only keeping the most relevant variables from the original dataset (this technique is called feature selection).

- By finding a smaller set of new variables, each being a combination of the input variables, containing basically the same information as the input variables (this technique is called dimensionality reduction). 

The question that I pose for this step is: **feature selection and dimensionality reduction, which way is better suited for this task?**

I chose [Chi2]([https://nlp.stanford.edu/IR-book/html/htmledition/feature-selectionchi2-feature-selection-1.html]) test and Mutual Information ([MI](https://nlp.stanford.edu/IR-book/html/htmledition/mutual-information-1.html)) test as representations of feature selection methods, and Principal Component Analysis ([PCA](http://cs229.stanford.edu/notes/cs229-notes10.pdf)) and Singular Value Decomposition ([SVD](http://theory.stanford.edu/~tim/s15/l/l9.pdf)) as representations of dimensionality reduction methods. I designed experiments by comparing these four methods in order to answer this question.

Based on the improved model in step 3, I added different feature reduction methods before feeding them into logistic regression classifier. Initially, the feature size is 3683 (including "UNK"). Aiming to compare these four methods more thoroughly, I run 11 experiments for each feature reduction method from 3683 dimension to one of the reduced dimensions [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]. I chose 1000 as the maximum reduced dimension for this task because the training dataset has nearly 1000 instances. More features would make the classifier overfitting so there is no interest to compare it.

The experimental results are shown in the figure below. Overall, feature selection methods represented by Chi2 and MI outperform dimensionality reduction methods represented by PCA ans SVD. Additionally, performance of PCA and SVD are almost the same and they make the classifier even worse than the baseline model; performance of Chi2 is slightly better than MI and they both outperform the baseline model. 

<p align="center">
  <img src="https://github.com/PittNLP/meg168-project1/blob/master/fig_step5.jpg" />
</p>

The result seems surprising to me because I didn't expect that PCA and SVD perform so bad. We know that when the reduced dimension is the same, PCA and SVD methods contain more information of features than feature selection methods because they combine variables to components which keep as much variance as possible instead of directly deleting some features. So Chi2 and MI methods lose more information of features after reducing the dimension. However, their lost information is much less important to the classifier because they only keep the most relevant features which could discriminate target classes (labels) and drop others. Even though PCA and SVD methods lose least information of features, they didn't take target classes into consideration. For this specific task which has unbalanced training data, instances from class 1 occupy around 80% of rows in total feature matrix. PCA and SVD methods may only extract important features (which has more variance) for class 1 while ignoring some important features for class 3 and 4 because they are much less represented in the initial feature matrix (4% of rows). This may explain why PCA and SVD methods perform worse in this task than expected (previous works[1] suggest PCA and SVD perform well in general cases). 

Additionally, we notice that the performance of Chi2 is better than MI. It didn't surprise me because Chi2 is generally more efficient than MI[2]. Chi2 arrives to the best accuracy when dimension is equal to 200, then the performance decreases while the dimension increases. We may conclude that around 200 tokens in the vocabulary are essential for the classifier and more useless features make the classifier overfitted.

It's hard for me to explain why the performance of PCA and SVD methods firstly decreases and then increases while the dimension increases. Intuitively, it should firstly increase because more dimension keeps more information of features which helps classify, then it decreases because less variant features bring more noise than useful information for the classifier. We need to design more experiments to understand better this case. Since this is not the main issue for my posed question, I didn't invest more time to study it.

Moreover, I listed top 30 discriminant tokens selected by Chi2 test and MI test as below. There are several interesting words which may flag toxic comments according to common sense: hidden, defeated, left-wing, murder, junk, humorous, worshippers, hurting, fools, entrenched, kill, ghettos, judgement, court, dumped, unproductive, weaker, etc. It shows that these methods are able to select reasonable tokens for toxic prediction. It also explains why these two methods perform well even with only 50 dimensions of features. 

Top 30 discriminant tokens by Chi2 test are ['hidden', 'competitors', 'left-wing', 'agenda', 'suggesting', 'openly', 'proof', 'defeated', 'mission', 'consecutive', 'horse', 'tactics', 'lock', 'murder', 'junk', 'humorous', 'informative', 'worshippers', '*cough', 'richmond', 'protect', 'ideas', '20th', 'hurting', 'collective', 'fools', 'entrenched', 'deck', 'nobody', 'kill'].

Top 30 discriminant tokens by Mutual Information test are ['staff', 'futures', 'nationals', 'questioning', 'favours', 'parallel', 'ghettos', 'officer', 'cause', 'judgement', 'revenue', 'examinations', 'specific', 'known', 'anywhere', 'disappear', 'protect', 'court', 'indians', 'dumped', 'foreigners', 'hard', 'month', 'pays', 'benedict', 'unproductive', 'without', 'silent', 'weaker', 'russians']

In conclusion, feature selection is better suited than dimensionality reduction for this task. One possible reason is that principal components with maximum variance may be not discriminant tokens for the target classes because the training dataset is too unbalanced. Human natural languages vary quite a lot even when people express non-toxic comments. Another reason could be that only 15% tokens (around 200) out of the total vocabulary (3000+) are useful for this task. So it doesn't make much sense to combine several variables into principal components because it may lose important information for this task while combining them.

**References**:

[1] Sorzano C O S, Vargas J, Montano A P. A survey of dimensionality reduction techniques[J]. arXiv preprint arXiv:1403.2877, 2014.

[2] Yang Y, Pedersen J O. A comparative study on feature selection in text categorization[C]//Icml. 1997, 97: 412-420.
