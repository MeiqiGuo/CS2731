# PROJECT 2

This project is about the letter-based language models and their applications.

## README

## Environment

python 3.6.5

tensorflow 1.12

## Run experiments

All the results shown in my report could be reproduced by running these commands. I only submitted the trained RNN model datafile because it takes long time for training. Those ngram models could be trained in less than 10 minutes, so I didn't upload them.

Run --help for more detail:

>python ngram.py --help

>python rnn.py --help

>python pinyin.py --help

Run code for step 1

1) Choose the best N for Ngram models evaluated on dev dataset:

>python ngram.py "english/train" "english/dev" "english/test" 0 1 -dev

2) Run 6-gram language model evaluated on dev dataset:

>python ngram.py "english/train" "english/dev" "english/test" 6 1 -dev

Run code for step 2

1) Choose discount parameter for the 9-gram model with Jelinek-Mercer smoothing on dev dataset:

>python ngram.py "english/train" "english/dev" "english/test" 9 2 -dev

2) Run 9-gram model with Jelinek-Mercer smoothing on dev dataset and set the discount parameter equal to 0.7:

>python ngram.py "english/train" "english/dev" "english/test" 9 2 -dev --discount 0.7

3) Run 9-gram model with Witten-Bell smoothing on dev dataset:

>python ngram.py "english/train" "english/dev" "english/test" 9 2 -wb -dev

4) Run bootstrap test for comparing these three models on test dataset:

>python ngram.py "english/train" "english/dev" "english/test" 0 2 -comp

Run code for step 3

1) Train a 2-layer LSTM on training data with 256 embedding dimension and 512 hidden units for 60 epochs

>python rnn.py "english/train" "english/dev" "save/vocab.pkl" "save/training_checkpoints_2layerDropout2" 256 512 60

2) Test the trained model on test data. Notice that the result could only be reproduced on GPU (issue reported [here](https://github.com/keras-team/keras/issues/9463))

>python rnn.py "english/train" "english/test" "save/vocab.pkl" "save/training_checkpoints_2layerDropout2" 256 512 0

3) Continue training model based on the most recent checkpoints weights fot another 10 epochs

>python rnn.py "english/train" "english/dev" "save/vocab.pkl" "save/training_checkpoints_2layerDropout2" 256 512 10 -load_previous

Run code for step 4

1) Choose the best N-gram models and evaluate on dev data::

>python pinyin.py "chinese/train.han" "chinese/dev.han" "chinese/test.han" "chinese/charmap" "chinese/dev.pin" "chinese/test.pin" 0 -dev

2) Train 3-gram model and evaluate on dev data:

>python pinyin.py "chinese/train.han" "chinese/dev.han" "chinese/test.han" "chinese/charmap" "chinese/dev.pin" "chinese/test.pin" 3 -dev

3) Train 3-gram model and evaluate on test data:

>python pinyin.py "chinese/train.han" "chinese/dev.han" "chinese/test.han" "chinese/charmap" "chinese/dev.pin" "chinese/test.pin" 3

4) Evaluate a baseline with random prediction approach on dev data:

>python pinyin.py "chinese/train.han" "chinese/dev.han" "chinese/test.han" "chinese/charmap" "chinese/dev.pin" "chinese/test.pin" 3 -dev -baseline

5) Evaluate a baseline with random prediction approach on test data:

>python pinyin.py "chinese/train.han" "chinese/dev.han" "chinese/test.han" "chinese/charmap" "chinese/dev.pin" "chinese/test.pin" 3 -baseline

## REPORT

### Step 1

During this step, I built Ngram models without smoothing. 

The total vocabulary size of train, dev and test dataset is 101 (includng two special charaters for starting symbol and ending symbol of sentences).

Regarding the evaluator for testing, it predicts the character which has the highest probability of the ngram model. If several characters all have the highest probability, it predicts the first one appearing in the vocabulary. Notice that it randomly predict the next charater if the given (up to N-1) previous charaters don't exist in the trained language model since there is no smoothing in this step. 

The final accuracy for each Ngram models on the dev dataset is shown in the figure below. We find that the performance achieves the best accuracy 0.548 when N is equal to 6. We chose the 6-gram model as the best non-smoothing model. When N increases beyond 6, the accuracy for predicting the next character decreases. It may be because of the lack of smoothing. We implement smoothing in the next step.

<p align="center">
  <img src="https://github.com/PittNLP/meg168-project2/blob/master/save/ngram_acc.jpg" />
</p>

### Step 2

I improved my basic model by implementing two smoothing methods: Jelinek-Mercer smoothing and Witten-Bell smoothing[1]. 

For Jelinek-Mercer smoothing, I assume that the discount paramter "lambda" is constant and independent of the context. In order to choose the optimal discount paramter, I evaluated possible values from 0.1 to 1 on dev dataset. Results are shown in figure below. 

<p align="center">
  <img src="https://github.com/PittNLP/meg168-project2/blob/master/save/ngram_acc_JMsmoothing_all.jpg" />
</p>

We found that the accuracy acheives the best when discount equal to 0.7. Notice that there is no smoothing when it is equal to 1, which explains why there is a sharp decrease at 1. In order to better choose the optimal discount value, I evaluated possible values from 0.6 to 0.9 on dev dataset, shown in figure below. The best accuracy is 0.6176 when discount value is equal to 0.7. We chose the 9-gram model with the discount parameter equal to 0.7 as the best model with Jelinek-Mercer smoothing. 

<p align="center">
  <img src="https://github.com/PittNLP/meg168-project2/blob/master/save/ngram_acc_JMsmoothing.jpg" />
</p>

For Witten-Bell smoothing, the discount paramters are computed by the frequency of context. We don't need to choose it by ourselves. Please refer to references for more detial. The final accuracy on dev test is 0.6174.

I would like to compare these three Ngram language models: 6-gram model without smoothing, 9-gram model with Jelinek-Mercer smoothing (discount parameter is set to 0.7) and 9-gram model with Witten-Bell smoothing. I rigorously compared the performance of these three models on test dataset and making the significant test with bootstrap method (sampled 10 times). 

The accuracies on test dataset of these three models are respectively 0.5570, 0.6078 and 0.6078. 9-gram model with Jelinek-Mercer smoothing is better than 6-gram model without smoothing by 0.0508 in accuracy with p_value 0.0; 9-gram model with Witten-Bell smoothing is better than 6-gram model without smoothing by 0.0508 in accuracy with p_value 0.0; 9-gram model with Witten-Bell smoothing is better than Jelinek-Mercer smoothing by 0.0 in accuracy with p_value 0.3. 

In conclusion, the 9-gram model with either Jelinek-Mercer smoothing or Witten-Bell smoothing is significantly better than the 6-gram model without smoothing by 5.08% in accuracy at the 0.05 level. The 9-gram model with Witten-Bell smoothing is not better than Jelinek-Mercer smoothing, neither in the reverse direction.

## Step 3

After trying different architectures of RNN model, I chose a 2-layer LSTM with dropout layers after both LSTM layers with a rate equal to 0.2. The model has 512 hidden units for both LSTM layers and the embedding dimension is 256. The estimated time for training one epoch is 70s on GPU (1 core). This model gives an accuracy on test data as 0.6265, which is better than 9-gram model with smoothing obtained in step 2 by 0.0187 in accuracy. Notice that the final result shown here is after training 60 epochs. 

I would like to share several observations when I was trying different architectures and tuning the hyperparameters. The metric for evaluating the model performance is the accuracy of the next character prediction task.

- Keeping the same architure of RNN model, bigger the model size (more hidden units and more embedding dimension), better the performance.

- 2-layer LSTM outperforms single-layer LSTM.

- 2-layer LSTM with dropout outperforms the one without dropout.

- With the 2-layer LSTM without dropout model, the accuracy doesn't increase when the loss decreases after a certain epochs of training. It may be a sign of overfitting. 

- The rate of dropout impacts the performance of model. Here, 0.2 is better than 0.5.

## Step 4

Even though the previous step shows that RNN model outperforms the ngram models, I still chose ngram model with Witten-Bell smoothing for this han prediction task. It is because 1) the fine-tuned RNN model outperforms **slightly** the ngram model with smoothing; 2) a RNN model lack of tuning performs even worse than ngram model with smoothing; 3) training an RNN model takes more time; 4) Witten-Bell smoothing doesn't need hyper-parameters and it performs as well as Jelinek-Mercer smoothing according to step 2.

I followed the instructions for the implementation:

- Get vocabulary set from all dataset "chinese/train.han", "chinese/dev.han", "chinese/test.han".

- Build a mapping dictionary from pinyin to list of han characters candidates.

- Train han characters language model on the training dataset "chinese/train.han" by 3-gram model with  Witten-Bell smoothing.

- Evaluator predicts the han character based on the pinyin and the history of correct han characters. If the pinyin is the space token, then predict " " directly; if the pinyin only has one letter, then add it to the han candidate list if it doesn't already exist in the list. Then compute the probability of each han character in the candidate list and choose the one with the highest probability as the predicted han character. 

I chose the best N-gram model with Witten-Bell smoothing by comparing the accuracy of predicting the next han character on dev dataset. According to fugure shown as below, the 3-gram model gave the highest accuracy as 0.8415 on dev dataset. 

<p align="center">
  <img src="https://github.com/PittNLP/meg168-project2/blob/master/save/han_ngram_acc.jpg" />
</p>

This 3-gram model achieved an accuracy of 0.8008 on test dataset. In order to show the difficulty of this task, I built a baseline to compare with this 3-gram model. The baseline predicts randomly a han character from the candidate list. The accuracies of baseline on dev dataset and test dataset are respectively 0.4487 and 0.3111. This result shows that 1) the prediction task on test dataset is more difficult than the dev dataset, which explains why this 3-gram model performs slightly worse on test dataset than dev dataset; 2) our 3-gram model performs well enough on this prediction task because it is much better than a random guess by 0.4897 in accuracy.


### References:

[1] https://nlp.stanford.edu/~wcmac/papers/20050421-smoothing-tutorial.pdf 



