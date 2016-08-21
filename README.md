## Weibo Classifier based on CNN
Code for the paper [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882) (EMNLP 2014).

User portraiting analyzing Project. It is a classifier that can predict the users' gender, age and location according to the Weibo content.

### Requirements
Code is written in Python requires Theano.

Using the pre-trained word2vec vector training on the Chinese wiki corpus, such as `wiki_chs_pyvec`.


### Traning models

#### Data Preprocessing
To classify the raw data, run

```
python classifier_gender.py / classifier_birth.py / classifier_location.py
```
To divide the weibo into 2, 3 and 8 categories according to gender, age, location.


To load the word vector of the raw dara, run

```
python load_w2v_gender.py / load_w2v_birth.py / load_w2v_location.py 
```
This will create a pickle object called `weibo~.p` in the same folder, which contains the dataset in the right format.

#### Training
To train Model and save the paramaters, run

```
python conv_net_gender.py / conv_net_birth.py / conv_net_location.py 
```
The parameters are saved as `~Model`, and the word vectors are saved as `~Vector` in the same folder.

exampeles of the gender classifier output:

```
loading data... data loaded!
model architecture: CNN-non-static
using: word2vec vectors
[('image shape', 508, 100), ('filter shape', [(50, 1, 3, 100), (50, 1, 4, 100), (50, 1, 5, 100)]), ('hidden_units', [50, 2]), ('dropout', [0.5]), ('batch_size', 100), ('non_static', True), ('learn_decay', 0.95), ('conv_non_linear', 'relu'), ('sqr_norm_lim', 9), ('shuffle_batch', True)]
... training
epoch: 1, training time: 11963.42 secs, train perf: 76.93 %, val perf: 77.00 %
epoch: 2, training time: 15271.62 secs, train perf: 77.60 %, val perf: 77.65 %
epoch: 3, training time: 15026.33 secs, train perf: 78.74 %, val perf: 78.37 %
epoch: 4, training time: 15478.64 secs, train perf: 79.69 %, val perf: 78.99 %
epoch: 5, training time: 15259.53 secs, train perf: 81.37 %, val perf: 79.39 %
epoch: 6, training time: 15503.38 secs, train perf: 82.94 %, val perf: 79.75 %
epoch: 7, training time: 15674.22 secs, train perf: 83.54 %, val perf: 80.09 %
epoch: 8, training time: 15579.79 secs, train perf: 84.42 %, val perf: 80.20 %
epoch: 9, training time: 15705.87 secs, train perf: 85.57 %, val perf: 80.34 %
epoch: 10, training time: 15684.07 secs, train perf: 87.16 %, val perf: 79.98 %
cv: 0, perf: 0.797593524393
```

### Predicting user portrait

#### Data Preprocessing
To load the word vector of the raw dara, run

```
python load_test.py
```
This will create a pickle object called `~Test.p` in the same folder, which contains the dataset in the right format.

#### Predicting
To load the model and word vector for predicting, run

```
python predict_gender.py / predict_birth.py / predict_location.py
```
This will creat a file called `~Predict` in the same floder which contains the weibo contents and their predicting results.

### Merge the results 
To predict the portrait of every user according to their weibos predicting results. The portrait will generate on the basis of proportion of the predicting results, run

```
python predict.py 
```
This will create a file called `temp.csv` in the same folder, which contains all the users' portarits.



