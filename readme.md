# Neural Machine Translation with Character Level Decoder

This repository is a modularized re-write of [Stanford CS224N](http://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/) [assignment 5](http://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/assignments/a5.pdf). Specifically the Character-Level LSTM Decoder for Neural Machine Translation. 

I wanted to re-write it as a way of understanding the different components of the model and then eventually the tricks used to train the model - I have somewhat annotated a few notebooks for my own reference.  

## Usage
Docker support coming soon. Meanwhile: 

1. Clone repository
2. Install the requirements using 
  ```bash
  pipenv install
  ```
  if running the train and test tasks. If browsing notebooks, use:
  ```bash
  pipenv install --dev
  ```
3. Download and place the Assignment 5 data from [Stanford CS224N](http://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/) in `nmt/datasets/data/`.
4. Run tasks as
  ```bash
  pipenv run sh tasks/<task-name>.sh
  ```
  Possible tasks:
  * `train_local.sh`  : training using small sample (equivalent to train-local-q2 from the assignment)
  * `test_local.sh`   : testing using small sample (equivalent of test-local-q2 from the assignement) - should produce BLEU score of ~99.27
  * `train.sh`        : training using all data on GPU.
  * `test.sh`         : testing using all data - should produce BLEU score of ~29.40
  
###  Dependencies for installation
* Python 3.6 (if using Pipfile; 3.6+ if using requirements.txt)
* Pipenv

## References
* [Stanford CS224N](http://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/)
* [Assignment 5 Handout](http://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/assignments/a5.pdf)
* [Achieving Open Vocabulary Neural Machine Translation with Hybrid Word-Character Models](https://arxiv.org/abs/1604.00788) -  Minh-Thang Luong and Christopher Manning. 
* [Character-Aware Neural Language Models](https://arxiv.org/abs/1508.06615) - Kim et al. (2016)
* [Attention? Attention!](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html) - Lilian Weng
* [D2l.ai](http://d2l.ai/chapter_recurrent-neural-networks/text-preprocessing.html) - Aston Zhang, Zachary C. Lipton, Mu Li, Alexander J. Smola
