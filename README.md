# Paper-2

This is a neural machine translation project from Indonesian to English translation, using the method in Paper: <Machine Translation in Low-Resource Languages by an Adversarial Neural Network.>

Require:
=====================
Tensorflow 2.4.1  
numpy 1.19.5  
Python 3.7.6  
=====================

Run：
=====================
train.py
=====================

The reults are saved in the following files:
=====================
Discriminator weight: model/ind/D_weight  
Generator weight: model/ind/G_weight  
Training losses: model/ind/log  
translation results: model/ind/result  
=====================

Discriminator weight and Generator weight are saved every 20 epochs  
Training losses are saved every 1 epoch  
translation results are saved every 10 epochs  

This model gets good performance in 500 epochs  
Pretrained results have been saved in model/ind  
