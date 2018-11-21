# ImageCaptioning
Generating image descriptions using deep learning (CNN, RNN)

# About the project

This was a final project for [PSIML2018](http://psiml.petnica.rs/) organised by [Microsoft Development Center Serbia](https://www.microsoft.com/sr-latn-rs/mdcs). The idea is to describe images based on its content using neural neutworks (convolutional + recurrent).

# Problem to solve

The system should try to give possible description given an image as showed on Figure 1. 
Questions that should be answered: 
- What kind of architecture we need?
- Which part of the image affect which word in the output?

Figure 1: Possible explanations of the image


# Solution

Solution to this problem is found in paper [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555). 
Architecture that is proposed in this paper (Figure 2) contains Convolutional Neural Network (marked as DNN) which extracts features from images. These features now represent input to RNN (LSTM) together with embeddings of the words. 

Figure 2: Architecture from Show and Tell paper

