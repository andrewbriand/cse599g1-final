# Ultimate Tic-Tac-Toe
## Andrew Briand's CSE599G1 final project

## Abstract

The goal of this project was to create an AI to play Ultimate Tic-Tac-Toe using an approach similar to the original AlphaGo \[1\].

## Problem Statement

Ultimate Tic-Tac-Toe is played on a board composed of 9 standard 3x3 Tic-Tac-Toe boards arranged in a 3x3 grid. The goal of the game is to win 3 of these standard 3x3 Tic-Tac-Toe boards such that they form a row of three in the grid before the opponent does. You can play Ultimate Tic-Tac-Toe against yourself or with a friend here to help you understand the rules: https://ultimate-t3.herokuapp.com/local-game.

## Related work



## Methodology

Due to time and resource constraints, it was not possible to replicate the entire training regime and search algorithm of the AlphaGo paper. The method used is as follows.

### Phase 1

In the first phase of training, networks were trained to predict the moves made in games played by the top 20 AIs in the Ultimate Tic-Tac-Toe competition on codingame.net. 4,906 games comprised of 189,662 unique position-move pairs were downloaded from the site and then split 80-10-10 into train, dev, and test sets by game. After experimenting by hand with various architectures, a hyperparameter search was performed with Bayesian optimization to find the best network architecture and learning rate.

The networks' input consisted of an 9x9x2 tensor in which the first channel was set to 1 at cells occupied by the current player and the second channel was set to 1 at cells occupied by the opposing player. All other inputs were set to 0. The output consisted of a 9x9 tensor representing a probability distribution over possible moves.

During training, positions were randomly rotated and reflected before being input into the network in an attempt to reduce overfitting and make the network more robust. This is possible because reflecting or rotating an Ultimate Tic-Tac-Toe position yields an equivalent position. 

Experiments were also run with networks whose input consisted of a 9x9x3 tensor, in which the third channel was set to 1 for cells where a legal move could be played and 0 elsewhere, but this offered no performance improvement.

Networks were trained with the Adam optimization algorithm using categorical cross-entropy loss. Hyperparameters were selected to maximize validation accuracy, i.e. how often a given network assigned the largest probability in its distribution to the move actually made in games in the validation set. During the hyperparameter search, networks were trained for 20 epochs. Then, two architectures were selected and these were trained for 200 epochs on the training set. The validation accuracy for these 200-epoch runs was maximized at around epoch 100, so the networks were then retrained on the training and validation sets for 100 epochs for evaluation on the test set.

## Evaluation

## Results
### Phase 1
#### Architecture

Two architectures were chosen from the hyperparameter search. The search chose a learning rate of 1e-2 for both networks.

The first consists of the following for a total of 126k trainable parameters:

1. 32 3x3 filters convolved over the input with a stride of 3.
2. 32 3x3 filters convolved over the input with a stride of 1.
3. A ReLU activation applied to the concatenation of 1 upsampled to a 9x9x32 tensor and 2.
4. A batch-normalization layer.
5. 12 convolutional layers consisting of 32 3x3 filters, each followed by ReLU activation and batch normalization.
6. A convolution layer consiting of 1 1x1 filter, followed by ReLU activation and batch normalization. 
7. A dense layer with 81 outputs with softmax activation to form the 9x9 output tensor. 

The second consists of the following for a total of 36k trainable parameters:
1. 32 3x3 filters convolved over the input with a stride of 3.
2. 8 3x3 filters convolved over the input with a stride of 1.
3. A ReLU activation applied to the concatenation of 1 upsampled to a 9x9x32 tensor and 2.
4. A batch-normalization layer.
5. 6 residual units in which the concatenation of the upsampled output of 8 3x3 filters with a stride of 3 and 16 3x3 filters with a stride of 1 undergoes ReLU activation and batch-normalization and is then fed into a convolution of 24 1x1 filters. The result of this undergoes ReLU activation and batch normalization and is then added to the input of the unit.
6. A convolutional layer consisting of 1 1x1 filter followed by ReLU activation and batch normalization.
7. A dense layer with 81 outputs with softmax activation to form the 9x9 output tensor.

#### Accuracy

The first network achieved an accuracy of 48.7% on the test set and the second achieved an accuracy of 49.2% on the test set.

### Phase 2


