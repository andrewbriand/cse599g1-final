# Ultimate Tic-Tac-Toe
## Andrew Briand's CSE 599G1 final project

## Abstract

The goal of this project was to create an AI to play Ultimate Tic-Tac-Toe using an approach similar to the original AlphaGo \[1\].

## Problem Statement

Ultimate Tic-Tac-Toe is played on a board composed of 9 standard 3x3 Tic-Tac-Toe boards arranged in a 3x3 grid. The goal of the game is to win 3 of these standard 3x3 Tic-Tac-Toe boards such that they form a row of three in the grid before the opponent does. A good explanation of the rules can be found here: https://ultimate-t3.herokuapp.com/rules.

## Related Work

I have personally created several AI to play ultimate tic tac toe without deep learning in the past. I used my best one for evaluation at the end of the project.

There is a leaderboard for Ultimate Tic-Tac-Toe AIs at https://www.codingame.com/multiplayer/bot-programming/tic-tac-toe/leaderboard. This is where I collected data for the project.

There is a paper by Bertholon et al. \[2\] which provides an optimal strategy for a _different_ variant of Ultimate Tic-Tac-Toe in which players can play in boards already won by themselves or the opponent. I know of no solution to the variant I tackle in this project. 

## Methodology

Due to time and resource constraints, it was not possible to replicate the entire training regime and search algorithm of the AlphaGo paper. The method that was used is as follows.

### Phase 1

In the first phase of training, networks were trained to predict the moves made in games played by the top 20 AIs in the Ultimate Tic-Tac-Toe competition on codingame.net. 4,906 games comprised of 189,662 unique position-move pairs were downloaded from the site and then split 80-10-10 into train, dev, and test sets by game. After experimenting by hand with various architectures, a hyperparameter search was performed with Bayesian optimization to find the best network architecture, learning rate, and regularization.

The networks' input consisted of an 9x9x2 tensor in which the first channel was set to 1 at cells occupied by the current player and the second channel was set to 1 at cells occupied by the opposing player. All other inputs were set to 0. The output consisted of a 9x9 tensor representing a probability distribution over possible moves.

During training, positions were randomly rotated and reflected before being input into the network in an attempt to reduce overfitting and make the network more robust. This is possible because reflecting or rotating an Ultimate Tic-Tac-Toe position yields an equivalent position. 

Experiments were also run with networks whose input consisted of a 9x9x3 tensor, in which the third channel was set to 1 for cells where a legal move could be played and 0 elsewhere, but this offered no performance improvement.

Networks were trained with the Adam optimization algorithm to minimize categorical cross-entropy loss. Hyperparameters were selected to maximize validation accuracy, i.e. how often a given network assigned the largest probability in its distribution to the move actually made in games in the validation set. During the hyperparameter search, networks were trained for 20 epochs. Then, two architectures were selected and these were trained for 200 epochs on the training set. The validation accuracy for these 200-epoch runs was maximized at around epoch 100, so the networks were then retrained on the training and validation sets for 100 epochs for evaluation on the test set.

### Phase 2

In the second phase of training, networks played games against themselves and then their parameters were adjusted to maximize the probabilities of moves that led to a win and minimize the probabilities of moves that led to a loss. 

Games were played in batches of 160. Each batch consisted of 10 sub-batches of 16 games played between the current network and a network randomly selected from a pool of networks. Games were played out by sampling according to a distribution obtained by multiplying the network's output element-wise by a 9x9 tensor containing 1's in legal move positions and 0's elsewhere. The resulting tensor was then divided by the sum of its elements to ensure it was a distribution. The pool was initialized to contain only the network and its parameters resulting from phase 1. The current network was initialized to the same network with the same parameters. After each batch of 160 games, an update was applied to the current network's parameters \[1\]:

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{200}&space;\large&space;\Delta&space;\rho&space;=&space;\frac{\alpha}{n}&space;\sum_{i=1}^{n}&space;\sum_{t=1}&space;^{T^i}&space;\frac{\partial&space;\log&space;p_\rho&space;(a_t^i&space;|&space;s_t^i)}{\partial&space;\rho}z_t^i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{200}&space;\large&space;\Delta&space;\rho&space;=&space;\frac{\alpha}{n}&space;\sum_{i=1}^{n}&space;\sum_{t=1}&space;^{T^i}&space;\frac{\partial&space;\log&space;p_\rho&space;(a_t^i&space;|&space;s_t^i)}{\partial&space;\rho}z_t^i" title="\large \Delta \rho = \frac{\alpha}{n} \sum_{i=1}^{n} \sum_{t=1} ^{T^i} \frac{\partial \log p_\rho (a_t^i | s_t^i)}{\partial \rho}z_t^i" /></a>

where z_t^i was 1 when the player whose turn it is at time t goes on to win the game, -1 when the other player wins, and 0 when the game is a draw. Several methods of determining alpha were tested. An initial alpha of 0.001 multiplied by a factor of 0.8 every 20 batches was chosen. 

After 20 batches, the current network was added to the pool. This process was repeated 20 times.

### Phase 3

In phase 3, the network produced at the end of phase 2 played games against itself to produce examples for a value network. The value network's goal is to predict the winner of the game given a single position. 

#### Generation of games

Games were generated by sampling an integer U uniformly from the range \[0, 81). Moves were then played until turn U-1 by sampling from the network's output as in phase 2. The move at turn U was chosen uniformly at random from the legal moves at that time. If a game finished before turn U, that game was ignored. The rest of the moves in the game were sampled as before from the network's output. Each game produced one training example consisting of the position at time U+1 and the result of the game for the player whose turn it was at turn U+1. 200k training positions were generated in this manner. A validation and test set of 20k positions each were generated in the same way.

#### Value network architecture 

I did not have enough time to try many different architectures for phase three. The best I found I constructed by simply replacing the final layer in networks 1 and 2 from phase 2 with a dense layer with ReLU activation and adding an additional dense layer with a single output with tanh activation.

#### Value network training

Value networks were trained for 100 epochs with Adam optimization to minimize Mean Squared Error.  

### Search Algorithm

The search algorithm constructs a tree of nodes, each of which corresponds to a board position. Each node contains 4 values, W, N, Q, and P. The constant c is set to 5. When deciding what move to make, the search first performs _n_ rollouts where each consists of the following 4 steps:

1. Traverse to a leaf node of the tree, choosing the child that maximizes Q + c * P.
2. Evaluate the board position of the leaf node with the value network to obtain a score v, and evalute the board position with the policy network to obtain P values for every legal move at the current board position. Create a child node of the leaf node for each legal move containing the result of applying that move to the leaf node's board position and the P value calculated from the policy network. Initialize the W, N, and Q values of these nodes to 0.
3. Move back up the tree, setting W = W + v, N = N + 1, and Q = W / N for each node encountered along the way.

After all _n_ rollouts have been performed, the algorithm chooses the move leading to the child with the maximum N value.

## Evaluation

### Phase 1

Networks produced by phase 1 were evaluated based on their accuracy on the test set.

### Phase 2

Networks produced by phase 2 were evaluated based on their win-rate against the network from which they were produced (the first network in the pool) and against each other. Moves played in these games were sampled just as during the games played during phase 2 for training purposes.

### Phase 3

Value networks produced by phase 3 were evaluated based on their mean squared error on the test set.

### Search Algorithm

The final search algorithm was evaluated on its win-rate against an MCTS-based AI with a hand-coded policy that I created last year (currently around 50th place on the codingame.net leaderboard). The amount of rollouts the network-based search and the hand-coded search were permitted to perform was varied to examine its effect on win-rate.

## Results
### Phase 1
#### Architecture

Two architectures were chosen from the hyperparameter search. The search chose a learning rate of 1e-2 for both networks. All convolutions were zero-padded when necessary to preserve the size of the input.

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

The first network achieved a win-rate of 90.7% in 1000 games against its initial version. The second network achieved a win-rate of 77.1% in 1000 games against its initial version. The first network achieved a win-rate of 73% in 1000 games against the second network.

### Phase 3

The first network was used to generate data as it beat the second network head-to-head in phase 2. The modified versions of networks 1 and 2 achieved test MSE's of 0.68 and 0.71 respectively.

### Evaluation against another AI

#### Phase 2

The best network resulting from phase 2 was played against my previous hand-coded policy AI with move sampling as before (i.e. with no search). Below is a chart of the number of MCTS rollouts the other AI was permitted to do vs the win-rate and draw-rate of the network. Each match consisted of 30 games, 15 as X and 15 as O.

|Rollouts|250|500|1000|5000|10000|
|--------|---|---|----|----|----|
|Win rate|43.3%|30.0%|40.0%|16.7%|3.3%|
|Draw rate|16.7%|6.7%|16.7%|10.0%|13.3%|

#### Search algorithm

The search algorithm detailed above was played against by previous hand-coded policy AI with the best policy and value networks from phases 2 and 3. Eight games were played and the AIs were permitted 250 rollouts each. The network AI won 3 and lost 5 games. I believe the search's poor performance is due to the poor performance of the value network. The performance of the policy network alone is quite good above and in phases 1 and 2. If I had more time, I would be able to improve the value network and evaluate more rollout numbers for the two AIs. 

## Future Improvements

During game generation for stage 3, it would be preferable to add noise to the network's output distribution for moves up to turn U-1. This is done in the AlphaGo paper to increase the variety of the positions in the training set. This would be beneficial as there were some duplicates and probably many similar positions in the set produced during this project. The AlphaGo paper doesn't specify how noise is added and unfortunately I didn't have enough time to play around with this.

Additionally, considering the value networks I trained had such poor performance, it would be interesting to try a search algorithm using only the policy network and random rollouts. Of course, it would also be interesting to try adding other features of the original AlphaGo, like having a fast rollout policy and a fast tree policy. Experimenting with different values of c and different softmax temperatures for the policy network would also be promising.

I would also like to evaluate the network by uploading it to the codingame leaderboard, but this would require further work as the upload size is very limited and only a single source file is allowed, so I would likely have to compress the network's parameters and write them into the source file itself as an array. 

## Video

<iframe width="560" height="315" src="https://www.youtube.com/embed/cAkqTiut-tE" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Cited Works

1.  Silver, David, et al. “Mastering the Game of Go with Deep Neural Networks and Tree Search.” Nature, vol. 529, no. 7587, Jan. 2016, pp. 484–89. www.nature.com, https://doi.org/10.1038/nature16961.

2. Bertholon, Guillaume, et al. “At Most 43 Moves, At Least 29: Optimal Strategies and Bounds for Ultimate Tic-Tac-Toe.” ArXiv:2006.02353 [Cs], June 2020. arXiv.org, http://arxiv.org/abs/2006.02353


