## Ultimate Tic-Tac-Toe
### Andrew Briand's CSE599G1 final project

### Abstract

The goal of this project was to create an AI to play Ultimate Tic-Tac-Toe using an approach similar to the original AlphaGo \[1\].

### Problem Statement

Ultimate Tic-Tac-Toe is played on a board composed of 9 standard 3x3 Tic-Tac-Toe boards arranged in a 3x3 grid. The goal of the game is to win 3 of these standard 3x3 Tic-Tac-Toe boards such that they form a row of three in the grid before the opponent does. You can play Ultimate Tic-Tac-Toe against yourself or with a friend here to help you understand the rules: https://ultimate-t3.herokuapp.com/local-game.

### Related work



### Methodology

Due to time and resource constraints, it was not possible to replicate the entire training regime and search algorithm of the AlphaGo paper. The method used is as follows.

#### Phase 1

In the first phase of training, networks were trained to predict the moves made in games played by the top 20 AIs in the Ultimate Tic-Tac-Toe competition on codingame.net. XXX games comprised of XXX positions were downloaded from the site and then split 80-10-10 into train, dev, and test sets. After experimenting by hand with various architectures, a hyperparameter search was performed with Bayesian optimization to find the best network architecture and learning rate.

The networks' input consisted of an 9x9x2 tensor in which the first channel was set to 1.0 at cells occupied by the current player and the second channel was set to 1.0 at cells occupied by the opposing player. All other inputs were set to 0.0. The output consisted of a 9x9 tensor representing a probability distribution over possible moves.

During training, positions were randomly rotated and reflected before being input into the network in an attempt to reduce overfitting and make the network more robust. This is possible because reflecting or rotating an Ultimate Tic-Tac-Toe position yields an equivalent position. 

Experiments were also run with networks whose input consisted of a 9x9x3 tensor, in which the third channel was set to 1.0 for cells where a legal move could be played and 0.0 elsewhere, but this offered no performance improvement.

Networks were trained with the Adam optimization algorithm using categorical cross-entropy loss. Hyperparameters were selected to maximize accuracy, i.e. how often a given network assigned the largest probability in its distribution to the move truly made in the original game. During the hyperparameter search, networks were trained for 20 epochs. Then, two architectures were selected and these were trained for 200 epochs on the training set. The validation accuracy for these 200-epoch runs was maximized at around epoch 100, so the networks were then retrained on the training and validation sets for 100 epochs for evaluation on the test test.

##### Architecture

Two good architectures were chosen from the hyperparameter search 


### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [Basic writing and formatting syntax](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/andrewbriand/cse599g1-final/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
