# CS231N

## Sanity checks for a NN

1. ### Check data loss, meaining zero regularisation, first (for cifar-10):

   1. softmax loss should be  -log(0.1) as there are 10 classes and each class should give probabilty of 1/10
   2. expect all desired margins to be violated (since all scores are approximately zero), and hence expect a loss of 9 (since margin is 1 for each wrong class)
   3. if these conditions are not fulfilled then there might be an **initiliastion issue**

2. ### Increasing regularisation strength from zero should increase loss

3. ### Overfit a tiny subset of data

   - train on about 20-50 examples and achieve zero cost and 100% training accuracy
   - also set reg to 0, can prevent zero cost

## Train/Val Accuracy

- The gap between the training and validation accuracy indicates the amount of overfitting.

1. ### Validation error curve shows a very small validation accuracy compared to the training accuracy (large diff)

    - this indicates strong overfitting (note, it's possible for the validation accuracy to even start to go down after some point)
    - increase regularization (stronger L2 weight penalty, more dropout, etc.) or collect more data

2. ### Validation accuracy tracks the training accuracy, very close

   - model capacity is not high enough: make the model larger by increasing the number of parameters

## Ratio of weights and updates

- ratio of update magnitudes to the value magnitudes
- updates are the changes in grad, so in vanilla sgd it will be learning_rate*grad
- rough heuristic: ratio should be about **1e-3**
- if it is lower, then **learning rate is too low** and vice-versa
- compute and track the norm of the gradients and their updates

## First-layer visualisations

- Noisy features indicate could be a symptom: Unconverged network, improperly set learning rate, very low weight regularization penalty
- Nice, smooth, clean and diverse features are a good indication that the training is proceeding well