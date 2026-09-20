# Feed-Forward Neural Networks with Backpropagation

An implementation of configurable feed-forward neural networks for **classification and regression**, with backpropagation, optional momentum, and experiments comparing zero, one, and two hidden layers.

[Read the full paper](FinalPaper.pdf) · [Source code](src/main/)

## Technical highlights

- Configurable input size, hidden-layer sizes, and output size.
- Forward propagation and gradient-based weight updates.
- Sigmoid hidden-layer activations, softmax classification outputs, and linear regression outputs.
- Optional momentum and configurable learning rate.
- Min-max scaling, one-hot encoding, and dataset-specific preprocessing.
- Evaluation utilities for classification error, mean squared error, and average convergence rate.

## Experimental study

The paper investigates the relationship between dataset characteristics, network depth, and model performance across six datasets:

| Classification | Regression |
| --- | --- |
| Wisconsin breast cancer | Computer hardware |
| Soybean | Abalone |
| Glass | Forest fires |

The workflow uses stratified ten-fold cross-validation and a tuning subset. It varies learning rate, momentum, and hidden-layer size, then reports 0/1 loss for classification and mean squared error for regression.

## Run an experiment

Use a Java JDK with `javac` and `java` available. From the repository root:

```bash
  mkdir -p out
  javac -d out $(find src/main -name '*.java')
  cp -R src/resources/* out/
  java -cp out main.drivers.ComputerDriver
```

The inspected driver reads `machine.data`, constructs a regression network, trains it, and prints predictions and evaluation statistics.

Network architecture and training settings are configured in the driver. Other dataset entry points include `AbaloneDriver`, `BreastDriver`, `ForestDriver`, `GlassDriver`, and `SoybeanDriver`, with additional variants for tuning and reporting. A single driver invocation does not reproduce every configuration in the paper.

## Repository guide

| File | Purpose |
| --- | --- |
| [`NeuralNetwork.java`](src/main/nn/NeuralNetwork.java) | Network initialization, forward propagation, training, and convergence tracking. |
| [`MinMaxScale.java`](src/main/utils/MinMaxScale.java) | Scaling utilities. |
| [`OneHotEncoder.java`](src/main/utils/OneHotEncoder.java) | Classification-label encoding. |
| [`LossFunctions.java`](src/main/utils/LossFunctions.java) | Evaluation metrics. |
| [`TenFoldCrossValidation.java`](src/main/utils/TenFoldCrossValidation.java) | Dataset splitting utilities. |
| [`drivers`](src/main/drivers/) | Dataset drivers. |

## Limitations and reproduction notes

Learning rates, momentum settings, and layer sizes were explored over a limited set of choices, and that only six datasets were evaluated.

For a rigorous rerun, record the chosen driver and configuration, validate fold and tuning-set separation, and ensure preprocessing is fit only on training data. The inspected driver contains experimental preprocessing and splitting choices that should be reviewed before treating its output as an unbiased held-out estimate.

## Contributors and related work

- **Michael Downs:** implementation of the code and hyperparameter tuning; abstract, software design, and shared writing/review.
- **Max Hymer:** paper review/editing and shared writing.

The follow-up [NeuralNetGAs project](https://github.com/michaeldowns203/NeuralNetGAs) explores genetic algorithms, differential evolution, and particle swarm optimization as alternative network-training methods.
