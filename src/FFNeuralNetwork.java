import java.util.*;
import java.util.stream.Collectors;

public class FFNeuralNetwork {
    private int numInputs;
    private int numOutputs;
    private List<Integer> hiddenLayers;
    private List<double[][]> weights;  // List of weight matrices
    private List<double[]> biases;     // List of bias vectors
    private boolean isClassification;

    // Initialize the lists in the constructor
    public FFNeuralNetwork(int numInputs, List<Integer> hiddenLayers, int numOutputs, boolean isClassification) {
        this.numInputs = numInputs;
        this.numOutputs = numOutputs;
        this.hiddenLayers = hiddenLayers;
        this.isClassification = isClassification;
        this.weights = new ArrayList<>();
        this.biases = new ArrayList<>();
        initializeWeights();
    }

    // Updated initializeWeights to work with double[][] for weights and double[] for biases
    private void initializeWeights() {
        int prevLayerSize = numInputs; // Start with input size

        // Initialize weights and biases for hidden layers
        for (int layerSize : hiddenLayers) {
            weights.add(randomMatrix(layerSize, prevLayerSize));  // weight matrix for each layer
            biases.add(randomVector(layerSize));                  // bias vector for each layer
            prevLayerSize = layerSize;
        }

        // Initialize weights and biases for output layer
        weights.add(randomMatrix(numOutputs, prevLayerSize));     // weight matrix between last hidden layer and output
        biases.add(randomVector(numOutputs));                     // bias vector for output layer
    }

    // Randomly generate weight vector
    private double[] randomVector(int size) {
        double[] vector = new double[size];
        for (int i = 0; i < size; i++) {
            vector[i] = Math.random();
        }
        return vector;
    }

    // Randomly generate weight matrix of doubles
    private double[][] randomMatrix(int rows, int cols) {
        double[][] matrix = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = Math.random();  // Initialize with random values
            }
        }
        return matrix;
    }

    public List<Double> predict(List<Double> instance) {
        double[] inputs = instance.stream().mapToDouble(Double::doubleValue).toArray();  // Convert to double[]

        for (int i = 0; i < hiddenLayers.size(); i++) {
            inputs = activate(multiply(weights.get(i), inputs), biases.get(i), "linear");  // Linear activation for hidden layers
        }

        // Output layer
        inputs = activate(multiply(weights.get(weights.size() - 1), inputs), biases.get(biases.size() - 1), isClassification ? "softmax" : "linear");
        return Arrays.stream(inputs).boxed().collect(Collectors.toList());  // Convert back to List<Double>
    }

    // Activation function: Linear or Softmax
    private double[] activate(double[] values, double[] bias, String activationType) {
        double[] output = new double[values.length];
        if (activationType.equals("linear")) {
            for (int i = 0; i < values.length; i++) {
                output[i] = values[i] + bias[i];  // Simple linear activation
            }
        } else if (activationType.equals("softmax")) {
            double sum = 0.0;
            for (double value : values) {
                sum += Math.exp(value);
            }
            for (int i = 0; i < values.length; i++) {
                output[i] = Math.exp(values[i]) / sum;  // Softmax activation
            }
        }
        return output;
    }

    // Multiply weight matrix with input vector
    private double[] multiply(double[][] matrix, double[] vector) {
        double[] result = new double[matrix.length];  // Result vector size is number of rows in matrix
        for (int i = 0; i < matrix.length; i++) {
            double sum = 0.0;
            for (int j = 0; j < vector.length; j++) {
                sum += matrix[i][j] * vector[j];  // Matrix-vector multiplication
            }
            result[i] = sum;
        }
        return result;
    }

    // Train method using backpropagation
    public void train(List<List<Double>> data, List<String> labels, int epochs, double learningRate) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < data.size(); i++) {
                List<Double> instance = data.get(i);
                String label = labels.get(i);

                // Forward pass
                List<Double> output = predict(instance);

                // Backpropagation
                backpropagate(output, label, learningRate);
            }
        }
    }

    private void backpropagate(List<Double> output, String label, double learningRate) {
        // Convert label to a one-hot encoded target vector (assuming classification)
        List<Double> target = new ArrayList<>(Collections.nCopies(outputLayer.size(), 0.0));
        int targetIndex = Integer.parseInt(label) - 1;  // Assuming label is "1", "2", ..., map it to the correct index
        target.set(targetIndex, 1.0);

        // Calculate the error at the output layer
        List<Double> outputErrors = new ArrayList<>();
        for (int i = 0; i < outputLayer.size(); i++) {
            double error = output.get(i) - target.get(i);
            outputErrors.add(error);
        }

        // Calculate gradients and update weights for output layer (output -> hidden layer)
        for (int i = 0; i < outputLayer.size(); i++) {
            // Derivative of the activation function (using linear activation for regression or softmax)
            double outputDerivative = output.get(i) * (1 - output.get(i));  // Example for sigmoid; adjust if using softmax or other activation
            for (int j = 0; j < hiddenLayers.get(hiddenLayers.size() - 1).size(); j++) {
                // Gradient descent weight adjustment
                double deltaWeight = learningRate * outputErrors.get(i) * outputDerivative * hiddenLayers.get(hiddenLayers.size() - 1).get(j);
                outputWeights[i][j] -= deltaWeight;
            }
        }

        // Backpropagate through hidden layers (hidden -> hidden, or hidden -> input)
        List<Double> nextLayerErrors = outputErrors;  // Start with output layer errors
        for (int layerIndex = hiddenLayers.size() - 1; layerIndex >= 0; layerIndex--) {
            List<Double> currentLayerErrors = new ArrayList<>(Collections.nCopies(hiddenLayers.get(layerIndex).size(), 0.0));
            List<Double> currentLayer = hiddenLayers.get(layerIndex);

            // Calculate error for current hidden layer
            for (int i = 0; i < currentLayer.size(); i++) {
                double errorSum = 0.0;
                if (layerIndex == hiddenLayers.size() - 1) {
                    // Calculate error from output layer
                    for (int k = 0; k < outputLayer.size(); k++) {
                        errorSum += nextLayerErrors.get(k) * outputWeights[k][i];
                    }
                } else {
                    // Calculate error from next hidden layer
                    for (int k = 0; k < hiddenLayers.get(layerIndex + 1).size(); k++) {
                        errorSum += nextLayerErrors.get(k) * hiddenWeights[layerIndex + 1][k][i];
                    }
                }
                currentLayerErrors.set(i, errorSum);
            }

            // Update weights for the current hidden layer
            for (int i = 0; i < currentLayer.size(); i++) {
                // Derivative of the activation function (using sigmoid or other)
                double hiddenDerivative = currentLayer.get(i) * (1 - currentLayer.get(i));  // Adjust for your activation function
                for (int j = 0; j < (layerIndex == 0 ? inputLayer.size() : hiddenLayers.get(layerIndex - 1).size()); j++) {
                    double deltaWeight = learningRate * currentLayerErrors.get(i) * hiddenDerivative *
                            (layerIndex == 0 ? inputLayer.get(j) : hiddenLayers.get(layerIndex - 1).get(j));
                    hiddenWeights[layerIndex][i][j] -= deltaWeight;
                }
            }

            // Set next layer errors to the current hidden layer errors for backpropagating further
            nextLayerErrors = currentLayerErrors;
        }
    }

}
