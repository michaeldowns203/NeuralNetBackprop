import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class NeuralNetworkPrint {
    private int inputSize;
    private int[] hiddenLayerSizes;
    private int outputSize;
    private String activationType;
    private double learningRate;
    private boolean useMomentum;
    private double momentumCoefficient;

    private List<double[][]> weights;
    private List<double[]> biases;
    private List<double[][]> deltaWeights;

    public NeuralNetworkPrint(int inputSize, int[] hiddenLayerSizes, int outputSize, String activationType,
                         double learningRate, boolean useMomentum, double momentumCoefficient) {
        this.inputSize = inputSize;
        this.hiddenLayerSizes = hiddenLayerSizes;
        this.outputSize = outputSize;
        this.activationType = activationType;
        this.learningRate = learningRate;
        this.useMomentum = useMomentum;
        this.momentumCoefficient = momentumCoefficient;

        initializeWeights();
    }

    private void initializeWeights() {
        Random rand = new Random();
        weights = new ArrayList<>();
        biases = new ArrayList<>();
        deltaWeights = new ArrayList<>();

        if (hiddenLayerSizes.length == 0) {
            // No hidden layers, just connect input to output
            weights.add(new double[inputSize][outputSize]);
            biases.add(new double[outputSize]);
            deltaWeights.add(new double[inputSize][outputSize]);
        } else {
            // Same as before: Input to first hidden layer, hidden layers, and output
            weights.add(new double[inputSize][hiddenLayerSizes[0]]);
            biases.add(new double[hiddenLayerSizes[0]]);
            deltaWeights.add(new double[inputSize][hiddenLayerSizes[0]]);

            for (int i = 1; i < hiddenLayerSizes.length; i++) {
                weights.add(new double[hiddenLayerSizes[i - 1]][hiddenLayerSizes[i]]);
                biases.add(new double[hiddenLayerSizes[i]]);
                deltaWeights.add(new double[hiddenLayerSizes[i - 1]][hiddenLayerSizes[i]]);
            }

            weights.add(new double[hiddenLayerSizes[hiddenLayerSizes.length - 1]][outputSize]);
            biases.add(new double[outputSize]);
            deltaWeights.add(new double[hiddenLayerSizes[hiddenLayerSizes.length - 1]][outputSize]);
        }

        //xaiver initialization
        for (double[][] layerWeights : weights) {
            for (int i = 0; i < layerWeights.length; i++) {
                for (int j = 0; j < layerWeights[i].length; j++) {
                    layerWeights[i][j] = rand.nextGaussian() * Math.sqrt(2.0 / (layerWeights.length + layerWeights[0].length));

                }
            }
        }
        // Random initialization for biases
        for (int i = 0; i < biases.size(); i++) {
            for (int j = 0; j < biases.get(i).length; j++) {
                biases.get(i)[j] = rand.nextGaussian() * 0.01;  // Small random values for bias
            }
        }

    }

    // Sigmoid activation function for hidden layers
    private double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    // Softmax activation function for classification
    private double[] softmax(double[] z) {
        // Find the maximum value in z to subtract for numerical stability
        double max = Arrays.stream(z).max().orElse(0.0);

        // Calculate the sum of the exponentials after adjusting for numerical stability
        double sum = 0.0;
        double[] output = new double[z.length];
        for (int i = 0; i < z.length; i++) {
            output[i] = Math.exp(z[i] - max); // Subtract max for numerical stability
            sum += output[i];
        }

        // Normalize the outputs to get probabilities
        for (int i = 0; i < z.length; i++) {
            output[i] /= sum;
        }

        return output;
    }


    public double[] forwardPass(double[] input) {
        // Clear layerOutputs at the beginning of every forward pass
        layerOutputs.clear();

        // Set inputLayer if it's not already set
        if (inputLayer == null) {
            setInputLayer(input);
        }

        double[] currentOutput = input;

        // Store the input layer in layerOutputs
        storeLayerOutput(0, currentOutput);

        if (hiddenLayerSizes.length == 0) {
            // Directly go from input to output if no hidden layers
            double[] finalOutput = new double[outputSize];
            double[] z = new double[outputSize];

            for (int j = 0; j < outputSize; j++) {
                z[j] = 0.0;
                for (int k = 0; k < inputSize; k++) {
                    z[j] += input[k] * weights.get(0)[k][j]; // Only one weight matrix
                }
                z[j] += biases.get(0)[j];  // Only one bias set
            }

            // Apply activation
            if (activationType.equals("softmax")) {
                finalOutput = softmax(z);
            } else {
                finalOutput = z;  // For linear activation or regression
            }

            return finalOutput;
        }
        else {

            // Loop through hidden layers
            for (int i = 0; i < weights.size() - 1; i++) {
                int previousLayerSize = currentOutput.length;
                int currentLayerSize = weights.get(i)[0].length; // Number of neurons in the current layer

                double[] newOutput = new double[currentLayerSize]; // New output size

                // For each neuron in the current layer
                for (int j = 0; j < newOutput.length; j++) {
                    double z = 0.0;

                    // Sum up the weighted inputs from the previous layer's outputs (currentOutput)
                    for (int k = 0; k < currentOutput.length; k++) {
                        z += currentOutput[k] * weights.get(i)[k][j]; // Use k for previous layer, j for current layer
                    }

                    z += biases.get(i)[j]; // Add the bias term for neuron j in layer i
                    newOutput[j] = sigmoid(z); // Apply sigmoid activation
                }

                // Store the output of the current hidden layer
                storeLayerOutput(i + 1, newOutput);

                currentOutput = newOutput; // Update currentOutput to the new layer's output
            }
        }

        // Output layer activation (softmax or linear)
        double[] finalOutput = new double[outputSize];
        double[] z = new double[outputSize];

        // For the output layer, compute the weighted sum
        for (int j = 0; j < outputSize; j++) {
            z[j] = 0.0;
            for (int k = 0; k < currentOutput.length; k++) {
                z[j] += currentOutput[k] * weights.get(weights.size() - 1)[k][j]; // Last layer's weights
            }
            z[j] += biases.get(biases.size() - 1)[j];
        }

        // Apply softmax to the entire output vector if softmax is the activation type
        if (activationType.equals("softmax")) {
            finalOutput = softmax(z);
        } else {
            finalOutput = z; // For linear activation
        }

        // Store the output of the output layer
        storeLayerOutput(weights.size(), finalOutput);

        return finalOutput;
    }

    public void backPropagation(double[] actualOutput, double[] predictedOutput) {
        double[] error = new double[predictedOutput.length];

        // Error calculation for the output layer
        if (activationType.equals("softmax")) {
            // Cross-Entropy Loss for Classification
            for (int i = 0; i < actualOutput.length; i++) {
                error[i] = predictedOutput[i] - actualOutput[i];
            }
        } else {
            // Mean Squared Error for Regression
            for (int i = 0; i < predictedOutput.length; i++) {
                error[i] = 2 * (predictedOutput[i] - actualOutput[i]);
            }
        }

        /*
        // Print gradient information for the output layer
        System.out.println("Gradient at the Output Layer:");
        for (int i = 0; i < error.length; i++) {
            System.out.printf("Output Neuron %d: %.4f\n", i + 1, error[i]);
        }
        */

        // No hidden layers, update directly
        if (hiddenLayerSizes.length == 0) {
            updateWeights(0, error);
        } else {
            // Update weights for the output layer
            int lastLayerIdx = weights.size() - 1;
            double[] delta = error.clone();

            // Update weights for the output layer
            updateWeights(lastLayerIdx, delta);

            // Backpropagate through hidden layers
            for (int layerIdx = lastLayerIdx - 1; layerIdx >= 0; layerIdx--) {
                double[] newDelta = new double[weights.get(layerIdx)[0].length];
                double[] currentOutput = layerOutputs.get(layerIdx + 1); // Get the current output from the forward pass

                for (int i = 0; i < weights.get(layerIdx)[0].length; i++) {
                    double gradient = 0;
                    for (int j = 0; j < delta.length; j++) {
                        gradient += weights.get(layerIdx + 1)[i][j] * delta[j];
                    }

                    // Calculate the sigmoid derivative using the stored output
                    double sigmoidDerivative = currentOutput[i] * (1 - currentOutput[i]);
                    newDelta[i] = gradient * sigmoidDerivative;
                }

                delta = newDelta;
                updateWeights(layerIdx, delta);
            }
        }
    }

    private void updateWeights(int layerIdx, double[] delta) {
        double[][] weightMatrix = weights.get(layerIdx);
        double[] inputLayerForWeights = (layerIdx == 0) ? inputLayer : outputLayer(layerIdx - 1);

        //System.out.println("Weight updates for Layer " + (layerIdx + 1) + ":");

        for (int i = 0; i < weightMatrix.length; i++) {
            for (int j = 0; j < weightMatrix[i].length; j++) {
                double gradient = delta[j] * inputLayerForWeights[i];
                double deltaW = -learningRate * gradient;

                // Apply momentum if enabled
                if (useMomentum) {
                    deltaW += momentumCoefficient * deltaWeights.get(layerIdx)[i][j];
                    deltaWeights.get(layerIdx)[i][j] = deltaW;
                }

                // Print the weight update details
                //System.out.printf("Weight[%d][%d] before update: %.4f, Gradient: %8.4f, Update: %8.4f\n",
                //        i, j, weightMatrix[i][j], gradient, deltaW);

                // Update the weight
                weightMatrix[i][j] += deltaW;

                // Print the updated weight
                //System.out.printf("Weight[%d][%d] after update: %.4f\n", i, j, weightMatrix[i][j]);
            }

            // Update biases
            for (int j = 0; j < biases.get(layerIdx).length; j++) {
                double biasUpdate = -learningRate * delta[j];
                //System.out.printf("Bias[%d] before update: %8.4f, Update: %.4f\n", j, biases.get(layerIdx)[j], biasUpdate);
                biases.get(layerIdx)[j] += biasUpdate;
                //System.out.printf("Bias[%d] after update: %.4f\n", j, biases.get(layerIdx)[j]);
            }
        }

        System.out.println();
    }

    // Sigmoid derivative for backpropagation
    private double sigmoidDerivative(double z) {
        return sigmoid(z) * (1 - sigmoid(z));
    }

    // This will store the input layer values when the network starts processing
    private double[] inputLayer;

    // Store the output of each layer after forward pass
    private List<double[]> layerOutputs = new ArrayList<>();

    // Helper to retrieve the input to the network
    private double[] getInputLayer() {
        return inputLayer;
    }

    // Helper to store input values to the network before the forward pass
    public void setInputLayer(double[] inputLayer) {
        this.inputLayer = inputLayer;
    }

    // Helper to get the output of the current layer
    private double[] outputLayer(int layerIdx) {
        return layerOutputs.get(layerIdx);
    }

    // Store the output of each layer after forward pass
    private void storeLayerOutput(int layerIdx, double[] output) {
        if (layerOutputs.size() > layerIdx) {
            layerOutputs.set(layerIdx, output);  // Update output if it already exists
        } else {
            layerOutputs.add(output);  // Add new output if it doesn't exist
        }
    }


    public void train (double[][] inputData, double[][] targetData, int maxEpochs){
        for (int epoch = 0; epoch < maxEpochs; epoch++) {
            for (int i = 0; i < inputData.length; i++) {
                double[] input = inputData[i];
                double[] target = targetData[i];

                double[] predictedOutput = forwardPass(input);
                backPropagation(target, predictedOutput);
            }
        }
    }

    private void clipGradients(double[] gradients, double clipValue) {
        for (int i = 0; i < gradients.length; i++) {
            if (gradients[i] > clipValue) {
                gradients[i] = clipValue;
            } else if (gradients[i] < -clipValue) {
                gradients[i] = -clipValue;
            }
        }
    }

    public void printWeightsAndInputs() {
        for (int layerIdx = 0; layerIdx < weights.size(); layerIdx++) {
            double[][] weightMatrix = weights.get(layerIdx);

            System.out.println("Layer " + (layerIdx + 1) + " (Input size: " + weightMatrix.length + ", Output size: " + weightMatrix[0].length + "):");

            // Print inputs to the current layer
            if (layerIdx == 0) {
                System.out.print("Inputs to Layer 1: ");
                double[] inputToFirstLayer = getInputLayer();
                for (double input : inputToFirstLayer) {
                    System.out.printf("%8.4f ", input);
                }
                System.out.println();
                System.out.println("Outputs for:");
            } else {
                System.out.print("Inputs to Layer " + (layerIdx + 1) + ": ");
                double[] inputToLayer = layerOutputs.get(layerIdx);
                for (double input : inputToLayer) {
                    System.out.printf("%8.4f ", input);
                }
                System.out.println();
                System.out.println("Outputs for:");
            }

            // Print weights for the current layer
            for (int i = 0; i < weightMatrix.length; i++) {
                System.out.print("Input " + (i + 1) + ": ");
                for (int j = 0; j < weightMatrix[i].length; j++) {
                    System.out.printf("%8.4f ", weightMatrix[i][j]);
                }
                System.out.println();
            }

            System.out.println();
        }
    }

    public void printActivations() {
        for (int layerIdx = 0; layerIdx < layerOutputs.size(); layerIdx++) {
            double[] activations = layerOutputs.get(layerIdx);
            System.out.println("Activations at Layer " + (layerIdx + 1) + ":");
            for (int i = 0; i < activations.length; i++) {
                System.out.printf("Neuron %d: %.4f%n", i + 1, activations[i]);
            }
            System.out.println();
        }
        System.out.println();
    }



}
