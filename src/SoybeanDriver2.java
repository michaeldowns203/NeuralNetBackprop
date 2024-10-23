import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

//normal 10 fold
public class SoybeanDriver2 {
    // Method to scale labels using Min-Max Scaling
    public static List<Double> minMaxScaleLabels(List<Double> labels) {
        // Scale the labels using Min-Max scaling
        double minLabel = Collections.min(labels);
        double maxLabel = Collections.max(labels);
        List<Double> scaledLabels = new ArrayList<>();
        for (double label : labels) {
            double scaledLabel = (label - minLabel) / (maxLabel - minLabel);
            scaledLabels.add(scaledLabel);
        }

        return scaledLabels;  // Return the scaled labels
    }


    // Method to scale features using Min-Max Scaling
    public static List<List<Double>> minMaxScale(List<List<Double>> data) {
        int numFeatures = data.get(0).size();  // Process all columns as features
        List<Double> minValues = new ArrayList<>(Collections.nCopies(numFeatures, Double.MAX_VALUE));
        List<Double> maxValues = new ArrayList<>(Collections.nCopies(numFeatures, Double.MIN_VALUE));

        // Find the min and max values for each feature
        for (List<Double> row : data) {
            for (int i = 0; i < numFeatures; i++) {
                double value = row.get(i);
                if (value < minValues.get(i)) minValues.set(i, value);
                if (value > maxValues.get(i)) maxValues.set(i, value);
            }
        }

        // Scale the dataset based on min and max values
        List<List<Double>> scaledData = new ArrayList<>();
        for (List<Double> row : data) {
            List<Double> scaledRow = new ArrayList<>();
            for (int i = 0; i < numFeatures; i++) {
                double value = row.get(i);
                double scaledValue = (value - minValues.get(i)) / (maxValues.get(i) - minValues.get(i));
                scaledRow.add(scaledValue);
            }
            scaledData.add(scaledRow);  // Only include scaled features
        }

        return scaledData;
    }
    public static List<List<Object>> extractTenPercent(List<List<Object>> dataset) {
        // Create a map to hold instances of each class
        Map<String, List<List<Object>>> classMap = new HashMap<>();

        // Populate the class map
        for (List<Object> row : dataset) {
            String label = row.get(row.size() - 1).toString();
            classMap.putIfAbsent(label, new ArrayList<>());
            classMap.get(label).add(row);
        }

        List<List<Object>> removedInstances = new ArrayList<>();

        // Extract 10% of instances while maintaining class proportions
        for (List<List<Object>> classInstances : classMap.values()) {
            Random random = new Random(123);
            Collections.shuffle(classInstances, random); // Shuffle instances within each class

            // Determine the number of instances to remove (10%)
            int numToRemove = (int) (classInstances.size() * 0.1);

            // Extract the instances and add them to the removed list
            removedInstances.addAll(classInstances.subList(0, numToRemove));

            // Retain the remaining instances in the class instances list
            classInstances.subList(0, numToRemove).clear(); // Remove the extracted instances
        }

        return removedInstances;
    }

    public static List<List<List<Object>>> splitIntoStratifiedChunks(List<List<Object>> dataset, int numChunks) {
        // Extract 10% of the dataset
        //List<List<Object>> removedInstances = extractTenPercent(dataset);

        // Create a map to hold instances of each class
        Map<String, List<List<Object>>> classMap = new HashMap<>();

        // Populate the class map with the remaining instances
        for (List<Object> row : dataset) {
            String label = row.get(row.size() - 1).toString();
            classMap.putIfAbsent(label, new ArrayList<>());
            classMap.get(label).add(row);
        }

        // Create chunks for stratified sampling
        List<List<List<Object>>> chunks = new ArrayList<>();
        for (int i = 0; i < numChunks; i++) {
            chunks.add(new ArrayList<>());
        }

        // Distribute remaining instances into chunks while maintaining class proportions
        for (List<List<Object>> classInstances : classMap.values()) {
            Random random = new Random(123);
            Collections.shuffle(classInstances, random); // Shuffle instances within each class

            // Calculate the chunk size for remaining instances
            int chunkSize = classInstances.size() / numChunks;

            // Distribute the remaining instances into chunks
            for (int i = 0; i < numChunks; i++) {
                int start = i * chunkSize;
                int end = (i == numChunks - 1) ? classInstances.size() : start + chunkSize;
                chunks.get(i).addAll(classInstances.subList(start, end));
            }
        }

        return chunks;
    }


    public static void main(String[] args) throws IOException {
        String inputFile1 = "src/soybean-small.data";
        try {
            FileInputStream fis = new FileInputStream(inputFile1);
            InputStreamReader isr = new InputStreamReader(fis);
            BufferedReader stdin = new BufferedReader(isr);

            // First, count the number of lines to determine the size of the lists
            int lineCount = 0;
            while (stdin.readLine() != null) {
                lineCount++;
            }
            // Reset the reader to the beginning of the file
            stdin.close();
            fis = new FileInputStream(inputFile1);
            isr = new InputStreamReader(fis);
            stdin = new BufferedReader(isr);

            // Initialize the lists
            List<List<Object>> dataset = new ArrayList<>();
            List<Object> labels = new ArrayList<>();

            String line;
            int lineNum = 0;

            // Read the file and fill the dataset
            while ((line = stdin.readLine()) != null) {
                String[] rawData = line.split(",");
                List<Object> row = new ArrayList<>();

                // Assign the label (last column)
                labels.add(rawData[35]);

                // Fill the data rows
                for (int i = 0; i < rawData.length - 1; i++) {
                    row.add(Double.parseDouble(rawData[i]));
                }
                row.add(labels.get(lineNum)); // Add the label to the row
                dataset.add(row);
                lineNum++;
            }

            stdin.close();

            // Extract 10% of the dataset for testing
            //List<List<Object>> testSet = extractTenPercent(dataset);

            // Split the remaining dataset into stratified chunks
            List<List<List<Object>>> chunks = splitIntoStratifiedChunks(dataset, 10);

// Initialize the NeuralNetwork
            int inputSize = 9;  // Number of input features (columns 2 to 10 in the dataset)
            int[] hiddenLayerSizes = {5, 3};  // Example: 2 hidden layers, with 5 and 3 neurons respectively
            int outputSize = 1;  // Single output for classification (e.g., binary or multi-class)
            String activationType = "softmax";  // Use softmax for classification
            double learningRate = 0.01;  // Learning rate for gradient descent
            boolean useMomentum = false;  // Disable momentum in this example
            double momentumCoefficient = 0.0;  // Momentum coefficient (irrelevant if useMomentum is false)

            NeuralNetwork neuralNet = new NeuralNetwork(inputSize, hiddenLayerSizes, outputSize, activationType, learningRate, useMomentum, momentumCoefficient);

            // Convert dataset to neural network input format
            double[][] trainInputs;
            double[][] trainLabels;
            double[][] testInputs;
            double[][] testLabels;

            // Perform stratified 10-fold cross-validation
            for (int i = 0; i < 10; i++) {
                // Prepare training data
                List<List<Double>> trainingData = new ArrayList<>();
                List<Integer> trainingLabels = new ArrayList<>();

                List<List<Object>> testSet = chunks.get(i);

                // Combine the other 9 chunks into the training set
                for (int j = 0; j < 10; j++) {
                    if (j != i) {
                        for (List<Object> row : chunks.get(j)) {
                            trainingLabels.add((Integer) row.get(row.size() - 1));  // Last column is label
                            List<Double> features = new ArrayList<>();
                            for (int k = 0; k < row.size() - 1; k++) {
                                features.add((Double) row.get(k));
                            }
                            trainingData.add(features);
                        }
                    }
                }

                // Convert training data and labels to arrays
                trainInputs = new double[trainingData.size()][inputSize];
                trainLabels = new double[trainingData.size()][1]; // Assuming single output per example

                for (int t = 0; t < trainingData.size(); t++) {
                    trainInputs[t] = trainingData.get(t).stream().mapToDouble(Double::doubleValue).toArray();
                    trainLabels[t][0] = trainingLabels.get(t);
                }

                // Train the neural network
                neuralNet.train(trainInputs, trainLabels, 1000);  // Train for 1000 epochs

                // Prepare test data for the current fold
                List<List<Double>> testData = new ArrayList<>();
                List<Integer> testLabelsList = new ArrayList<>();

                for (List<Object> row : testSet) {
                    testLabelsList.add((Integer) row.get(row.size() - 1)); // Last column is label
                    List<Double> features = new ArrayList<>();
                    for (int k = 0; k < row.size() - 1; k++) {
                        features.add((Double) row.get(k));
                    }
                    testData.add(features);
                }

                // Convert test data and labels to arrays
                testInputs = new double[testData.size()][inputSize];
                testLabels = new double[testLabelsList.size()][1];

                for (int t = 0; t < testData.size(); t++) {
                    testInputs[t] = testData.get(t).stream().mapToDouble(Double::doubleValue).toArray();
                    testLabels[t][0] = testLabelsList.get(t);
                }

                // Test the neural network
                int correctPredictions = 0;
                for (int t = 0; t < testInputs.length; t++) {
                    double[] prediction = neuralNet.forwardPass(testInputs[t]);

                    // Convert prediction (softmax might return probabilities) to label
                    int predictedLabel = (prediction[0] >= 0.5) ? 1 : 0; // Adjust this based on your actual classification

                    if (predictedLabel == (int) testLabels[t][0]) {
                        correctPredictions++;
                    }
                }

                // Calculate accuracy
                double accuracy = (double) correctPredictions / testSet.size();
                System.out.println("Fold " + (i + 1) + " Accuracy: " + accuracy);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


}
