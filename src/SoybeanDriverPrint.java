import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

//test model - normal 10 fold
public class SoybeanDriverPrint {

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
                String label = rawData[35];
                String numLabel = label.replaceAll("[^0-9.]", "");
                labels.add(Double.parseDouble(numLabel));

                // Fill the data rows
                for (int i = 0; i < rawData.length - 1; i++) {
                    row.add(Double.parseDouble(rawData[i]));
                }
                row.add(labels.get(lineNum)); // Add the label to the row
                dataset.add(row);
                lineNum++;
            }

            stdin.close();

            // Split the remaining dataset into stratified chunks
            List<List<List<Object>>> chunks = TenFoldCrossValidation.splitIntoStratifiedChunksC(dataset, 10);

            // Loss instance variables
            double total01loss = 0;
            double totalACR = 0;

            for (int i = 0; i < 10; i++) {
                List<List<Object>> trainingSet = new ArrayList<>();
                List<List<Double>> trainingData = new ArrayList<>();
                List<List<Double>> trainingLabels = new ArrayList<>();
                List<Integer> predictedList = new ArrayList<>();
                List<Integer> actualList = new ArrayList<>();

                List<List<Object>> testSet = chunks.get(i);

                int correctPredictions = 0;

                for (int j = 0; j < 10; j++) {
                    if (j != i) {
                        for (List<Object> row : chunks.get(j)) {
                            List<Object> all = new ArrayList<>();
                            for (int k = 0; k < row.size(); k++) {
                                all.add((Double) row.get(k));
                            }
                            trainingSet.add(all);
                        }
                    }
                }

                List<List<Double>> scaledTrainingData = MinMaxScale.minMaxScale(trainingSet);
                List<List<Double>> scaledTestData = MinMaxScale.minMaxScale(testSet);

                // Loop through the scaledTrainingData to extract features and labels
                for (int j = 0; j < scaledTrainingData.size(); j++) {
                    if (j != i) { // If excluding a specific chunk (e.g., for cross-validation)
                        List<Double> row = scaledTrainingData.get(j);
                        List<Double> features = new ArrayList<>(row.subList(0, row.size() - 1)); // All but the last element
                        Double label = row.get(row.size() - 1); // The last element as the label

                        trainingData.add(features); // Add features to trainingData
                        trainingLabels.add(Collections.singletonList(label)); // Add label to trainingLabels
                    }
                }

                double[][] trainInputs = new double[trainingData.size()][];
                double[][] trainOutputs = new double[trainingLabels.size()][];

                for (int t = 0; t < trainingData.size(); t++) {
                    trainInputs[t] = trainingData.get(t).stream().mapToDouble(Double::doubleValue).toArray();
                    trainOutputs[t] = trainingLabels.get(t).stream().mapToDouble(Double::doubleValue).toArray();
                }

                double[][] trainOutputsOHE = OneHotEncoder.oneHotEncode(trainOutputs);

                double[][] testInputs = new double[scaledTestData.size()][];
                for (int t = 0; t < scaledTestData.size(); t++) {
                    testInputs[t] = scaledTestData.get(t).subList(0, scaledTestData.get(t).size() - 1)
                            .stream().mapToDouble(Double::doubleValue).toArray();
                }

                int inputSize = trainInputs[0].length;
                int[] hiddenLayerSizes = {};
                int outputSize = 4;
                String activationType = "softmax";
                double learningRate = 0.001;
                boolean useMomentum = false;
                double momentumCoefficient = 0.01;

                NeuralNetworkPrint neuralNet = new NeuralNetworkPrint(inputSize, hiddenLayerSizes, outputSize, activationType, learningRate, useMomentum, momentumCoefficient);

                int maxEpochs = 1000;
                double tolerance = 0.0001;
                neuralNet.train(trainInputs, trainOutputsOHE, tolerance, maxEpochs);

                neuralNet.printWeightsAndInputs();
                neuralNet.printActivations();

                for (int t = 0; t < testInputs.length; t++) {
                    double[] prediction = neuralNet.forwardPass(testInputs[t]);
                    double actual = scaledTestData.get(t).get(scaledTestData.get(t).size() - 1);
                    int actualClass = 0;

                    if (actual == 0.0)
                        actualClass = 1;
                    else if (actual < 0.4)
                        actualClass = 2;
                    else if (actual < 0.8)
                        actualClass = 3;
                    else
                        actualClass = 4;

                    double maxProb = prediction[0];
                    int maxIndex = 0;

                    for (int g = 1; g < prediction.length; g++) {
                        if (prediction[g] > maxProb) {
                            maxProb = prediction[g];
                            maxIndex = g;
                        }
                    }

                    if (maxIndex == 0)
                        predictedList.add(1);
                    else if (maxIndex == 1)
                        predictedList.add(2);
                    else if (maxIndex == 2)
                        predictedList.add(3);
                    else
                        predictedList.add(4);

                    actualList.add(actualClass);

                    System.out.printf("Test Instance: %s | Predicted: %d | Actual: %d%n",
                            Arrays.toString(testInputs[t]), predictedList.get(t), actualClass);


                    if (predictedList.get(t).equals(actualClass)) {
                        correctPredictions++;
                    }
                }
                // Calculate 0/1 loss
                double loss01 = 1.0 - (double) correctPredictions / testSet.size();
                total01loss += loss01;
                System.out.printf("Fold %d 0/1 loss: %.4f%n", i+1, loss01);

                double acrFold = neuralNet.getAvConvergenceRate();
                totalACR += acrFold;
            }

            double AACR = totalACR / 10;
            System.out.printf("Average Convergence Rate across all epochs across 10 folds: %.4f%n", AACR);

            double average01loss = total01loss / 10;
            System.out.printf("Average 0/1 Loss: %.4f%n", average01loss);
        }
        catch (IOException e) {
            e.printStackTrace();
        }
    }
}
