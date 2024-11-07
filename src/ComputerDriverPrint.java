import java.util.*;
import java.io.*;

//test model - normal 10 fold
public class ComputerDriverPrint {

    public static void main(String[] args) throws IOException {
        String inputFile1 = "src/machine.data";
        try {
            FileInputStream fis = new FileInputStream(inputFile1);
            InputStreamReader isr = new InputStreamReader(fis);
            BufferedReader stdin = new BufferedReader(isr);

            int lineCount = 0;
            while (stdin.readLine() != null) {
                lineCount++;
            }

            stdin.close();
            fis = new FileInputStream(inputFile1);
            isr = new InputStreamReader(fis);
            stdin = new BufferedReader(isr);

            List<List<Object>> dataset = new ArrayList<>();
            String line;

            while ((line = stdin.readLine()) != null) {
                String[] rawData = line.split(",");
                List<Object> row = new ArrayList<>();
                for (int i = 2; i <= 8; i++) {
                    row.add(Double.parseDouble(rawData[i]));
                }
                dataset.add(row);
            }

            stdin.close();

            //List<List<Object>> testSet = extractTenPercent(dataset);
            List<List<List<Object>>> chunks = TenFoldCrossValidation.splitIntoStratifiedChunksR(dataset, 10);

            double totalMSE = 0;
            double totalACR = 0;

            for (int i = 0; i < 10; i++) {
                List<List<Object>> trainingSet = new ArrayList<>();
                List<List<Double>> trainingData = new ArrayList<>();
                List<List<Double>> trainingLabels = new ArrayList<>();
                List<Double> predictedList = new ArrayList<>();
                List<Double> actualList = new ArrayList<>();

                List<List<Object>> testSet = chunks.get(i);

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

                double[][] testInputs = new double[scaledTestData.size()][];
                for (int t = 0; t < scaledTestData.size(); t++) {
                    testInputs[t] = scaledTestData.get(t).subList(0, scaledTestData.get(t).size() - 1)
                            .stream().mapToDouble(Double::doubleValue).toArray();
                }

                int inputSize = trainInputs[0].length;
                int[] hiddenLayerSizes = {5,5};
                int outputSize = 1;
                String activationType = "linear";
                double learningRate = 0.001;
                boolean useMomentum = true;
                double momentumCoefficient = 0.5;

                NeuralNetworkPrint neuralNet = new NeuralNetworkPrint(inputSize, hiddenLayerSizes, outputSize, activationType, learningRate, useMomentum, momentumCoefficient);

                int maxEpochs = 1000;
                double tolerance = 0.0001;
                neuralNet.train(trainInputs, trainOutputs, tolerance, maxEpochs);

                neuralNet.printWeightsAndInputs();
                neuralNet.printActivations();

                for (int t = 0; t < testInputs.length; t++) {
                    double[] prediction = neuralNet.forwardPass(testInputs[t]);
                    double actual = scaledTestData.get(t).get(scaledTestData.get(t).size() - 1);

                    predictedList.add(prediction[0]);
                    actualList.add(actual);

                    System.out.printf("Test Instance: %s | Predicted: %.4f | Actual: %.4f%n",
                            Arrays.toString(testInputs[t]), prediction[0], actual);
                }
                double mse = LossFunctions.calculateMSE(actualList, predictedList);
                totalMSE += mse;
                System.out.printf("Fold %d Mean Squared Error: %.4f%n", i+1,  mse);

                double acrFold = neuralNet.getAvConvergenceRate();
                totalACR += acrFold;
            }

            double AACR = totalACR / 10;
            System.out.printf("Average Convergence Rate across all epochs across 10 folds: %.4f%n", AACR);

            double averageMSE = totalMSE / 10;
            System.out.printf("Average Mean Squared Error across 10 folds: %.4f%n", averageMSE);
        }
        catch (IOException e) {
            e.printStackTrace();
        }
    }
}
