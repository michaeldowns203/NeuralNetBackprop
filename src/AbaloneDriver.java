import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

//10% cross validation for tuning
public class AbaloneDriver {

    public static void main(String[] args) throws IOException {
        String inputFile1 = "src/abalone.data";
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
            //instance variable; flag to skip the first line
            boolean firstLine = true;
            int lineNum = 0;

            // Read the file and fill the dataset
            while ((line = stdin.readLine()) != null) {
                //skips the first line as it includes headers not data
                if (firstLine) {
                    firstLine = false;
                    continue;
                }
                String[] rawData = line.split(",");
                List<Object> row = new ArrayList<>();

                // Assign the label (last column)
                labels.add(Double.parseDouble(rawData[8]));

                // Fill the data row
                for (int i = 0; i < rawData.length - 2; i++) {
                    row.add(Double.parseDouble(rawData[i + 1]));
                }
                row.add(labels.get(lineNum)); // Add the label to the row
                dataset.add(row);
                lineNum++;
            }

            stdin.close();

            // Extract stratified tuning data (10%)
            List<List<Object>> testSet = TenFoldCrossValidation.extractTenPercentR(dataset);

            // Split the remaining dataset into stratified chunks
            List<List<List<Object>>> chunks = TenFoldCrossValidation.splitIntoStratifiedChunksR10(dataset, 10);

            // Loss instance variables
            double totalMSE = 0;

            for (int i = 0; i < 10; i++) {
                List<List<Object>> trainingSet = new ArrayList<>();
                List<List<Double>> trainingData = new ArrayList<>();
                List<List<Double>> trainingLabels = new ArrayList<>();
                List<Double> predictedList = new ArrayList<>();
                List<Double> actualList = new ArrayList<>();


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
                int[] hiddenLayerSizes = {4, 2};
                int outputSize = 1;
                String activationType = "linear";
                double learningRate = 0.01;
                boolean useMomentum = false;
                double momentumCoefficient = 0.01;

                NeuralNetwork neuralNet = new NeuralNetwork(inputSize, hiddenLayerSizes, outputSize, activationType, learningRate, useMomentum, momentumCoefficient);

                int maxEpochs = 100;
                neuralNet.train(trainInputs, trainOutputs, maxEpochs);

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
                System.out.println("Fold " + (i + 1) + " Mean Squared Error: " + mse);
            }

            double averageMSE = totalMSE / 10;
            System.out.println("Average Mean Squared Error across 10 folds: " + averageMSE);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}

