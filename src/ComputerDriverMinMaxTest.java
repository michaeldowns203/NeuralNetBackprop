import java.util.*;
import java.io.*;

public class ComputerDriverMinMaxTest {

    // Method to scale data and labels using Min-Max Scaling
    public static List<List<Double>> minMaxScale(List<List<Object>> dataWithLabels) {
        int numFeatures = dataWithLabels.get(0).size() - 1; // Last column is the label
        List<Double> minValues = new ArrayList<>(Collections.nCopies(numFeatures + 1, Double.MAX_VALUE));
        List<Double> maxValues = new ArrayList<>(Collections.nCopies(numFeatures + 1, Double.MIN_VALUE));

        // Find the min and max values for each feature and label
        for (List<Object> row : dataWithLabels) {
            for (int i = 0; i <= numFeatures; i++) {
                double value = (Double) row.get(i);
                if (value < minValues.get(i)) minValues.set(i, value);
                if (value > maxValues.get(i)) maxValues.set(i, value);
            }
        }

        // Scale the dataset based on min and max values
        List<List<Double>> scaledData = new ArrayList<>();
        for (List<Object> row : dataWithLabels) {
            List<Double> scaledRow = new ArrayList<>();
            for (int i = 0; i <= numFeatures; i++) {
                double value = (Double) row.get(i);
                double scaledValue;
                if (minValues.get(i).equals(maxValues.get(i))) {
                    scaledValue = 0.0;  // Avoid division by zero if min and max are the same
                } else {
                    scaledValue = (value - minValues.get(i)) / (maxValues.get(i) - minValues.get(i));
                }
                scaledRow.add(scaledValue);
            }
            scaledData.add(scaledRow);
        }

        return scaledData;
    }

    public static List<List<Object>> extractTenPercent(List<List<Object>> dataset) {
        dataset.sort(Comparator.comparingDouble(row -> (Double) row.get(row.size() - 1)));
        List<List<List<Object>>> groups = new ArrayList<>();
        int groupSize = 10;

        for (int i = 0; i < dataset.size(); i += groupSize) {
            int end = Math.min(i + groupSize, dataset.size());
            groups.add(new ArrayList<>(dataset.subList(i, end)));
        }

        List<List<Object>> tuningData = new ArrayList<>();
        int tuningSize = (int) Math.ceil(dataset.size() * 0.1);

        for (int fold = 0; fold < groupSize; fold++) {
            for (int i = fold; i < groups.size(); i += groupSize) {
                if (i < groups.size() && fold < groups.get(i).size()) {
                    tuningData.add(groups.get(i).get(fold));
                }
                if (tuningData.size() >= tuningSize) {
                    break;
                }
            }
            if (tuningData.size() >= tuningSize) {
                break;
            }
        }

        return tuningData.subList(0, Math.min(tuningData.size(), tuningSize));
    }

    public static List<List<List<Object>>> splitIntoStratifiedChunks(List<List<Object>> dataset, int numChunks) {
        dataset.sort(Comparator.comparingDouble(row -> (Double) row.get(row.size() - 1)));
        List<List<Object>> tuningData = extractTenPercent(dataset);
        List<List<Object>> remainingData = new ArrayList<>(dataset.subList(tuningData.size(), dataset.size()));

        List<List<List<Object>>> chunks = new ArrayList<>();
        for (int i = 0; i < numChunks; i++) {
            chunks.add(new ArrayList<>());
        }

        int groupSize = 10;
        List<List<Object>> group = new ArrayList<>();

        for (int i = 0; i < remainingData.size(); i++) {
            group.add(remainingData.get(i));
            if (group.size() == groupSize || i == remainingData.size() - 1) {
                for (int j = 0; j < group.size(); j++) {
                    int chunkIndex = j % numChunks;
                    chunks.get(chunkIndex).add(group.get(j));
                }
                group.clear();
            }
        }

        return chunks;
    }

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

            List<List<Object>> testSet = extractTenPercent(dataset);
            List<List<List<Object>>> chunks = splitIntoStratifiedChunks(dataset, 10);

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

                List<List<Double>> scaledTrainingData = minMaxScale(trainingSet);
                List<List<Double>> scaledTestData = minMaxScale(testSet);

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
                int[] hiddenLayerSizes = {6,4};
                int outputSize = 1;
                String activationType = "linear";
                double learningRate = 0.001;
                boolean useMomentum = false;
                double momentumCoefficient = 0.01;

                NeuralNetwork neuralNet = new NeuralNetwork(inputSize, hiddenLayerSizes, outputSize, activationType, learningRate, useMomentum, momentumCoefficient);

                int maxEpochs = 1000;
                neuralNet.train(trainInputs, trainOutputs, maxEpochs);

                for (int t = 0; t < testInputs.length; t++) {
                    double[] prediction = neuralNet.forwardPass(testInputs[t]);
                    double actual = scaledTestData.get(t).get(scaledTestData.get(t).size() - 1);

                    predictedList.add(Math.abs(prediction[0]));
                    actualList.add(actual);

                    System.out.printf("Test Instance: %s | Predicted: %.4f | Actual: %.4f%n",
                            Arrays.toString(testInputs[t]), Math.abs(prediction[0]), actual);
                }

                double mse = calculateMSE(actualList, predictedList);
                totalMSE += mse;
                System.out.println("Fold " + (i + 1) + " Mean Squared Error: " + mse);
            }

            double averageMSE = totalMSE / 10;
            System.out.println("Average Mean Squared Error across 10 folds: " + averageMSE);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static double calculateMSE(List<Double> actual, List<Double> predicted) {
        double sum = 0;
        for (int i = 0; i < actual.size(); i++) {
            double error = actual.get(i) - predicted.get(i);
            sum += error * error;
        }
        return sum / actual.size();
    }
}
