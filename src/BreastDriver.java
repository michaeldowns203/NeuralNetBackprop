import java.util.*;
import java.io.*;

// 10% cross-validation for tuning
public class BreastDriver {

    public static void main(String[] args) throws IOException {
        String inputFile1 = "src/breast-cancer-wisconsin.data";
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
                labels.add(Double.parseDouble(rawData[10]));

                // Fill the data row (columns 2 to 10)
                for (int i = 1; i < rawData.length - 1; i++) {
                    if (rawData[i].equals("?")) {
                        row.add((Math.random() * 10) + 1); // Handle missing values
                    } else {
                        row.add(Double.parseDouble(rawData[i]));
                    }
                }
                row.add(labels.get(lineNum)); // Add the label to the row
                dataset.add(row);
                lineNum++;
            }

            stdin.close();

            // Extract 10% of the dataset for testing
            List<List<Object>> testSet = TenFoldCrossValidation.extractTenPercentC(dataset);

            // Split the remaining dataset into stratified chunks
            List<List<List<Object>>> chunks = TenFoldCrossValidation.splitIntoStratifiedChunksC10(dataset, 10);

            // Loss instance variables
            double totalAccuracy = 0;
            double totalPrecision = 0;
            double totalRecall = 0;
            double totalF1 = 0;
            double total01loss = 0;

            for (int i = 0; i < 10; i++) {
                List<List<Object>> trainingSet = new ArrayList<>();
                List<List<Double>> trainingData = new ArrayList<>();
                List<List<Double>> trainingLabels = new ArrayList<>();
                List<Double> predictedList = new ArrayList<>();
                List<Double> actualList = new ArrayList<>();

                int correctPredictions = 0;
                int truePositives = 0;
                int falsePositives = 0;
                int falseNegatives = 0;


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
                int[] hiddenLayerSizes = {6,4};
                int outputSize = 2;
                String activationType = "softmax";
                double learningRate = 0.001;
                boolean useMomentum = false;
                double momentumCoefficient = 0.1;

                NeuralNetwork neuralNet = new NeuralNetwork(inputSize, hiddenLayerSizes, outputSize, activationType, learningRate, useMomentum, momentumCoefficient);

                int maxEpochs = 100;
                neuralNet.train(trainInputs, trainOutputsOHE, maxEpochs);

                for (int t = 0; t < testInputs.length; t++) {
                    double[] prediction = neuralNet.forwardPass(testInputs[t]);
                    double actual = scaledTestData.get(t).get(scaledTestData.get(t).size() - 1);

                    double benignOdds = prediction[0];
                    if (benignOdds > 0.5)
                        predictedList.add(0.0);
                    else
                        predictedList.add(1.0);

                    actualList.add(actual);

                    System.out.printf("Test Instance: %s | Predicted: %.4f | Actual: %.4f%n",
                            Arrays.toString(testInputs[t]), predictedList.get(t), actual);


                if (predictedList.get(t) == (actual)) {
                    correctPredictions++;
                }

                // Get true positives, false positives, and false negatives
                if (predictedList.get(t) == 1.0) {
                    if (actual == 1.0) {
                        truePositives++;
                    } else {
                        falsePositives++;
                    }
                } else if (actual == 1.0) {
                    falseNegatives++;
                }
            }

            // Calculate precision and recall
            double precision = truePositives / (double) (truePositives + falsePositives);
            double recall = truePositives / (double) (truePositives + falseNegatives);
            totalPrecision += precision;
            totalRecall += recall;

            double f1Score = 2 * (precision * recall) / (precision + recall);
            totalF1 += f1Score;

            // Calculate accuracy for this fold
            double accuracy = (double) correctPredictions / testSet.size();
            totalAccuracy += accuracy;

            // Calculate 0/1 loss
            double loss01 = 1.0 - (double) correctPredictions / testSet.size();
            total01loss += loss01;

            // Print loss info
            System.out.println("Number of correct predictions: " + correctPredictions);
            System.out.println("Number of test instances: " + testSet.size());
            System.out.println("Fold " + (i + 1) + " Accuracy: " + accuracy);
            System.out.println("Fold " + (i + 1) + " 0/1 loss: " + loss01);
            System.out.println("Precision for class Malignant (4) (hold-out fold " + (i + 1) + "): " + precision);
            System.out.println("Recall for class Malignant (4) (hold-out fold " + (i + 1) + "): " + recall);
            System.out.println("F1 Score for class Malignant (4) (hold-out fold " + (i + 1) + "): " + f1Score);
        }

        // Average accuracy across all 10 folds
        double averageAccuracy = totalAccuracy / 10;
        double average01loss = total01loss / 10;
        double averagePrecision = totalPrecision / 10;
        double averageRecall = totalRecall / 10;
        double averageF1 = totalF1 / 10;
        System.out.println("Average Accuracy: " + averageAccuracy);
        System.out.println("Average 0/1 Loss: " + average01loss);
        System.out.println("Average Precision for class Malignant (4): " + averagePrecision);
        System.out.println("Average Recall for class Malignant (4): " + averageRecall);
        System.out.println("Average F1 for class Malignant (4): " + averageF1);

    } catch (IOException e) {
        e.printStackTrace();
    }
}

}
