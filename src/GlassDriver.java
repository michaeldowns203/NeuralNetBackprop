import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

//10% cross validation for tuning
public class GlassDriver {
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
        List<List<Object>> removedInstances = extractTenPercent(dataset);

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
        String inputFile1 = "src/glass.data";
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
                labels.add(Integer.parseInt(rawData[10]));

                // Fill the data row (columns 2 to 10)
                for (int i = 1; i < rawData.length - 1; i++) {
                    row.add(Double.parseDouble(rawData[i]));
                }
                row.add(labels.get(lineNum)); // Add the label to the row
                dataset.add(row);
                lineNum++;
            }

            System.out.println(dataset.size());

            stdin.close();

            // Extract 10% of the dataset for testing
            List<List<Object>> testSet = extractTenPercent(dataset);

            // Split the remaining dataset into stratified chunks
            List<List<List<Object>>> chunks = splitIntoStratifiedChunks(dataset, 10);

            // Loss instance variables
            double totalAccuracy = 0;
            double totalPrecision = 0;
            double totalRecall = 0;
            double totalF1 = 0;
            double total01loss = 0;

            // Perform stratified 10-fold cross-validation
            for (int i = 0; i < 10; i++) {
                // Create training sets from chunks
                List<List<Double>> trainingData = new ArrayList<>();
                List<String> trainingLabels = new ArrayList<>();

                // Combine the other 9 chunks into the training set
                for (int j = 0; j < 10; j++) {
                    if (j != i) {
                        for (List<Object> row : chunks.get(j)) {
                            trainingLabels.add(String.valueOf(row.get(row.size() - 1)));  // Last column is label
                            List<Double> features = new ArrayList<>();
                            for (int k = 0; k < row.size() - 1; k++) {
                                features.add((Double) row.get(k));
                            }
                            trainingData.add(features);
                        }
                    }
                }

                // Initialize and train the k-NN model
                int k = 10; // You can tune this value later
                KNN knn = new KNN(k, 1, 1); // Bandwidth and error threshold are irrelevant
                knn.fit(trainingData, trainingLabels);
                //knn.edit();
                //knn.kMeansAndReduce(133, 1000);

                // Test the classifier using the test set
                int correctPredictions = 0;
                int truePositives = 0;
                int falsePositives = 0;
                int falseNegatives = 0;
                List<String> testLabels = new ArrayList<>();

                for (List<Object> row : testSet) {
                    testLabels.add(String.valueOf(row.get(row.size() - 1))); // Last column is label
                }

                for (int j = 0; j < testSet.size(); j++) {
                    List<Double> testInstance = new ArrayList<>();
                    for (int l = 0; l < testSet.get(j).size() - 1; l++) {
                        testInstance.add((Double) testSet.get(j).get(l));
                    }

                    String predicted = knn.predict(testInstance);
                    String actual = testLabels.get(j);

                    // Print the test data, predicted label, and actual label
                    System.out.print("Test Data: [ ");
                    for (Double feature : testInstance) {
                        System.out.print(feature + " ");
                    }
                    System.out.println("] Predicted: " + predicted + " Actual: " + actual);

                    if (predicted.equals(actual)) {
                        correctPredictions++;
                    }
                    // Get true positives, false positives, and false negatives
                    if (predicted.equals("1")) {
                        if (actual.equals("1")) {
                            truePositives++;
                        } else {
                            falsePositives++;
                        }
                    } else if (actual.equals("1")) {
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
                System.out.println("Precision for class 1 (hold-out fold " + (i + 1) + "): " + precision);
                System.out.println("Recall for class 1 (hold-out fold " + (i + 1) + "): " + recall);
                System.out.println("F1 Score for class 1 (hold-out fold " + (i + 1) + "): " + f1Score);
            }

            // Average accuracy across all 10 folds
            double averageAccuracy = totalAccuracy / 10;
            double average01loss = total01loss / 10;
            double averagePrecision = totalPrecision / 10;
            double averageRecall = totalRecall / 10;
            double averageF1 = totalF1 / 10;
            System.out.println("Average Accuracy: " + averageAccuracy);
            System.out.println("Average 0/1 Loss: " + average01loss);
            System.out.println("Average Precision for class 1: " + averagePrecision);
            System.out.println("Average Recall for class 1: " + averageRecall);
            System.out.println("Average F1 for class 1: " + averageF1);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}
