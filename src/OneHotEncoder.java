import java.util.*;

public class OneHotEncoder {
    public static double[][] oneHotEncode(double[][] input) {
        // Get unique values and their indices for encoding
        Map<Double, Integer> valueToIndexMap = new HashMap<>();
        int index = 0;
        for (double[] row : input) {
            for (double value : row) {
                if (!valueToIndexMap.containsKey(value)) {
                    valueToIndexMap.put(value, index++);
                }
            }
        }

        int totalUniqueValues = valueToIndexMap.size();

        // Calculate the total number of elements in the output 2D array
        int totalRows = input.length;
        int totalColumns = input[0].length * totalUniqueValues;

        // Create the 2D array to hold the one-hot encoded data
        double[][] oneHotEncoded = new double[totalRows][totalColumns];

        // Fill the one-hot encoded array
        for (int i = 0; i < input.length; i++) {
            int columnOffset = 0;
            for (int j = 0; j < input[i].length; j++) {
                double value = input[i][j];
                int encodedIndex = valueToIndexMap.get(value);

                // Set the appropriate index to 1.0 in the output array
                oneHotEncoded[i][columnOffset + encodedIndex] = 1.0;
                columnOffset += totalUniqueValues;
            }
        }

        return oneHotEncoded;
    }

    // Method to print 2D array for verification
    public static void printOneHotEncodedArray(double[][] array) {
        for (double[] row : array) {
            System.out.println(Arrays.toString(row));
        }
    }
}

