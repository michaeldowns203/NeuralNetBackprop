import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class MinMaxScale {
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
}
