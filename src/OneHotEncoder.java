import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class OneHotEncoder {

    public static List<List<Double>> oneHotEncode(List<String> data, int[] categoricalColumns) {
        Map<Integer, List<String>> uniqueCategories = new HashMap<>();

        // First pass to gather unique categories
        for (String record : data) {
            String[] values = record.split(",");
            for (int col : categoricalColumns) {
                String category = values[col].trim();
                uniqueCategories.putIfAbsent(col, new ArrayList<>());
                if (!uniqueCategories.get(col).contains(category)) {
                    uniqueCategories.get(col).add(category);
                }
            }
        }

        List<List<Double>> encodedData = new ArrayList<>();

        // Second pass to create one-hot encoded records
        for (String record : data) {
            String[] values = record.split(",");
            List<Double> encodedRecord = new ArrayList<>();

            for (int col = 0; col < values.length; col++) {
                String value = values[col].trim();
                if (isInArray(categoricalColumns, col)) {
                    // One-hot encode categorical column
                    List<String> categories = uniqueCategories.get(col);
                    for (String category : categories) {
                        encodedRecord.add(category.equals(value) ? 1.0 : 0.0);
                    }
                } else {
                    // Add non-categorical value directly (assuming it's numeric)
                    encodedRecord.add(Double.parseDouble(value));
                }
            }
            encodedData.add(encodedRecord);
        }

        return encodedData;
    }

    private static boolean isInArray(int[] array, int value) {
        for (int i : array) {
            if (i == value) return true;
        }
        return false;
    }
}

