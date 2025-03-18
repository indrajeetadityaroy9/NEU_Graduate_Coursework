import java.util.*;

public class ex_3 {
    public static void main(String[] args) {
        int replications = 100000;    // number of simulation replications
        int testSize = 12;            // number of questions per test
        int totalQuestions = 120;     // total questions in the bank
        int classIQuestions = 90;     // 90 class-I questions; hence 30 class-II

        // Build the question bank: 1 for Class I, 2 for Class II.
        int[] questionTypes = new int[totalQuestions];
        for (int i = 0; i < classIQuestions; i++) {
            questionTypes[i] = 1;
        }
        for (int i = classIQuestions; i < totalQuestions; i++) {
            questionTypes[i] = 2;
        }

        // Create a custom Rng instance.
        Rng rng = new Rng();

        // Create a Ddh instance and an empty list for the discrete-data histogram.
        Ddh ddh = new Ddh();
        List<DdhNode> ddhList = new ArrayList<>();
        boolean first = true;

        // Run the Monte Carlo simulation.
        for (int rep = 0; rep < replications; rep++) {
            // Sample 12 questions without replacement from the bank.
            int[] selectedTypes = sampleWithoutReplacement(questionTypes, testSize, rng);
            int totalScore = 0;
            // For each selected question, simulate the score.
            for (int type : selectedTypes) {
                int score = 0;
                double r = rng.random();
                if (type == 1) { // Class I question
                    // Grade distribution for Class I: A (4 pts, 0.6), B (3 pts, 0.3), C (2 pts, 0.1).
                    if (r < 0.6)
                        score = 4;
                    else if (r < 0.6 + 0.3)
                        score = 3;
                    else
                        score = 2;
                } else { // Class II question
                    // Grade distribution for Class II: B (3 pts, 0.1), C (2 pts, 0.4),
                    // D (1 pt, 0.4), F (0 pts, 0.1).
                    if (r < 0.1)
                        score = 3;
                    else if (r < 0.1 + 0.4)
                        score = 2;
                    else if (r < 0.1 + 0.4 + 0.4)
                        score = 1;
                    else
                        score = 0;
                }
                totalScore += score;
            }
            if (first) {
                DdhNode node = new DdhNode();
                node.init(totalScore);
                ddhList.add(node);
                first = false;
            } else {
                ddh.insert(ddhList, totalScore);
            }
        }

        // At this point, ddhList holds the aggregated discrete data (each test score).
        // Sort and traverse the ddh list to print the histogram and statistics.
        ddh.sort(ddhList);
        ddh.traverse(ddhList);

        //Calculate and output the probability of passing (score >= 36).
        long passCount = 0;
        for (DdhNode node : ddhList) {
            if (node.value >= 36) {
                passCount += node.count;
            }
        }
        double passProbability = (double) passCount / replications;
        System.out.println("\nProbability of passing (score >= 36): " + String.format("%.3f", passProbability));
    }

    // Helper method: sample without replacement
    public static int[] sampleWithoutReplacement(int[] array, int sampleSize, Rng rng) {
        final int n = array.length;
        int[] copy = Arrays.copyOf(array, n);

        for (int i = 0; i < sampleSize; i++) {
            int remaining = n - i;
            int j = i + (int) (rng.random() * remaining);
            int temp = copy[i];
            copy[i] = copy[j];
            copy[j] = temp;
        }
        
        if (sampleSize == n) {
            return copy;
        } else {
            int[] result = new int[sampleSize];
            System.arraycopy(copy, 0, result, 0, sampleSize);
            return result;
        }
    }
}
