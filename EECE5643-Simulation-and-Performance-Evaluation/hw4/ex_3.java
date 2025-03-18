import java.util.*;

public class ex_3 {
    public static void main(String[] args) {
        int replications = 100000;
        int testSize = 12;
        int totalQuestions = 120;
        int classIQuestions = 90;

        int[] questionTypes = new int[totalQuestions];
        for (int i = 0; i < classIQuestions; i++) {
            questionTypes[i] = 1;
        }
        for (int i = classIQuestions; i < totalQuestions; i++) {
            questionTypes[i] = 2;
        }

        Rng rng = new Rng();
        Ddh ddh = new Ddh();
        List<DdhNode> ddhList = new ArrayList<>();
        boolean first = true;

        for (int i = 0; i < replications; i++) {
            int[] selectedTypes = sampleWithoutReplacement(questionTypes, testSize, rng);
            int totalScore = 0;
            for (int type : selectedTypes) {
                int score = 0;
                double r = rng.random();
                if (type == 1) {
                    if (r < 0.6)
                        score = 4;
                    else if (r < 0.6 + 0.3)
                        score = 3;
                    else
                        score = 2;
                } else {
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

        ddh.sort(ddhList);
        ddh.traverse(ddhList);

        long passCount = 0;
        for (DdhNode node : ddhList) {
            if (node.value >= 36) {
                passCount += node.count;
            }
        }
        double passProbability = (double) passCount / replications;
        System.out.println("\nProbability of passing (score >= 36): " + String.format("%.3f", passProbability));
    }

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
