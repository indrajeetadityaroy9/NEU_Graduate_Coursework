import java.util.Random;

public class ex_3 {
    public static int equilikely(long a, long b, Random r) {
        return (int) (a + (long) ((b - a + 1) * r.nextDouble()));
    }

    private static final int[] distribution = {
        1,
        2, 2,
        3, 3,
        4, 4,
        5, 5,
        6, 6, 6, 6
    };

    public static int loadedDieFace(Random r) {
        long roll = equilikely(1, 13, r);
        return distribution[(int) roll - 1];
    }

    public static double simulate(int numTrials, Random r) {
        int count = 0;
        for (int i = 0; i < numTrials; i++) {
            int die1 = loadedDieFace(r);
            int die2 = loadedDieFace(r);
            if (die1 + die2 == 7) {
                count++;
            }
        }
        return (double) count / numTrials;
    }

    public static void main(String[] args) {
        int N = 1000000;
        long[] seeds = {1L, 42L, 2023L, 40599L, 99999L};

        for (long seed : seeds) {
            Random rng = new Random(seed);
            double probability = simulate(N, rng);
            System.out.printf("Seed=%d -> P(X+Y=7) = %.5f%n", seed, probability);
        }
    }
}
