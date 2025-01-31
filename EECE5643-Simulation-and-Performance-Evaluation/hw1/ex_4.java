import java.util.Random;

public class ex_4 {

    public static double simulate(int numTrials, double rho, Random r) {
        int count = 0;

        for (int i = 0; i < numTrials; i++) {
            double theta1 = 2 * Math.PI * r.nextDouble();
            double theta2 = 2 * Math.PI * r.nextDouble();
            double diff = Math.abs(theta1 - theta2);
            double distance = 2 * rho * Math.sin(diff / 2.0);

            if (distance > rho) {
                count++;
            }
        }

        return (double) count / numTrials;
    }

    public static void main(String[] args) {
        int N = 1000000;
        double rho = 1.0;
        long[] seeds = {1L, 42L, 2023L, 40599L, 99999L};

        for (long seed : seeds) {
            Random rng = new Random(seed);
            double prob = simulate(N, rho, rng);
            System.out.printf("Seed=%d -> P(X+Y > rho) = %.5f%n", seed, prob);
        }
    }
}
