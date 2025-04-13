package hw5;

public class mva {
    private static final int K = 7;
    private static final int N = 5;
    private static final int[] S = {20, 5, 15, 10, 10, 15, 20};
    private static final double[] V = {1, 1, 0.6, 0.4, 0.4, 0.3, 0.3};

    public static void main(String[] args) {
        MVAMetrics metrics = performMVA(N);
        printResults(metrics);
    }

    private static MVAMetrics performMVA(int maxCustomers) {
        double[][] responseTimes = new double[K][maxCustomers + 1];
        double[] throughput = new double[maxCustomers + 1];
        double[][] queueLengths = new double[K][maxCustomers + 1];

        for (int n = 1; n <= maxCustomers; n++) {
            double totalResponse = 0.0;
            for (int i = 0; i < K; i++) {
                responseTimes[i][n] = S[i] * (1 + queueLengths[i][n - 1]);
                totalResponse += V[i] * responseTimes[i][n];
            }
            throughput[n] = (double) n / totalResponse;
            for (int i = 0; i < K; i++) {
                queueLengths[i][n] = throughput[n] * V[i] * responseTimes[i][n];
            }
        }
        return new MVAMetrics(throughput, queueLengths);
    }

    private static void printResults(MVAMetrics metrics) {
        System.out.printf("System Throughput for N=%d: %.4f customers/sec%n",  N, metrics.throughput()[N]);
        for (int i = 0; i < K; i++) {
            System.out.printf("Device %d Queue Length for N=%d: %.4f customers%n", i, N, metrics.queueLengths()[i][N]);
        }
        System.out.println("\nThroughput per device:");
        for (int i = 0; i < K; i++) {
            double deviceThroughput = V[i] * metrics.throughput()[N];
            System.out.printf("Device %d Throughput: %.4f customers/sec%n", i, deviceThroughput);
        }
    }

    private record MVAMetrics(double[] throughput, double[][] queueLengths) {}
}
