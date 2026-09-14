import java.text.DecimalFormat;

class Outliers {
    long lo;
    long hi;
}

public class ex_4 {
    final static int NUM = 10000;
    static double MIN = 0.0;
    static double MAX = 2.0;
    static int K = 20;
    static double DELTA = (MAX - MIN) / K;

    public static void main(String[] args) {
        double x;
        int j;
        long index = 0;
        long[] count = new long[K];
        double[] midpoint = new double[K];
        double sum = 0.0, sumsqr = 0.0, mean, stdev;
        Outliers o = new Outliers();
        o.lo = 0;
        o.hi = 0;

        Rng r = new Rng();

        for (j = 0; j < K; j++) {
            count[j] = 0;
            midpoint[j] = MIN + (j + 0.5) * DELTA;
        }

        for (int i = 0; i < NUM; i++) {
			x = r.random() + r.random();
			if ((x >= MIN) && (x < MAX)) {
				j = (int)((x - MIN) / DELTA);
				count[j]++;
			} else if (x < MIN) {
				o.lo++;
			} else {
				o.hi++;
			}
		}
		index = NUM;

        for (j = 0; j < K; j++) {
            sum += midpoint[j] * count[j];
        }
        mean = sum / index;

        for (j = 0; j < K; j++) {
            sumsqr += Math.pow(midpoint[j] - mean, 2) * count[j];
        }
        stdev = Math.sqrt(sumsqr / index);

        DecimalFormat f = new DecimalFormat("###0.000");
        System.out.println("bin\tmidpoint\tcount\tproportion\tdensity\n");
        for (j = 0; j < K; j++) {
            System.out.print((j + 1) + "\t");
            System.out.print(f.format(midpoint[j]) + "\t\t");
            System.out.print(count[j] + "\t");
            System.out.print(f.format((double) count[j] / index) + "\t\t");
            System.out.println(f.format((double) count[j] / (index * DELTA)));
        }
        System.out.println("\nsample size ........... = " + index);
        System.out.println("mean ........... = " + f.format(mean));
        System.out.println("stdev .......... = " + f.format(stdev) + "\n");

        if (o.lo > 0)
            System.out.println("NOTE: there were " + o.lo + " low outliers");
        if (o.hi > 0)
            System.out.println("NOTE: there were " + o.hi + " high outliers");
    }
}
