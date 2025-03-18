import java.util.*;
import java.text.*;

public class ex_2 {
    public static void main(String[] args) {
        int n = 1000;       // number of boxes
        int balls = 10000;  // total number of balls
        long[] x = new long[n + 1];  // using indices 1 to n (index 0 is unused)

        // Initialize boxes: set x[1]..x[n] to 0
        for (int i = 1; i <= n; i++) {
            x[i] = 0;
        }
        
        Rng rng = new Rng();
        
        // Place 'balls' balls randomly into n boxes
        for (int j = 1; j <= balls; j++) {
            long i = equilikely(1, n, rng);
            x[(int)i]++;
        }
        
        // Feed the ball counts (as discrete sample values) to ddh.
        Ddh ddh = new Ddh();
        List<DdhNode> ddhList = new ArrayList<>();
        
        // For each box, insert its ball count into the ddh list.
        boolean first = true;
        for (int i = 1; i <= n; i++) {
            double data = (double) x[i];
            if (first) {
                DdhNode node = new DdhNode();
                node.init(data);
                ddhList.add(node);
                first = false;
            } else {
                ddh.insert(ddhList, data);
            }
        }
        
        // Sort the ddh list and traverse it to print the histogram.
        ddh.sort(ddhList);
        ddh.traverse(ddhList);
    }
    
    public static long equilikely(long a, long b, Rng r) {
        // Generate an equilikely random variate between a and b (inclusive)
        return a + (long) ((b - a + 1) * r.random());
    }
}
