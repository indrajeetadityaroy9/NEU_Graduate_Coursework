import java.util.*;

public class ex_2 {
    public static void main(String[] args) {
        int n = 1000;
        int balls = 10000;
        long[] x = new long[n + 1];

        for (int i = 1; i <= n; i++) {
            x[i] = 0;
        }
        
        Rng rng = new Rng();
        
        for (int j = 1; j <= balls; j++) {
            long i = equilikely(1, n, rng);
            x[(int)i]++;
        }
        
        Ddh ddh = new Ddh();
        List<DdhNode> ddhList = new ArrayList<>();
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
        
        ddh.sort(ddhList);
        ddh.traverse(ddhList);
    }
    
    public static long equilikely(long a, long b, Rng r) {
        return a + (long) ((b - a + 1) * r.random());
    }
}
