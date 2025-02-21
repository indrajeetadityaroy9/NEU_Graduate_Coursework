import java.io.*;
import java.text.*;
import java.util.LinkedList;

class Ssq2Sum {
  double delay;
  double wait;
  double service;
  double interarrival;

  void initSumParas() {
    delay = 0.0;
    wait = 0.0;
    service = 0.0;
    interarrival = 0.0;
  }
}

public class ex_4a {

  static long LAST = 1000000;
  static double START = 0.0;
  static double sarrival = START;

  static class Ssq2 {
    private double sarrival = ex_4a.START;

    double getArrival(Rng r) {
      double interarrival = exponential(2.0, r);
      sarrival += interarrival;
      return sarrival;
    }

    double getService(Rng r) {
      return uniform(1.0, 2.0, r);
    }

    double exponential(double m, Rng r) {
      return -m * Math.log(1.0 - r.random());
    }

    double uniform(double a, double b, Rng r) {
      return a + (b - a) * r.random();
    }
  }

  public static void main(String[] args) {

    DecimalFormat f = new DecimalFormat("###0.00");
    
    for (int capacity = 1; capacity <= 6; capacity++) {
      long index = 0;              
      double arrival = START;      
      double delay;           
      double service;           
      double wait;            
      double departure = START;
      Ssq2Sum sum = new Ssq2Sum();
      sum.initSumParas();

      sarrival = START;
      
      Ssq2 s = new Ssq2();
      Rng r = new Rng();
      r.putSeed(123456789);
      
      LinkedList<Double> departureQueue = new LinkedList<Double>();
      
      long blocked = 0;
      double lastArrival = START;

      while (index < LAST) {
        index++;
        arrival = s.getArrival(r);
        if (index > 1){
          sum.interarrival += (arrival - lastArrival);
        }
        lastArrival = arrival;
        
        while (!departureQueue.isEmpty() && departureQueue.peek() <= arrival) {
          departureQueue.poll();
        }
        
        if (departureQueue.size() >= capacity) {
          blocked++;
          continue;
        }
        
        if (arrival < departure)
          delay = departure - arrival;
        else
          delay = 0.0;
        service = s.getService(r);
        wait = delay + service;
        departure = arrival + wait;
        departureQueue.add(departure);
        
        sum.delay += delay;
        sum.wait += wait;
        sum.service += service;
      }
      
      System.out.println("\nFor capacity " + capacity + " and " + index + " jobs:");
      System.out.println("   rejection probability ........ =   " + f.format((double)blocked/index));
      System.out.println("   average interarrival time ...... =   " + f.format(sum.interarrival / index));
      System.out.println("   average wait ................... =   " + f.format(sum.wait / index));
      System.out.println("   average delay .................. =   " + f.format(sum.delay / index));
      System.out.println("   average service time ........... =   " + f.format(sum.service / index));
      System.out.println("   average # in the node .......... =   " + f.format(sum.wait / departure));
      System.out.println("   average # in the queue ......... =   " + f.format(sum.delay / departure));
      System.out.println("   utilization .................... =   " + f.format(sum.service / departure));
    }
  }
}
