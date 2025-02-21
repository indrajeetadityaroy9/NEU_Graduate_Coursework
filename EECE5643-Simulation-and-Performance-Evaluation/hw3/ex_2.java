import java.io.*;
import java.lang.*;
import java.text.*;

class Ssq2Sum {                              
  double delay;       
  double wait;        
  double service;     
  double interarrival;

  void initSumParas() {
    delay = 0.0;
    wait  = 0.0;
    service = 0.0;
    interarrival = 0.0;
  }
}

public class ex_2 {

  static long   LAST  = 100000000;     
  static double START = 0.0;       
  static double sarrival = START;  

  public static void main(String[] args) {
    
    long   index     = 0;          
    double arrival   = START;      
    double delay;                  
    double service;                
    double wait;                   
    double departure = START;

    Ssq2Sum sum = new Ssq2Sum();
    sum.initSumParas();
      
    ex_2   s = new ex_2();        
    Rng    r = new Rng();         
    r.putSeed(123456789);         

    while (index < LAST) {
      index++;
      arrival = getArrival(r);

      if (arrival < departure)
        delay = departure - arrival;
      else
        delay = 0.0;

      service = s.getService(r);

      wait = delay + service;
      departure = arrival + wait;

      sum.delay   += delay;
      sum.wait    += wait;
      sum.service += service;
    }

    sum.interarrival = arrival - START;

    DecimalFormat f = new DecimalFormat("###0.00");

    System.out.println("\nfor " + index + " jobs");
    System.out.println("   average interarrival time =   " + f.format(sum.interarrival / index));
    System.out.println("   average wait ............ =   " + f.format(sum.wait / index));
    System.out.println("   average delay ........... =   " + f.format(sum.delay / index));
    System.out.println("   average service time .... =   " + f.format(sum.service / index));
    System.out.println("   average # in the node ... =   " + f.format(sum.wait / departure));
    System.out.println("   average # in the queue .. =   " + f.format(sum.delay / departure));
    System.out.println("   utilization ............. =   " + f.format(sum.service / departure));
  }

  static double exponential(double m, Rng r) {
    return -m * Math.log(1.0 - r.random());
  }

  static double getArrival(Rng r) {
    sarrival += exponential(2.0, r); 
    return sarrival;
  }

  double uniform(double a, double b, Rng r) {
    return a + (b - a) * r.random();
  }

  long geometric(double p, Rng r) {
    long count = 0;
    while (r.random() < p) {
      count++;
    }
    return count;
  }

  double getService(Rng r) {
    long tasks = 1 + geometric(0.9, r);
    double sum = 0.0;
    for (long k = 0; k < tasks; k++) {
      sum += uniform(0.1, 0.2, r);
    }
    return sum;
  }
}
