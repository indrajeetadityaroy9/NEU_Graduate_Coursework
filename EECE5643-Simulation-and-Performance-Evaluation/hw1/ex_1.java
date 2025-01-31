//Ex. 1.2.2

import java.io.*;
import java.util.*;
import java.text.*;

class Ssq1Sum {                                 /* sum of ...           */
  double delay;                                 /*   delay times        */
  double wait;                                  /*   wait times         */
  double service;                               /*   service times      */
  double interarrival;                          /*   interarrival times */

  void initSumParas() {
    delay = 0.0;
    wait = 0.0;
    service = 0.0;
    interarrival = 0.0;
  }
}

class ex_1 {
  static String FILENAME = "Ssq1.dat";          /* input data file */
  static double START = 0.0;

  public static void main(String[] args) throws IOException {
      
    FileInputStream fis = null;
    try {
      fis = new FileInputStream(FILENAME);
    } catch (FileNotFoundException fnfe) {
      System.out.println("Cannot open input file" + FILENAME);
      System.exit(1);
    }
      
    InputStreamReader r = new InputStreamReader(fis);
    BufferedReader in = new BufferedReader(r);
    try {
      String line = null;
      StringTokenizer st = null;
      long   index     = 0;                     /* job index            */
      double arrival   = START;                 /* arrival time         */
      double delay;                             /* delay in queue       */
      double service;                           /* service time         */
      double wait;                              /* delay + service      */
      double departure = START;                 /* departure time       */
      double max_delay = 0.0;
      double delayed_job_count = 0;
      Ssq1Sum sum = new Ssq1Sum();
      sum.initSumParas();

      ArrayList<Double> arrival_times   = new ArrayList<Double>();
      ArrayList<Double> departure_times = new ArrayList<Double>();

      while ( (line = in.readLine()) != null ) {
        index++;
        st = new StringTokenizer(line);
        arrival = Double.parseDouble(st.nextToken());
        if (arrival < departure)
          delay    = departure - arrival;       /* delay in queue    */
        else
          delay    = 0.0;                       /* no delay          */
        
        delayed_job_count += (delay > 0.0) ? 1 : 0;

        service = Double.parseDouble(st.nextToken());
        wait         = delay + service;
        departure    = arrival + wait;          /* time of departure */
        sum.delay   += delay;
        sum.wait    += wait;
        sum.service += service;

        max_delay = Math.max(max_delay, delay);

        arrival_times.add(arrival);
        departure_times.add(departure);
      }
      sum.interarrival = arrival - START;

      DecimalFormat f = new DecimalFormat("###0.00");

      Collections.sort(arrival_times);
      Collections.sort(departure_times);

      int jobs_before_t = count_jobs(arrival_times, 400);
      int jobs_after_t = count_jobs(departure_times, 400);
      int num_jobs_at_t = jobs_before_t - jobs_after_t;

      double delayed_jobs = delayed_job_count / index;
      double utilization = sum.service / departure;

      System.out.println("\nfor " + index + " jobs");
      System.out.println("   average interarrival time =  " + f.format(sum.interarrival / index));
      System.out.println("   average service time .... =  " + f.format(sum.service / index));
      System.out.println("   average delay ........... =  " + f.format(sum.delay / index));
      System.out.println("   average wait ............ =  " + f.format(sum.wait / index));
      System.out.println("   maximum delay ........... = " + f.format(max_delay));
      System.out.println("   number of jobs in the serivce node at t=400 = " + num_jobs_at_t);
      System.out.println("   proportion of jobs delayed = " + f.format(delayed_jobs));
      System.out.println("   server utilization = " + f.format(utilization));

    } catch (EOFException eofe) {
      System.out.println("Ssq1:" + eofe);
    }
    fis.close();
  }

  public static int count_jobs(ArrayList<Double> list, double t) {
    int idx = Collections.binarySearch(list, t);
    if (idx < 0) {
        idx = -idx - 1;
    } else {
        while ((idx + 1) < list.size() && list.get(idx+1) <= t) {
            idx++;
        }
    }
    return idx + 1;
  }

}
