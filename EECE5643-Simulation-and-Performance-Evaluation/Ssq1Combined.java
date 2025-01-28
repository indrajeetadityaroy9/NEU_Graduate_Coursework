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

public class Ssq1Combined {
    public static void main(String[] args) {
        try {
            // 1. Handle Ssq1.dat (arrival, service)
            handleSsq1Dat("Ssq1.dat");
            // 2. Handle ac.dat (arrival, departure)
            handleAcDat("ac.dat");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void handleSsq1Dat(String fileName) throws IOException {
        double START = 0.0;

        FileInputStream fis = null;
        try {
            fis = new FileInputStream(fileName);
        } catch (FileNotFoundException fnfe) {
            System.out.println("Cannot open input file " + fileName);
            System.exit(1);
        }

        BufferedReader in = new BufferedReader(new InputStreamReader(fis));

        String line;
        StringTokenizer st;
        long index       = 0;       // job index
        double arrival   = START;   // arrival time
        double delay;               // delay in queue
        double service;             // service time
        double wait;                // delay + service
        double departure = START;   // departure time
        double max_delay = 0.0;
        double delayed_job_count = 0;

        Ssq1Sum sum = new Ssq1Sum();
        sum.initSumParas();

        ArrayList<Double> arrivalTimes   = new ArrayList<>();
        ArrayList<Double> departureTimes = new ArrayList<>();

        while ((line = in.readLine()) != null) {
            index++;
            st = new StringTokenizer(line);
            arrival = Double.parseDouble(st.nextToken());
            
            if (arrival < departure)
                delay = departure - arrival;
            else
                delay = 0.0;
            
            if (delay > 0.0) {
                delayed_job_count++;
            }

            service = Double.parseDouble(st.nextToken());
            wait = delay + service;
            departure = arrival + wait;
            sum.delay   += delay;
            sum.wait    += wait;
            sum.service += service;

            max_delay = Math.max(max_delay, delay);
            arrivalTimes.add(arrival);
            departureTimes.add(departure);
        }
        sum.interarrival = arrival - START;

        Collections.sort(arrivalTimes);
        Collections.sort(departureTimes);

        double t = 400.0;
        int jobsBeforeT    = countJobs(arrivalTimes, t);
        int jobsAfterT = countJobs(departureTimes, t);
        int numJobsAtT = jobsBeforeT - jobsAfterT;

        DecimalFormat f = new DecimalFormat("###0.00");

        double delayedJobsRatio = (double) delayed_job_count / index;
        double utilization      = sum.service / departure;

        System.out.println("\n========== Results for file: " + fileName + " ==========");
        System.out.println("for " + index + " jobs");
        System.out.println("   average interarrival time =  " + f.format(sum.interarrival / index));
        System.out.println("   average service time .... =  " + f.format(sum.service / index));
        System.out.println("   average delay ........... =  " + f.format(sum.delay / index));
        System.out.println("   average wait ............ =  " + f.format(sum.wait / index));
        System.out.println("   maximum delay ........... =  " + f.format(max_delay));
        System.out.println("   number of jobs in service node at t=400 = " + numJobsAtT);
        System.out.println("   proportion of jobs delayed = " + f.format(delayedJobsRatio));
        System.out.println("   server utilization = " + f.format(utilization));

        fis.close();
    }

    public static void handleAcDat(String fileName) throws IOException {
        double START = 0.0;

        FileInputStream fis = null;
        try {
            fis = new FileInputStream(fileName);
        } catch (FileNotFoundException fnfe) {
            System.out.println("Cannot open input file " + fileName);
            System.exit(1);
        }

        BufferedReader in = new BufferedReader(new InputStreamReader(fis));

        String line;
        StringTokenizer st;
        long index       = 0;       // job index
        double arrival   = START;   // arrival time
        double delay;               // delay in queue
        double service;             // service time
        double wait;                // delay + service
        double departure = START;   // departure time

        Ssq1Sum sum = new Ssq1Sum();
        sum.initSumParas();

        while ((line = in.readLine()) != null) {
            index++;
            st = new StringTokenizer(line);
            arrival = Double.parseDouble(st.nextToken());
            
            if (arrival < departure)
                delay = departure - arrival;
            else
                delay = 0.0;

            double actualDeparture = Double.parseDouble(st.nextToken());
            service = actualDeparture - arrival - delay;
            wait = delay + service;
            sum.delay   += delay;
            sum.wait    += wait;
            sum.service += service;
            departure = actualDeparture;
        }
        
        sum.interarrival = arrival - START;

        DecimalFormat f = new DecimalFormat("###0.00");

        double avgServiceTime        = sum.service / index;
        double avgInterarrivalTime   = sum.interarrival / index;
        double utilization           = sum.service / departure;
        double trafficIntensity      = avgServiceTime / avgInterarrivalTime;

        System.out.println("\n========== Results for file: " + fileName + " ==========");
        System.out.println("for " + index + " jobs");
        System.out.println("   average interarrival time =  " + f.format(avgInterarrivalTime));
        System.out.println("   average service time .... =  " + f.format(avgServiceTime));
        System.out.println("   average delay ........... =  " + f.format(sum.delay / index));
        System.out.println("   average wait ............ =  " + f.format(sum.wait / index));
        System.out.println("   server utilization   = " + f.format(utilization));
        System.out.println("   traffic intensity    = " + f.format(trafficIntensity));

        fis.close();
    }

    private static int countJobs(ArrayList<Double> sortedList, double t) {
        int idx = Collections.binarySearch(sortedList, t);
        if (idx < 0) {
            idx = -idx - 1;
        } else {
            while (idx + 1 < sortedList.size() && sortedList.get(idx + 1) <= t) {
                idx++;
            }
        }
        return (idx >= 0) ? idx + 1 : 0;
    }
}
