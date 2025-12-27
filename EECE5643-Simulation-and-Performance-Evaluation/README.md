# EECE5643 Simulation & Performance Evaluation

Graduate-level coursework on modeling, simulation, and performance analysis of complex computer systems. The course uses stochastic modeling and discrete-event simulation, with Monte Carlo methods and statistical inference to analyze system behavior. Topics include Markov models, queueing networks, analytical performance modeling, and performance bounds. Applications cover large-scale distributed systems such as data centers, cloud platforms, high-performance computing clusters, storage systems, and parallel data processing frameworks.

## Contents

- `assignment1-single-server-queue/` (report: `assignment_1.pdf`)
- `assignment2-rng-simulation/` (report: `assignment_2.pdf`)
- `assignment3-statistical-analysis/` (report: `assignment_3.pdf`)
- `problem_set_1.pdf`, `problem_set_2.pdf`

## Running (Java)

From an assignment directory:

```bash
javac *.java
java ex_1
```

Note: `assignment1-single-server-queue/ex_1.java` and `assignment1-single-server-queue/ex_2.java` expect to be run from `assignment1-single-server-queue/` since they open `Ssq1.dat` / `ac.dat` by relative path.
