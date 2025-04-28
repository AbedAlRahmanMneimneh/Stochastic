# Bank Operational Dynamics Analysis Using Markov Chains

## Introduction

This project provides a comprehensive analysis of bank operational dynamics using stochastic modeling methods, specifically Markov Chains, Markov Processes, and Birth-Death Processes. The analysis helps understand customer queue behaviors and teller utilization patterns to optimize bank operations.

**BEFORE ANY ELABORATION PLEASE NOTE THE FOLLOWING IN ORDER TO PROPERLY VIEW AND UNDERSTAND THE REPORT:**

- **"bank-data.csv" :** This file contains the collected bank data.
- **"INE308 Tech Assignment - Spring 2025.pdf" :** This file contains the steps questions that needs anwering that were required by the instructor to tackle this problem.
- **"Project_Analysis.ipynb" :** This file is the notebook were the code was executed and explained and the output were elaborated on.
- **"Stochastic_Bank_Activity_Analysis.pdf" :** This file contains a printed out version of the  notebook that should be opened instead of th actual ipynb if you do not have the required environment or software to view or run it (recommended environments are: google collab, anaconda jupyter notebooks) 

### Purpose
The analysis aims to provide bank managers with data-driven insights to:
- Predict queue formations throughout the day
- Optimize staff scheduling based on customer flow patterns
- Improve customer experience by reducing waiting times
- Make informed decisions about teller allocation

### Data Description
This analysis uses hourly observations of bank queue states and teller activities collected over five business days during regular banking hours (8:00 AM to 4:00 PM). The data includes queue lengths, teller activity, and transaction times.

### Methodology Overview
The analysis employs three increasingly sophisticated stochastic modeling techniques:
1. **Discrete-time Markov Chains**: Analyzing hourly transition probabilities between queue states
2. **Continuous-time Markov Processes**: Modeling transition rates for more precise timing analysis
3. **Birth-Death Processes**: Specialized analysis of teller utilization and service efficiency

## Objective
To analyze the operational dynamics of a bank using Markov Chains and Birth-Death Processes, where the state of the system is observed hourly during business hours (8:00 AM to 4:00 PM).

---

## Part 1: Markov Chain Model

### Step 1: Data Collection and Transition Probability Matrix
**Approach:**
- Simulate data for five days (8 hours/day) for hourly observations of the queue states and teller activity.
- Define the states:
  - 0: No customers in queue.
  - 1: Few customers (1-3).
  - 2: Moderate queue (4-7 customers).
  - 3: Long queue (>7 customers) (special state).
  - 4: Idle period (no transactions, staff handling paperwork).
- Track the number of tellers actively serving customers and their transaction completion times.

**Generated Data:**
Simulated transition data for 5 days and the corresponding transition probability matrix.

---

### Step 2: Analysis Using the Markov Chain Model

1. **Probability Distribution:**
   - Compute the probability of transitioning from an empty queue to a long queue state within `t` hours (for t = 3, 4, 5, ..., 8).

2. **Probability of Long Queue at 12:00 PM:**
   - Use the transition probability matrix to calculate the probability of state 3 at 12:00 PM.

3. **Expected Number of Long Queues:**
   - Determine the expected number of times state 3 is observed during the day.

4. **Expected Time in Long Queue State:**
   - Compute the expected time spent in state 3.

5. **Long-Run Probabilities:**
   - Calculate the steady-state probabilities of each state.

6. **Expected Steps to Long Queue:**
   - Compute the expected number of steps to first reach state 3 starting from state 0.

7. **Expected Time to Return to Long Queue:**
   - Estimate the expected time to return to state 3 after leaving it.

---

## Part 2: Markov Process Model

### Step 1: Transition Rate Matrix
**Formula:**

$$
\lambda_{ij} = \frac{\text{Total number of transitions from state } i \text{ to state } j}{\text{Total duration (hours) spent in state } i}
$$
- Simulate the total transitions and duration in each state to construct the transition rate matrix.

---

### Step 2: Analysis Using the Markov Process Model

1. **Steady State Probabilities:**
   - Compute the steady-state probabilities for all states.

2. **Mean Sojourn Times:**
   - Calculate the mean time spent in each state before transitioning.

3. **Mean First Passage Time:**
   - Compute the mean time to first transition from state 3 to all other states.

---

## Part 3: Birth-Death Process

### Step 1: Transition Rate Matrix for Tellers
**Assumptions:**
- The number of states is equal to the maximum number of tellers (N).
- Service times follow an exponential distribution with an average service time.

**Generated Data:**
Simulated service times and customer arrivals to construct the transition rate matrix.

---

### Step 2: Analysis Using the Birth-Death Process Model

1. **Long-Run Probabilities:**
   - Assess teller utilization.

2. **Likelihood of All Tellers Busy:**
   - Compute the probability that all tellers are busy at 12:00 PM.

3. **Expected Time in Each State:**
   - Estimate how long the system stays in each state before transitioning.

4. **Expected Time to Full Occupancy:**
   - Calculate the expected time before all tellers are occupied for the first time.

5. **Impact of Increasing Tellers:**
   - Analyze how increasing the number of tellers affects service efficiency and congestion probabilities.

6. **Frequency of Full Capacity:**
   - Determine how often the system reaches full capacity during the working day.

## Appendix: Code and Simulated Data
Provide Python or R code for data simulation, matrix computation, and analysis.