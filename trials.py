import numpy as np
from scipy.linalg import expm
import numpy.linalg as la
import pandas as pd

bank_data = pd.read_csv("bank_data.csv")

def construct_birth_death_matrix(max_tellers, arrival_rate, service_rate):
    n_states = max_tellers + 1  # States 0 to max_tellers
    Q = np.zeros((n_states, n_states))
    
    # Calculate transaction rates based on the birth-death process
    for i in range(n_states):
        # Arrival rate (birth rate): state i to i+1
        if i < n_states - 1:
            Q[i, i+1] = arrival_rate
        
        # Service rate (death rate): state i to i-1
        if i > 0:
            # Total service rate is i*service_rate (i active tellers)
            Q[i, i-1] = i * service_rate
        
        # Diagonal elements: negative sum of row
        # Will be calculated after filling the off-diagonal elements
    
    # Set diagonal elements to ensure row sums to 0
    for i in range(n_states):
        Q[i, i] = -np.sum(Q[i, :])
    
    return Q

# Extract maximum number of tellers from the data
max_tellers = int(bank_data['Active_Tellers'].max())

# Calculate average service rate from the data (inverse of average transaction time)
# Convert minutes to hours for consistency
avg_service_time_hours = bank_data['Avg_Transaction_Time_Min'].mean() / 60  # convert to hours
service_rate = 1 / avg_service_time_hours if avg_service_time_hours > 0 else 1.0  # customers per hour per teller

# Estimate arrival rate from the data
# A simple estimation: average number of customers in the system times service rate
# This is based on Little's Law: L = λW
state_to_customers = {0: 0, 1: 2, 2: 5, 3: 10, 4: 0}  # average customers in each state
avg_customers = sum(state_to_customers[state] * (bank_data['State'] == state).mean() for state in range(5))
arrival_rate = avg_customers * service_rate / max_tellers  # estimated arrival rate

# Create the birth-death transition rate matrix
bd_matrix = construct_birth_death_matrix(max_tellers, arrival_rate, service_rate)

print(f"Birth-Death Process Parameters:")
print(f"- Maximum number of tellers: {max_tellers}")
print(f"- Average service time: {avg_service_time_hours:.4f} hours")
print(f"- Service rate per teller (μ): {service_rate:.4f} customers/hour")
print(f"- Arrival rate (λ): {arrival_rate:.4f} customers/hour")
print(f"\nBirth-Death Process Transition Rate Matrix:")
print(np.round(bd_matrix, 4))

# Calculate the traffic intensity (ρ = λ/μN)
traffic_intensity = arrival_rate / (service_rate * max_tellers)
print(f"\nTraffic intensity (ρ = λ/μN): {traffic_intensity:.4f}")

# 1. Steady‐state probabilities π solving πQ = 0, ∑π=1
def steady_state(Q):
    n = Q.shape[0]
    A = Q.T.copy()
    # replace one equation with sum π = 1
    A[-1, :] = 1
    b = np.zeros(n)
    b[-1] = 1
    return la.solve(A, b)

# 2. Transient prob. at time t (hours) starting from state 0
def transient_prob(Q, t, start_state=0):
    n = Q.shape[0]
    P0 = np.zeros(n)
    P0[start_state] = 1
    P_t = P0 @ expm(Q * t)
    return P_t

# 3. Mean holding time in each state i: 1 / (-Q[i,i])
def mean_holding_times(Q):
    return 1.0 / (-np.diag(Q))

# 4. Mean time to first hit state N (absorbing) from state 0
def mean_time_to_full(Q, full_state):
    N = full_state
    # we want h_i for i=0..N, with h_N = 0
    # for i < N: ∑_j q[i,j] (h_j - h_i) = -1
    # since ∑_j q[i,j]=0, this is ∑_j q[i,j] h_j = -1
    M = np.zeros((N, N))
    v = -np.ones(N)
    for i in range(N):
        # fill row i for unknowns h_0..h_{N-1}
        for j in range(N):
            M[i, j] = Q[i, j]
        # Q[i,i] is already included
    h_sub = la.solve(M, v)
    h = np.zeros(N+1)
    h[:N] = h_sub
    return h

# 5. Sensitivity: vary number of tellers and compare blocking prob & utilization
def analyze_varying_tellers(arrival_rate, service_rate, teller_range, t_hours=1.0):
    results = []
    for N in teller_range:
        # build Q for this N
        def construct_Q(N):
            Q = np.zeros((N+1, N+1))
            for i in range(N+1):
                if i < N:      Q[i, i+1] = arrival_rate
                if i > 0:      Q[i, i-1] = i * service_rate
            for i in range(N+1):
                Q[i,i] = -Q[i].sum()
            return Q
        Qn = construct_Q(N)
        π = steady_state(Qn)
        traffic_intensity = arrival_rate / (service_rate * N)
        blocking_prob = π[N]                 # prob all busy
        utilization = 1 - π[0]               # fraction time ≥1 teller busy
        results.append({
            'tellers': N,
            'ρ': traffic_intensity,
            'blocking': blocking_prob,
            'utilization': utilization
        })
    return results

# 6. Expected number of full‐capacity visits per day (steady‐state approx.)
#    rate of transitions N–1 → N is π[N-1] * λ; per day multiply by hours_per_day
def full_capacity_visits_per_day(π, arrival_rate, hours_per_day=8):
    return π[-2] * arrival_rate * hours_per_day

###################################
# —— Example usage with your data:
###################################

# (Assuming you already have max_tellers, arrival_rate, service_rate, bd_matrix = Q)

n = max_tellers
Q = bd_matrix

# 1. Long‐run probabilities
π = steady_state(Q)
print("1) Steady‐state π:", np.round(π,4))

# 2. P(all busy) at 12:00 PM
#    assume system opens at 9:00, so t = 3 hours
P3 = transient_prob(Q, t=3.0)
print("2) P(all busy at 12:00):", P3[n])

# 3. Mean time in each state
mht = mean_holding_times(Q)
print("3) Mean holding times (hrs):", np.round(mht,4))

# 4. Mean time to first full‐capacity
h = mean_time_to_full(Q, full_state=n)
print("4) E[time to full] from 0:", h[0])

# 5. Compare N = current–2 … current+2
teller_list = list(range(max(1,n-2), n+3))
sens = analyze_varying_tellers(arrival_rate, service_rate, teller_list)
print("5) Sensitivity analysis:")
for r in sens:
    print(f"   N={r['tellers']}: ρ={r['ρ']:.2f}, blocking={r['blocking']:.3f}, util={r['utilization']:.3f}")

# 6. Full‐capacity visits per 8hr day
visits = full_capacity_visits_per_day(π, arrival_rate, hours_per_day=8)
print(f"6) Expected full‐capacity visits/day: {visits:.2f}")
