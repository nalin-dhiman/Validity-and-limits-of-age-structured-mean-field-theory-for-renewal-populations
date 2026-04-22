import numpy as np
import matplotlib.pyplot as plt

def solve_m_transport():
    # Grid
    R_max = 65.0 # Max age
    dr = 0.001
    dt = 0.0001
    T = 0.5
    steps = int(T/dt)
    
    r_grid = np.arange(0, R_max+dr, dr)
    N_r = len(r_grid)
    
    # Adaptation Field m(r, t)
    # Init: 0
    m = np.zeros(N_r)
    
    # Params
    tau_a = 0.1
    kappa = 2.0
    
    # Input V(r). Assume Mean V(r) profile.
    # In subthreshold, V ~ u approx 0?
    # During spike, V resets?
    # V(r) for Finite Size neuron?
    # Mean V given age r.
    # For LIF, V(r) rises from V_reset to V_thresh.
    # V(r) = (mu - exp(-r/tau_m)*...)
    # Let's assume some profile.
    tau_m = 0.02
    mu = 1.0 # Supra-threshold
    V_reset = 0.0
    V_r = mu * (1 - np.exp(-r_grid/tau_m)) + V_reset * np.exp(-r_grid/tau_m)
    
    # Boundary Condition for m at r=0
    # m(0, t) = E[a | age=0]
    # When neuron spikes, age->0.
    # a -> a + kappa*V_thresh? No, a -> a + delta.
    # In this model: da/dt = -a/tau + kappa*V.
    # Spike doesn't jump 'a' unless discrete?
    # Equation: "da = (-a/tau + kappa*V)*dt".
    # There is no jump in 'a' at spike in this equation!
    # "da = ...dt".
    # Does 'a' jump?
    # Let's check `effective_neuron.py`.
    # Update: self.a += da.
    # Spikes: last_spike_time update.
    # NO jump in a.
    # So m(0) = E[a | spike].
    # Which is Integral(m(r) * rho(r) * q(r)) / Integral(rho(r)*q(r)).
    # i.e. Average adaptation of spiking neurons.
    
    m_rec = []
    
    for i in range(steps):
        # 1. Advection: m(r, t+dt) = m(r-dr*dt?, t)
        # Upwind: dm/dt + dm/dr = S
        # m_new[i] = m[i] - dt/dr * (m[i] - m[i-1]) + dt * S[i]
        
        courant = dt/dr
        S = (-m / tau_a + kappa * V_r)
        
        m_new = np.zeros_like(m)
        m_new[1:] = m[1:] - courant * (m[1:] - m[:-1]) + dt * S[1:]
        
        # Boundary m[0]
        # Need Flux of m at boundary.
        # This requires coupling with q(r).
        # Assume constant for test: m[0] = 1.0
        m_new[0] = np.mean(m) # Dummy
        
        m = m_new
        if i % 1000 == 0:
            m_rec.append(m.copy())
            
    # Plot
    plt.figure()
    for tr in m_rec:
        plt.plot(r_grid, tr)
    plt.title('Adaptation Field Evolution')
    plt.xlabel('Age')
    plt.ylabel('m(a)')
    plt.savefig('results/diagnostics/m_transport.png')
    print("Saved plot.")

if __name__ == "__main__":
    solve_m_transport()
