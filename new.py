import numpy as np
import matplotlib.pyplot as plt

def sir_euler(beta, gamma, S0, I0, R0, h, t_max):
    """Euler's method for solving the SIR model."""
    N = S0 + I0 + R0  # Total population
    steps = int(t_max / h)
    t_values = np.linspace(0, t_max, steps)
    S, I, R = np.zeros(steps), np.zeros(steps), np.zeros(steps)
    S[0], I[0], R[0] = S0, I0, R0
    
    for i in range(1, steps):
        S[i] = S[i-1] - h * beta * S[i-1] * I[i-1] / N
        I[i] = I[i-1] + h * (beta * S[i-1] * I[i-1] / N - gamma * I[i-1])
        R[i] = R[i-1] + h * gamma * I[i-1]
    
    return t_values, S, I, R

def sir_rk4(beta, gamma, S0, I0, R0, h, t_max):
    """Runge-Kutta 4th order method for solving the SIR model."""
    N = S0 + I0 + R0  # Total population
    steps = int(t_max / h)
    t_values = np.linspace(0, t_max, steps)
    S, I, R = np.zeros(steps), np.zeros(steps), np.zeros(steps)
    S[0], I[0], R[0] = S0, I0, R0
    
    for i in range(1, steps):
        def dSdt(S, I): return -beta * S * I / N
        def dIdt(S, I): return beta * S * I / N - gamma * I
        def dRdt(I): return gamma * I
        
        k1_S, k1_I, k1_R = h * dSdt(S[i-1], I[i-1]), h * dIdt(S[i-1], I[i-1]), h * dRdt(I[i-1])
        k2_S, k2_I, k2_R = h * dSdt(S[i-1] + k1_S/2, I[i-1] + k1_I/2), h * dIdt(S[i-1] + k1_S/2, I[i-1] + k1_I/2), h * dRdt(I[i-1] + k1_I/2)
        k3_S, k3_I, k3_R = h * dSdt(S[i-1] + k2_S/2, I[i-1] + k2_I/2), h * dIdt(S[i-1] + k2_S/2, I[i-1] + k2_I/2), h * dRdt(I[i-1] + k2_I/2)
        k4_S, k4_I, k4_R = h * dSdt(S[i-1] + k3_S, I[i-1] + k3_I), h * dIdt(S[i-1] + k3_S, I[i-1] + k3_I), h * dRdt(I[i-1] + k3_I)
        
        S[i] = S[i-1] + (k1_S + 2*k2_S + 2*k3_S + k4_S) / 6
        I[i] = I[i-1] + (k1_I + 2*k2_I + 2*k3_I + k4_I) / 6
        R[i] = R[i-1] + (k1_R + 2*k2_R + 2*k3_R + k4_R) / 6
    
    return t_values, S, I, R

def plot_sir(t, S, I, R, title):
    """Plot SIR results."""
    plt.figure(figsize=(8, 5))
    plt.plot(t, S, label='Susceptible', color='blue')
    plt.plot(t, I, label='Infected', color='red')
    plt.plot(t, R, label='Recovered', color='green')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

# Parameters
beta_values = [0.1, 0.2, 0.3, 0.4, 0.5]
gamma_values = [0.05, 0.1, 0.15, 0.2, 0.25]
S0, I0, R0 = 990, 10, 0
h_values = [0.1, 0.5, 1.0]
t_max = 100

# Run simulations for different beta, gamma, and step sizes
data = {}
for beta in beta_values:
    for gamma in gamma_values:
        for h in h_values:
            key = (beta, gamma, h)
            t_euler, S_euler, I_euler, R_euler = sir_euler(beta, gamma, S0, I0, R0, h, t_max)
            t_rk4, S_rk4, I_rk4, R_rk4 = sir_rk4(beta, gamma, S0, I0, R0, h, t_max)
            data[key] = (t_euler, S_euler, I_euler, R_euler, t_rk4, S_rk4, I_rk4, R_rk4)
            plot_sir(t_euler, S_euler, I_euler, R_euler, f'Euler (β={beta}, γ={gamma}, h={h})')
            plot_sir(t_rk4, S_rk4, I_rk4, R_rk4, f'RK4 (β={beta}, γ={gamma}, h={h})')
