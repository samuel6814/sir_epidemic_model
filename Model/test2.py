import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# Helper function to compute derivatives for the SIR model
def derivatives(state, t, beta, gamma, N):
    """
    Compute the derivatives for the SIR model
    
    Parameters:
    state (array): Current state [S, I, R]
    t (float): Current time (not used but included for compatibility)
    beta (float): Infection rate
    gamma (float): Recovery rate
    N (float): Total population
    
    Returns:
    array: Derivatives [dS/dt, dI/dt, dR/dt]
    """
    S, I, R = state
    # Normalize by population size to prevent overflow
    S = S / N
    I = I / N
    R = R / N
    
    dSdt = -beta * S * I * N
    dIdt = beta * S * I * N - gamma * I * N
    dRdt = gamma * I * N
    
    return np.array([dSdt, dIdt, dRdt])


# Euler method for solving the SIR model
def euler_method(t_span, h, beta, gamma, S0, I0, R0):
    """
    Solve the SIR model using Euler's method
    
    Parameters:
    t_span (array): Time span [t_start, t_end]
    h (float): Step size
    beta (float): Infection rate
    gamma (float): Recovery rate
    S0 (float): Initial number of susceptible individuals
    I0 (float): Initial number of infected individuals
    R0 (float): Initial number of recovered individuals
    
    Returns:
    tuple: (t_values, S_values, I_values, R_values)
    """
    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / h) + 1
    N = S0 + I0 + R0  # Total population
    
    t_values = np.linspace(t_start, t_end, n_steps)
    S_values = np.zeros(n_steps)
    I_values = np.zeros(n_steps)
    R_values = np.zeros(n_steps)
    
    # Initial conditions
    S_values[0] = S0
    I_values[0] = I0
    R_values[0] = R0
    
    for i in range(1, n_steps):
        state = np.array([S_values[i-1], I_values[i-1], R_values[i-1]])
        derivs = derivatives(state, t_values[i-1], beta, gamma, N)
        
        # Update with bounds checking
        S_values[i] = max(0, min(N, S_values[i-1] + h * derivs[0]))
        I_values[i] = max(0, min(N, I_values[i-1] + h * derivs[1]))
        R_values[i] = max(0, min(N, R_values[i-1] + h * derivs[2]))
        
        # Ensure total population remains constant
        if (total := S_values[i] + I_values[i] + R_values[i]) != 0:
            S_values[i] *= N / total
            I_values[i] *= N / total
            R_values[i] *= N / total
    
    return t_values, S_values, I_values, R_values


# Runge-Kutta 4th order method for solving the SIR model
def rk4_method(t_span, h, beta, gamma, S0, I0, R0):
    """
    Solve the SIR model using the 4th order Runge-Kutta method
    
    Parameters:
    t_span (array): Time span [t_start, t_end]
    h (float): Step size
    beta (float): Infection rate
    gamma (float): Recovery rate
    S0 (float): Initial number of susceptible individuals
    I0 (float): Initial number of infected individuals
    R0 (float): Initial number of recovered individuals
    
    Returns:
    tuple: (t_values, S_values, I_values, R_values)
    """
    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / h) + 1
    N = S0 + I0 + R0  # Total population
    
    t_values = np.linspace(t_start, t_end, n_steps)
    S_values = np.zeros(n_steps)
    I_values = np.zeros(n_steps)
    R_values = np.zeros(n_steps)
    
    # Initial conditions
    S_values[0] = S0
    I_values[0] = I0
    R_values[0] = R0
    
    for i in range(1, n_steps):
        state = np.array([S_values[i-1], I_values[i-1], R_values[i-1]])
        t = t_values[i-1]
        
        # Calculate k1, k2, k3, k4
        k1 = derivatives(state, t, beta, gamma, N)
        k2 = derivatives(state + 0.5 * h * k1, t + 0.5 * h, beta, gamma, N)
        k3 = derivatives(state + 0.5 * h * k2, t + 0.5 * h, beta, gamma, N)
        k4 = derivatives(state + h * k3, t + h, beta, gamma, N)
        
        # Update state using RK4 formula
        state_new = state + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Apply bounds checking
        S_values[i] = max(0, min(N, state_new[0]))
        I_values[i] = max(0, min(N, state_new[1]))
        R_values[i] = max(0, min(N, state_new[2]))
        
        if (total := S_values[i] + I_values[i] + R_values[i]) != 0:
            S_values[i] *= N / total
            I_values[i] *= N / total
            R_values[i] *= N / total
    
    return t_values, S_values, I_values, R_values


# Function to compare Euler and RK4 methods
def compare_methods(t_span, step_sizes, beta, gamma, S0, I0, R0):
    """
    Compare Euler and RK4 methods with different step sizes
    
    Parameters:
    t_span (array): Time span [t_start, t_end]
    step_sizes (list): List of step sizes to compare
    beta (float): Infection rate
    gamma (float): Recovery rate
    S0 (float): Initial number of susceptible individuals
    I0 (float): Initial number of infected individuals
    R0 (float): Initial number of recovered individuals
    """
    plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2)
    
    # Plot comparison for susceptible individuals
    ax1 = plt.subplot(gs[0, 0])
    ax1.set_title("Susceptible Population")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Number of individuals")
    
    # Plot comparison for infected individuals
    ax2 = plt.subplot(gs[0, 1])
    ax2.set_title("Infected Population")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Number of individuals")
    
    # Plot comparison for recovered individuals
    ax3 = plt.subplot(gs[1, 0])
    ax3.set_title("Recovered Population")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Number of individuals")
    
    # Plot error comparison
    ax4 = plt.subplot(gs[1, 1])
    ax4.set_title("Error Comparison (Infected Population)")
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Absolute Difference")
    
    # Use smallest step size RK4 as reference solution
    smallest_h = min(step_sizes)
    ref_t, ref_S, ref_I, ref_R = rk4_method(t_span, smallest_h, beta, gamma, S0, I0, R0)
    
    colors = ["b", "g", "r", "c", "m", "y"]
    line_styles_euler = ["--" for _ in range(len(step_sizes))]
    line_styles_rk4 = ["-" for _ in range(len(step_sizes))]
    
    for i, h in enumerate(step_sizes):
        if h == smallest_h:
            continue
        
        color = colors[i % len(colors)]
        
        # Euler method
        t_euler, S_euler, I_euler, R_euler = euler_method(t_span, h, beta, gamma, S0, I0, R0)
        
        # RK4 method
        t_rk4, S_rk4, I_rk4, R_rk4 = rk4_method(t_span, h, beta, gamma, S0, I0, R0)
        
        # Plot susceptible
        ax1.plot(t_euler, S_euler, line_styles_euler[i], color=color, label=f"Euler (h={h})")
        ax1.plot(t_rk4, S_rk4, line_styles_rk4[i], color=color, label=f"RK4 (h={h})")
        
        # Plot infected
        ax2.plot(t_euler, I_euler, line_styles_euler[i], color=color, label=f"Euler (h={h})")
        ax2.plot(t_rk4, I_rk4, line_styles_rk4[i], color=color, label=f"RK4 (h={h})")
        
        # Plot recovered
        ax3.plot(t_euler, R_euler, line_styles_euler[i], color=color, label=f"Euler (h={h})")
        ax3.plot(t_rk4, R_rk4, line_styles_rk4[i], color=color, label=f"RK4 (h={h})")
        
        # Calculate and plot errors for infected
        I_euler_error = np.abs(np.interp(t_euler, ref_t, ref_I) - I_euler)
        I_rk4_error = np.abs(np.interp(t_rk4, ref_t, ref_I) - I_rk4)
        
        ax4.plot(t_euler, I_euler_error, line_styles_euler[i], color=color, label=f"Euler Error (h={h})")
        ax4.plot(t_rk4, I_rk4_error, line_styles_rk4[i], color=color, label=f"RK4 Error (h={h})")
    
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig("sir_model_comparison.png", dpi=300)
    plt.show()


# Function to analyze the effect of different beta and gamma values
def analyze_parameters(t_span, h, beta_values, gamma_values, S0, I0, R0):
    """
    Analyze the effect of different beta and gamma values on the SIR model
    
    Parameters:
    t_span (array): Time span [t_start, t_end]
    h (float): Step size
    beta_values (list): List of beta values to analyze
    gamma_values (list): List of gamma values to analyze
    S0 (float): Initial number of susceptible individuals
    I0 (float): Initial number of infected individuals
    R0 (float): Initial number of recovered individuals
    """
    plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 1)
    
    ax1 = plt.subplot(gs[0])
    ax1.set_title("Effect of Infection Rate (β) on Infected Population")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Number of Infected Individuals")
    
    colors = ["b", "g", "r", "c", "m", "y"]
    
    # Plot effect of beta
    for i, beta in enumerate(beta_values):
        _, _, I_values, _ = rk4_method(t_span, h, beta, gamma_values[0], S0, I0, R0)
        color = colors[i % len(colors)]
        ax1.plot(np.linspace(t_span[0], t_span[1], len(I_values)), I_values, 
                 "-", color=color, label=f"β={beta}")
    
    ax2 = plt.subplot(gs[1])
    ax2.set_title("Effect of Recovery Rate (γ) on Infected Population")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Number of Infected Individuals")
    ax2.set_ylim(0, 500)  # Set y-axis limits to [0, 500]
    
    # Plot effect of gamma
    for i, gamma in enumerate(gamma_values):
        _, _, I_values, _ = rk4_method(t_span, h, beta_values[0], gamma, S0, I0, R0)
        color = colors[i % len(colors)]
        ax2.plot(np.linspace(t_span[0], t_span[1], len(I_values)), I_values, 
                 "-", color=color, label=f"γ={gamma}")
    
    ax1.legend()
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("sir_parameter_analysis.png", dpi=300)
    plt.show()

# Function to calculate the basic reproduction number R0
def calculate_r0(beta, gamma):
    """Calculate the basic reproduction number R0"""
    return beta / gamma


# Function to calculate summary statistics for the epidemic
def summary_statistics(t_span, h, beta, gamma, S0, I0, R0):
    """
    Calculate and return summary statistics for the epidemic
    
    Parameters:
    t_span (array): Time span [t_start, t_end]
    h (float): Step size
    beta (float): Infection rate
    gamma (float): Recovery rate
    S0 (float): Initial number of susceptible individuals
    I0 (float): Initial number of infected individuals
    R0 (float): Initial number of recovered individuals
    
    Returns:
    dict: Dictionary of summary statistics
    """
    _, _, I_values, R_values = rk4_method(t_span, h, beta, gamma, S0, I0, R0)
    N = S0 + I0 + R0
    
    max_infected = np.max(I_values)
    time_to_peak = np.argmax(I_values) * h
    final_size = R_values[-1]
    r0 = calculate_r0(beta, gamma)
    
    return {
        "r0": r0,
        "max_infected": max_infected,
        "time_to_peak": time_to_peak,
        "final_size": final_size,
        "attack_rate": final_size / N * 100
    }


# Main function to run the simulation
def main():
    beta = 0.3
    gamma = 0.1
    S0 = 990
    I0 = 10
    R0 = 0
    
    t_span = [0, 100]
    step_sizes = [0.1, 0.5, 1.0]
    
    t_euler, S_euler, I_euler, R_euler = euler_method(t_span, 0.1, beta, gamma, S0, I0, R0)
    t_rk4, S_rk4, I_rk4, R_rk4 = rk4_method(t_span, 0.1, beta, gamma, S0, I0, R0)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.title("SIR Model - Euler Method")
    plt.plot(t_euler, S_euler, "b-", label="Susceptible")
    plt.plot(t_euler, I_euler, "r-", label="Infected")
    plt.plot(t_euler, R_euler, "g-", label="Recovered")
    plt.xlabel("Time (days)")
    plt.ylabel("Population")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.title("SIR Model - RK4 Method")
    plt.plot(t_rk4, S_rk4, "b-", label="Susceptible")
    plt.plot(t_rk4, I_rk4, "r-", label="Infected")
    plt.plot(t_rk4, R_rk4, "g-", label="Recovered")
    plt.xlabel("Time (days)")
    plt.ylabel("Population")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("sir_basic_simulation.png", dpi=300)
    plt.show()
    
    compare_methods(t_span, step_sizes, beta, gamma, S0, I0, R0)
    
    beta_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    gamma_values = [0.05, 0.1, 0.15, 0.2, 0.25]
    analyze_parameters(t_span, 0.1, beta_values, gamma_values, S0, I0, R0)
    
    stats = summary_statistics(t_span, 0.1, beta, gamma, S0, I0, R0)
    print("\nSummary Statistics:")
    print(f"Basic Reproduction Number (R0): {stats['r0']:.2f}")
    print(f"Maximum Number of Infected: {stats['max_infected']:.2f}")
    print(f"Time to Peak Infection: {stats['time_to_peak']:.2f} days")
    print(f"Final Epidemic Size: {stats['final_size']:.2f}")
    print(f"Attack Rate: {stats['attack_rate']:.2f}%")


if __name__ == "__main__":
    main()