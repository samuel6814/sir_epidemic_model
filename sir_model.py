import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

class EpidemicModel:
    def __init__(self, infection_rate, recovery_rate, initial_susceptible, initial_infected, initial_recovered):
        """
        Initialize the SIR model with given parameters.
        
        Parameters:
        infection_rate (float): Rate of infection (beta)
        recovery_rate (float): Rate of recovery (gamma)
        initial_susceptible (float): Initial number of susceptible individuals
        initial_infected (float): Initial number of infected individuals
        initial_recovered (float): Initial number of recovered individuals
        """
        self.infection_rate = infection_rate
        self.recovery_rate = recovery_rate
        self.susceptible_start = initial_susceptible
        self.infected_start = initial_infected
        self.recovered_start = initial_recovered
        self.population_size = initial_susceptible + initial_infected + initial_recovered  # Total population size

    def compute_derivatives(self, state_vector, current_time):
        """
        Compute the rate of change for the SIR model.
        
        Parameters:
        state_vector (array): Current values [susceptible, infected, recovered]
        current_time (float): Current time (not used but included for compatibility)
        
        Returns:
        array: Derivatives [dS/dt, dI/dt, dR/dt]
        """
        susceptible, infected, recovered = state_vector
        
        # Normalize values to prevent overflow errors
        susceptible /= self.population_size
        infected /= self.population_size
        recovered /= self.population_size
        
        # Compute rate of change
        susceptible_change = -self.infection_rate * susceptible * infected * self.population_size
        infected_change = self.infection_rate * susceptible * infected * self.population_size - self.recovery_rate * infected * self.population_size
        recovered_change = self.recovery_rate * infected * self.population_size
        
        return np.array([susceptible_change, infected_change, recovered_change])
    
    def euler_solver(self, time_range, step_size):
        """
        Solve the SIR model using Euler's method.
        
        Parameters:
        time_range (array): Start and end time [t_start, t_end]
        step_size (float): Step size for integration
        
        Returns:
        tuple: (time_values, susceptible_values, infected_values, recovered_values)
        """
        t_start, t_end = time_range
        num_steps = int((t_end - t_start) / step_size) + 1
        
        time_values = np.linspace(t_start, t_end, num_steps)
        susceptible_values = np.zeros(num_steps)
        infected_values = np.zeros(num_steps)
        recovered_values = np.zeros(num_steps)
        
        # Set initial conditions
        susceptible_values[0] = self.susceptible_start
        infected_values[0] = self.infected_start
        recovered_values[0] = self.recovered_start
        
        # Euler integration loop
        for i in range(1, num_steps):
            current_state = np.array([susceptible_values[i-1], infected_values[i-1], recovered_values[i-1]])
            rate_of_change = self.compute_derivatives(current_state, time_values[i-1])
            
            # Update values with bounds checking
            susceptible_values[i] = max(0, min(self.population_size, susceptible_values[i-1] + step_size * rate_of_change[0]))
            infected_values[i] = max(0, min(self.population_size, infected_values[i-1] + step_size * rate_of_change[1]))
            recovered_values[i] = max(0, min(self.population_size, recovered_values[i-1] + step_size * rate_of_change[2]))
            
            # Ensure population size remains consistent
            total_population = susceptible_values[i] + infected_values[i] + recovered_values[i]
            if total_population != 0:
                susceptible_values[i] *= self.population_size / total_population
                infected_values[i] *= self.population_size / total_population
                recovered_values[i] *= self.population_size / total_population
        
        return time_values, susceptible_values, infected_values, recovered_values

    def runge_kutta_solver(self, time_range, step_size):
        """
        Solve the SIR model using the 4th order Runge-Kutta method.
        
        Parameters:
        time_range (array): Start and end time [t_start, t_end]
        step_size (float): Step size for integration
        
        Returns:
        tuple: (time_values, susceptible_values, infected_values, recovered_values)
        """
        t_start, t_end = time_range
        num_steps = int((t_end - t_start) / step_size) + 1
        
        time_values = np.linspace(t_start, t_end, num_steps)
        susceptible_values = np.zeros(num_steps)
        infected_values = np.zeros(num_steps)
        recovered_values = np.zeros(num_steps)
        
        # Set initial conditions
        susceptible_values[0] = self.susceptible_start
        infected_values[0] = self.infected_start
        recovered_values[0] = self.recovered_start
        
        # Runge-Kutta integration loop
        for i in range(1, num_steps):
            current_state = np.array([susceptible_values[i-1], infected_values[i-1], recovered_values[i-1]])
            time_current = time_values[i-1]
            
            # Compute k1, k2, k3, k4
            k1 = self.compute_derivatives(current_state, time_current)
            k2 = self.compute_derivatives(current_state + 0.5 * step_size * k1, time_current + 0.5 * step_size)
            k3 = self.compute_derivatives(current_state + 0.5 * step_size * k2, time_current + 0.5 * step_size)
            k4 = self.compute_derivatives(current_state + step_size * k3, time_current + step_size)
            
            # Compute the next state using the weighted sum of k1, k2, k3, k4
            updated_state = current_state + (step_size / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            
            # Apply bounds checking
            susceptible_values[i] = max(0, min(self.population_size, updated_state[0]))
            infected_values[i] = max(0, min(self.population_size, updated_state[1]))
            recovered_values[i] = max(0, min(self.population_size, updated_state[2]))
            
            # Maintain population consistency
            total_population = susceptible_values[i] + infected_values[i] + recovered_values[i]
            if total_population != 0:
                susceptible_values[i] *= self.population_size / total_population
                infected_values[i] *= self.population_size / total_population
                recovered_values[i] *= self.population_size / total_population
        
        return time_values, susceptible_values, infected_values, recovered_values

def main():
    # Define model parameters
    infection_rate = 0.3   # Beta: rate of infection
    recovery_rate = 0.1    # Gamma: rate of recovery
    initial_susceptible = 990
    initial_infected = 10
    initial_recovered = 0
    
    # Create an epidemic model instance
    model = EpidemicModel(infection_rate, recovery_rate, initial_susceptible, initial_infected, initial_recovered)
    
    # Define the simulation time span and step sizes
    time_span = [0, 100]
    step_size = 0.1

    # Solve the model using Euler's method
    time_euler, susceptible_euler, infected_euler, recovered_euler = model.euler_solver(time_span, step_size)
    
    # Solve the model using Runge-Kutta method
    time_rk, susceptible_rk, infected_rk, recovered_rk = model.runge_kutta_solver(time_span, step_size)
    
    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(time_euler, susceptible_euler, "b--", label="Susceptible (Euler)")
    plt.plot(time_euler, infected_euler, "r--", label="Infected (Euler)")
    plt.plot(time_euler, recovered_euler, "g--", label="Recovered (Euler)")
    
    plt.plot(time_rk, susceptible_rk, "b-", label="Susceptible (RK4)")
    plt.plot(time_rk, infected_rk, "r-", label="Infected (RK4)")
    plt.plot(time_rk, recovered_rk, "g-", label="Recovered (RK4)")
    
    plt.xlabel("Time (days)")
    plt.ylabel("Population")
    plt.legend()
    plt.title("SIR Model - Euler vs Runge-Kutta")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
