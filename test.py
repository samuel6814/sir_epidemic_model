# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Define the SIRModel class
class SIRModel:
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
        self.initial_susceptible = initial_susceptible
        self.initial_infected = initial_infected
        self.initial_recovered = initial_recovered
        self.total_population = initial_susceptible + initial_infected + initial_recovered  # Total population

    def derivatives(self, current_state, time):
        """
        Compute the derivatives for the SIR model.

        Parameters:
        current_state (array): Current state [S, I, R]
        time (float): Current time (not used but included for compatibility)

        Returns:
        array: Derivatives [dS/dt, dI/dt, dR/dt]
        """
        susceptible, infected, recovered = current_state
        # Normalize by population size to prevent overflow
        susceptible = susceptible / self.total_population
        infected = infected / self.total_population
        recovered = recovered / self.total_population

        dS_dt = -self.infection_rate * susceptible * infected * self.total_population
        dI_dt = self.infection_rate * susceptible * infected * self.total_population - self.recovery_rate * infected * self.total_population
        dR_dt = self.recovery_rate * infected * self.total_population

        return np.array([dS_dt, dI_dt, dR_dt])

    def euler_method(self, time_span, step_size):
        """
        Solve the SIR model using Euler's method.

        Parameters:
        time_span (array): Time span [t_start, t_end]
        step_size (float): Step size

        Returns:
        tuple: (time_values, susceptible_values, infected_values, recovered_values)
        """
        start_time, end_time = time_span
        num_steps = int((end_time - start_time) / step_size) + 1

        time_values = np.linspace(start_time, end_time, num_steps)
        susceptible_values = np.zeros(num_steps)
        infected_values = np.zeros(num_steps)
        recovered_values = np.zeros(num_steps)

        # Initial conditions
        susceptible_values[0] = self.initial_susceptible
        infected_values[0] = self.initial_infected
        recovered_values[0] = self.initial_recovered

        for i in range(1, num_steps):
            current_state = np.array([susceptible_values[i-1], infected_values[i-1], recovered_values[i-1]])
            derivatives = self.derivatives(current_state, time_values[i-1])

            # Update with bounds checking
            susceptible_values[i] = max(0, min(self.total_population, susceptible_values[i-1] + step_size * derivatives[0]))
            infected_values[i] = max(0, min(self.total_population, infected_values[i-1] + step_size * derivatives[1]))
            recovered_values[i] = max(0, min(self.total_population, recovered_values[i-1] + step_size * derivatives[2]))

            # Ensure total population remains constant
            if (total := susceptible_values[i] + infected_values[i] + recovered_values[i]) != 0:
                susceptible_values[i] *= self.total_population / total
                infected_values[i] *= self.total_population / total
                recovered_values[i] *= self.total_population / total

        return time_values, susceptible_values, infected_values, recovered_values

    def rk4_method(self, time_span, step_size):
        """
        Solve the SIR model using the 4th order Runge-Kutta method.

        Parameters:
        time_span (array): Time span [t_start, t_end]
        step_size (float): Step size

        Returns:
        tuple: (time_values, susceptible_values, infected_values, recovered_values)
        """
        start_time, end_time = time_span
        num_steps = int((end_time - start_time) / step_size) + 1

        time_values = np.linspace(start_time, end_time, num_steps)
        susceptible_values = np.zeros(num_steps)
        infected_values = np.zeros(num_steps)
        recovered_values = np.zeros(num_steps)

        # Initial conditions
        susceptible_values[0] = self.initial_susceptible
        infected_values[0] = self.initial_infected
        recovered_values[0] = self.initial_recovered

        for i in range(1, num_steps):
            current_state = np.array([susceptible_values[i-1], infected_values[i-1], recovered_values[i-1]])
            current_time = time_values[i-1]

            # Calculate k1, k2, k3, k4
            k1 = self.derivatives(current_state, current_time)
            k2 = self.derivatives(current_state + 0.5 * step_size * k1, current_time + 0.5 * step_size)
            k3 = self.derivatives(current_state + 0.5 * step_size * k2, current_time + 0.5 * step_size)
            k4 = self.derivatives(current_state + step_size * k3, current_time + step_size)

            # Update state using RK4 formula
            new_state = current_state + (step_size/6) * (k1 + 2*k2 + 2*k3 + k4)

            # Apply bounds checking
            susceptible_values[i] = max(0, min(self.total_population, new_state[0]))
            infected_values[i] = max(0, min(self.total_population, new_state[1]))
            recovered_values[i] = max(0, min(self.total_population, new_state[2]))

            if (total := susceptible_values[i] + infected_values[i] + recovered_values[i]) != 0:
                susceptible_values[i] *= self.total_population / total
                infected_values[i] *= self.total_population / total
                recovered_values[i] *= self.total_population / total

        return time_values, susceptible_values, infected_values, recovered_values

    def compare_methods(self, time_span, step_sizes):
        """
        Compare Euler and RK4 methods with different step sizes.

        Parameters:
        time_span (array): Time span [t_start, t_end]
        step_sizes (list): List of step sizes to compare
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
        smallest_step = min(step_sizes)
        ref_time, ref_susceptible, ref_infected, ref_recovered = self.rk4_method(time_span, smallest_step)

        colors = ["b", "g", "r", "c", "m", "y"]
        euler_styles = ["--" for _ in range(len(step_sizes))]
        rk4_styles = ["-" for _ in range(len(step_sizes))]

        for i, step in enumerate(step_sizes):
            if step == smallest_step:
                continue

            color = colors[i % len(colors)]

            # Euler method
            euler_time, euler_susceptible, euler_infected, euler_recovered = self.euler_method(time_span, step)

            # RK4 method
            rk4_time, rk4_susceptible, rk4_infected, rk4_recovered = self.rk4_method(time_span, step)

            # Plot susceptible
            ax1.plot(euler_time, euler_susceptible, euler_styles[i], color=color, label=f"Euler (h={step})")
            ax1.plot(rk4_time, rk4_susceptible, rk4_styles[i], color=color, label=f"RK4 (h={step})")

            # Plot infected
            ax2.plot(euler_time, euler_infected, euler_styles[i], color=color, label=f"Euler (h={step})")
            ax2.plot(rk4_time, rk4_infected, rk4_styles[i], color=color, label=f"RK4 (h={step})")

            # Plot recovered
            ax3.plot(euler_time, euler_recovered, euler_styles[i], color=color, label=f"Euler (h={step})")
            ax3.plot(rk4_time, rk4_recovered, rk4_styles[i], color=color, label=f"RK4 (h={step})")

            # Calculate and plot errors for infected
            euler_error = np.abs(np.interp(euler_time, ref_time, ref_infected) - euler_infected)
            rk4_error = np.abs(np.interp(rk4_time, ref_time, ref_infected) - rk4_infected)

            ax4.plot(euler_time, euler_error, euler_styles[i], color=color, label=f"Euler Error (h={step})")
            ax4.plot(rk4_time, rk4_error, rk4_styles[i], color=color, label=f"RK4 Error (h={step})")

        ax1.legend()
        ax2.legend()
        ax3.legend()
        ax4.legend()

        plt.tight_layout()
        plt.savefig("sir_model_comparison.png", dpi=300)
        plt.show()

    def analyze_parameters(self, time_span, step_size, infection_rates, recovery_rates):
        """
        Analyze the effect of different infection and recovery rates on the SIR model.

        Parameters:
        time_span (array): Time span [t_start, t_end]
        step_size (float): Step size
        infection_rates (list): List of infection rates (beta) to analyze
        recovery_rates (list): List of recovery rates (gamma) to analyze
        """
        plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 1)

        ax1 = plt.subplot(gs[0])
        ax1.set_title("Effect of Infection Rate (β) on Infected Population")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Number of Infected Individuals")

        original_infection_rate = self.infection_rate
        original_recovery_rate = self.recovery_rate

        colors = ["b", "g", "r", "c", "m", "y"]

        for i, beta in enumerate(infection_rates):
            self.infection_rate = beta
            _, _, infected_values, _ = self.rk4_method(time_span, step_size)

            color = colors[i % len(colors)]
            ax1.plot(np.linspace(time_span[0], time_span[1], len(infected_values)), infected_values, 
                     "-", color=color, label=f"β={beta}")

        self.infection_rate = original_infection_rate

        ax2 = plt.subplot(gs[1])
        ax2.set_title("Effect of Recovery Rate (γ) on Infected Population")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Number of Infected Individuals")

        for i, gamma in enumerate(recovery_rates):
            self.recovery_rate = gamma
            _, _, infected_values, _ = self.rk4_method(time_span, step_size)

            color = colors[i % len(colors)]
            ax2.plot(np.linspace(time_span[0], time_span[1], len(infected_values)), infected_values, 
                     "-", color=color, label=f"γ={gamma}")

        self.recovery_rate = original_recovery_rate

        ax1.legend()
        ax2.legend()

        plt.tight_layout()
        plt.savefig("sir_parameter_analysis.png", dpi=300)
        plt.show()

    def calculate_r0(self):
        """Calculate the basic reproduction number R0."""
        return self.infection_rate / self.recovery_rate

    def summary_statistics(self, time_span, step_size):
        """
        Calculate and return summary statistics for the epidemic.

        Parameters:
        time_span (array): Time span [t_start, t_end]
        step_size (float): Step size

        Returns:
        dict: Dictionary of summary statistics
        """
        _, _, infected_values, recovered_values = self.rk4_method(time_span, step_size)

        max_infected = np.max(infected_values)
        time_to_peak = np.argmax(infected_values) * step_size
        final_size = recovered_values[-1]
        r0 = self.calculate_r0()

        return {
            "r0": r0,
            "max_infected": max_infected,
            "time_to_peak": time_to_peak,
            "final_size": final_size,
            "attack_rate": final_size / self.total_population * 100
        }

# Main function to run the SIR model
def main():
    # Define initial parameters
    infection_rate = 0.3
    recovery_rate = 0.1
    initial_susceptible = 990
    initial_infected = 10
    initial_recovered = 0

    # Initialize the SIR model
    sir_model = SIRModel(infection_rate, recovery_rate, initial_susceptible, initial_infected, initial_recovered)
    time_span = [0, 100]
    step_sizes = [0.1, 0.5, 1.0]

    # Solve using Euler and RK4 methods
    euler_time, euler_susceptible, euler_infected, euler_recovered = sir_model.euler_method(time_span, 0.1)
    rk4_time, rk4_susceptible, rk4_infected, rk4_recovered = sir_model.rk4_method(time_span, 0.1)

    # Plot results
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.title("SIR Model - Euler Method")
    plt.plot(euler_time, euler_susceptible, "b-", label="Susceptible")
    plt.plot(euler_time, euler_infected, "r-", label="Infected")
    plt.plot(euler_time, euler_recovered, "g-", label="Recovered")
    plt.xlabel("Time (days)")
    plt.ylabel("Population")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.title("SIR Model - RK4 Method")
    plt.plot(rk4_time, rk4_susceptible, "b-", label="Susceptible")
    plt.plot(rk4_time, rk4_infected, "r-", label="Infected")
    plt.plot(rk4_time, rk4_recovered, "g-", label="Recovered")
    plt.xlabel("Time (days)")
    plt.ylabel("Population")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("sir_basic_simulation.png", dpi=300)
    plt.show()

    # Compare methods and analyze parameters
    sir_model.compare_methods(time_span, step_sizes)

    infection_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
    recovery_rates = [0.05, 0.1, 0.15, 0.2, 0.25]
    sir_model.analyze_parameters(time_span, 0.1, infection_rates, recovery_rates)

    # Print summary statistics
    stats = sir_model.summary_statistics(time_span, 0.1)
    print("\nSummary Statistics:")
    print(f"Basic Reproduction Number (R0): {stats['r0']:.2f}")
    print(f"Maximum Number of Infected: {stats['max_infected']:.2f}")
    print(f"Time to Peak Infection: {stats['time_to_peak']:.2f} days")
    print(f"Final Epidemic Size: {stats['final_size']:.2f}")
    print(f"Attack Rate: {stats['attack_rate']:.2f}%")

# Run the main function
if __name__ == "__main__":
    main()