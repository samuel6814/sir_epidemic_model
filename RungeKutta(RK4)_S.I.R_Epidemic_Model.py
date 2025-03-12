import numpy as np
import matplotlib.pyplot as plt

class SIRModelRK4:
    def __init__(self, beta, gamma, S0, I0, R0):
        self.beta = beta
        self.gamma = gamma
        self.S0 = S0
        self.I0 = I0
        self.R0 = R0
        self.N = S0 + I0 + R0

    def derivatives(self, state):
        S, I, R = state
        S /= self.N
        I /= self.N
        R /= self.N

        dSdt = -self.beta * S * I * self.N
        dIdt = self.beta * S * I * self.N - self.gamma * I * self.N
        dRdt = self.gamma * I * self.N

        return np.array([dSdt, dIdt, dRdt])

    def rk4_method(self, t_span, h):
        t_start, t_end = t_span
        n_steps = int((t_end - t_start) / h) + 1

        t_values = np.linspace(t_start, t_end, n_steps)
        S_values = np.zeros(n_steps)
        I_values = np.zeros(n_steps)
        R_values = np.zeros(n_steps)

        S_values[0] = self.S0
        I_values[0] = self.I0
        R_values[0] = self.R0

        for i in range(1, n_steps):
            state = np.array([S_values[i-1], I_values[i-1], R_values[i-1]])

            k1 = self.derivatives(state)
            k2 = self.derivatives(state + 0.5 * h * k1)
            k3 = self.derivatives(state + 0.5 * h * k2)
            k4 = self.derivatives(state + h * k3)

            state_new = state + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

            S_values[i] = max(0, min(self.N, state_new[0]))
            I_values[i] = max(0, min(self.N, state_new[1]))
            R_values[i] = max(0, min(self.N, state_new[2]))

        return t_values, S_values, I_values, R_values

def main():
    beta = 0.3
    gamma = 0.1
    S0 = 990
    I0 = 10
    R0 = 0
    t_span = [0, 100]
    h = 0.1

    sir_rk4 = SIRModelRK4(beta, gamma, S0, I0, R0)
    t_values, S_values, I_values, R_values = sir_rk4.rk4_method(t_span, h)

    plt.figure(figsize=(10, 6))
    plt.plot(t_values, S_values, 'b-', label='Susceptible')
    plt.plot(t_values, I_values, 'r-', label='Infected')
    plt.plot(t_values, R_values, 'g-', label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.legend()
    plt.title('SIR Model - Runge-Kutta (RK4) Method')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
