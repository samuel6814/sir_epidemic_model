import numpy as np

class SIRModel:
    def __init__(self, beta, gamma, S0, I0, R0):
        self.beta = beta
        self.gamma = gamma
        self.S0 = S0
        self.I0 = I0
        self.R0 = R0
        self.N = S0 + I0 + R0  # Total population
        
    def derivatives(self, state, t):
        S, I, R = state
        S, I, R = S / self.N, I / self.N, R / self.N
        
        dSdt = -self.beta * S * I * self.N
        dIdt = self.beta * S * I * self.N - self.gamma * I * self.N
        dRdt = self.gamma * I * self.N
        
        return np.array([dSdt, dIdt, dRdt])
    
    def euler_method(self, t_span, h):
        t_start, t_end = t_span
        n_steps = int((t_end - t_start) / h) + 1
        
        t_values = np.linspace(t_start, t_end, n_steps)
        S_values, I_values, R_values = np.zeros(n_steps), np.zeros(n_steps), np.zeros(n_steps)
        
        S_values[0], I_values[0], R_values[0] = self.S0, self.I0, self.R0
        
        for i in range(1, n_steps):
            state = np.array([S_values[i-1], I_values[i-1], R_values[i-1]])
            derivatives = self.derivatives(state, t_values[i-1])
            
            S_values[i] = max(0, min(self.N, S_values[i-1] + h * derivatives[0]))
            I_values[i] = max(0, min(self.N, I_values[i-1] + h * derivatives[1]))
            R_values[i] = max(0, min(self.N, R_values[i-1] + h * derivatives[2]))
            
            if (total := S_values[i] + I_values[i] + R_values[i]) != 0:
                S_values[i] *= self.N / total
                I_values[i] *= self.N / total
                R_values[i] *= self.N / total
        
        return t_values, S_values, I_values, R_values
    
    def rk4_method(self, t_span, h):
        t_start, t_end = t_span
        n_steps = int((t_end - t_start) / h) + 1
        
        t_values = np.linspace(t_start, t_end, n_steps)
        S_values, I_values, R_values = np.zeros(n_steps), np.zeros(n_steps), np.zeros(n_steps)
        
        S_values[0], I_values[0], R_values[0] = self.S0, self.I0, self.R0
        
        for i in range(1, n_steps):
            state = np.array([S_values[i-1], I_values[i-1], R_values[i-1]])
            t = t_values[i-1]
            
            k1 = self.derivatives(state, t)
            k2 = self.derivatives(state + 0.5 * h * k1, t + 0.5 * h)
            k3 = self.derivatives(state + 0.5 * h * k2, t + 0.5 * h)
            k4 = self.derivatives(state + h * k3, t + h)
            
            state_new = state + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
            
            S_values[i] = max(0, min(self.N, state_new[0]))
            I_values[i] = max(0, min(self.N, state_new[1]))
            R_values[i] = max(0, min(self.N, state_new[2]))
            
            if (total := S_values[i] + I_values[i] + R_values[i]) != 0:
                S_values[i] *= self.N / total
                I_values[i] *= self.N / total
                R_values[i] *= self.N / total
        
        return t_values, S_values, I_values, R_values
    
    def calculate_r0(self):
        return self.beta / self.gamma
    
    def summary_statistics(self, t_span, h):
        _, _, I_values, R_values = self.rk4_method(t_span, h)
        
        max_infected = np.max(I_values)
        time_to_peak = np.argmax(I_values) * h
        final_size = R_values[-1]
        r0 = self.calculate_r0()
        
        return {
            "r0": r0,
            "max_infected": max_infected,
            "time_to_peak": time_to_peak,
            "final_size": final_size,
            "attack_rate": final_size / self.N * 100
        }
