from model import SIRModel
import matplotlib.pyplot as plt
from visualization import compare_methods

def main():
    beta = 0.3
    gamma = 0.1
    S0, I0, R0 = 990, 10, 0
    sir_model = SIRModel(beta, gamma, S0, I0, R0)
    
    t_span = [0, 100]
    step_sizes = [0.1, 0.5, 1.0]
    
    compare_methods(sir_model, t_span, step_sizes)
    
    stats = sir_model.summary_statistics(t_span, 0.1)
    print("\nSummary Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")

if __name__ == "__main__":
    main()
