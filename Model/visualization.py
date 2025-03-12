import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from model import SIRModel

def compare_methods(model, t_span, step_sizes):
    plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2)
    
    ax1 = plt.subplot(gs[0, 0])
    ax1.set_title("Susceptible Population")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Number of individuals")
    
    ax2 = plt.subplot(gs[0, 1])
    ax2.set_title("Infected Population")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Number of individuals")
    
    ax3 = plt.subplot(gs[1, 0])
    ax3.set_title("Recovered Population")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Number of individuals")
    
    ax4 = plt.subplot(gs[1, 1])
    ax4.set_title("Error Comparison (Infected Population)")
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Absolute Difference")
    
    smallest_h = min(step_sizes)
    ref_t, _, ref_I, _ = model.rk4_method(t_span, smallest_h)
    
    colors = ["b", "g", "r", "c", "m", "y"]
    
    for i, h in enumerate(step_sizes):
        if h == smallest_h:
            continue
        
        color = colors[i % len(colors)]
        
        t_euler, _, I_euler, _ = model.euler_method(t_span, h)
        t_rk4, _, I_rk4, _ = model.rk4_method(t_span, h)
        
        ax2.plot(t_euler, I_euler, "--", color=color, label=f"Euler (h={h})")
        ax2.plot(t_rk4, I_rk4, "-", color=color, label=f"RK4 (h={h})")
        
        I_euler_error = np.abs(np.interp(t_euler, ref_t, ref_I) - I_euler)
        I_rk4_error = np.abs(np.interp(t_rk4, ref_t, ref_I) - I_rk4)
        
        ax4.plot(t_euler, I_euler_error, "--", color=color, label=f"Euler Error (h={h})")
        ax4.plot(t_rk4, I_rk4_error, "-", color=color, label=f"RK4 Error (h={h})")
    
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig("sir_model_comparison.png", dpi=300)
    plt.show()
