import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize

def sir_model_ode(y, t, N, beta, gamma):
    """
    Standard SIR Compartmental ODE Model:
    S = Susceptible, I = Infected, R = Recovered
    """
    S, I, R = y
    dS_dt = -beta * S * I / N
    dI_dt = (beta * S * I / N) - (gamma * I)
    dR_dt = gamma * I
    return [dS_dt, dI_dt, dR_dt]

def optimize_sir_parameters(daily_cases, initial_population=100000):
    """
    Uses gradient-based optimization (L-BFGS-B via scipy.optimize.minimize) 
    to dynamically estimate the population catchment N, transmission rate beta, 
    and recovery rate gamma from actual historical case curves.
    """
    t = np.arange(len(daily_cases))
    actual_cumulative = np.cumsum(daily_cases)
    
    def loss_function(params):
        N, beta, gamma = params
        if N <= max(actual_cumulative) or beta < 0 or gamma < 0:
            return 1e15
            
        I0 = max(1.0, actual_cumulative[0])
        S0 = N - I0
        R0 = 0.0
        
        y0 = [S0, I0, R0]
        solution = odeint(sir_model_ode, y0, t, args=(N, beta, gamma))
        predicted_cumulative = solution[:, 1] + solution[:, 2]
        
        mse = np.mean((predicted_cumulative - actual_cumulative) ** 2)
        return mse

    initial_guess = [float(initial_population), 0.3, 0.1]
    bounds = [(1000, 1e7), (0.01, 2.0), (0.001, 1.0)]
    
    result = minimize(loss_function, initial_guess, method='L-BFGS-B', bounds=bounds)
    
    opt_N, opt_beta, opt_gamma = result.x
    return {
        "optimized_N": round(opt_N, 2),
        "optimized_beta": round(opt_beta, 4),
        "optimized_gamma": round(opt_gamma, 4),
        "success": result.success
    }

if __name__ == "__main__":
    sample_cases = [1, 2, 5, 12, 25, 45, 70, 110, 150, 180, 200]
    res = optimize_sir_parameters(sample_cases)
    print("Optimization Result:", res)