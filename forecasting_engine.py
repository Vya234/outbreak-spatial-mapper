import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize


def sir_model_ode(y, t, N, beta, gamma):
    """
    Standard SIR compartmental model.

    S: susceptible population
    I: currently infectious population
    R: recovered population
    """
    S, I, R = y

    dS_dt = -beta * S * I / N
    dI_dt = beta * S * I / N - gamma * I
    dR_dt = gamma * I

    return [dS_dt, dI_dt, dR_dt]


def _validate_daily_counts(daily_cases):
    """Validate a sequence of non-negative daily logged counts."""
    values = np.asarray(daily_cases, dtype=float)

    if values.ndim != 1 or len(values) < 3:
        raise ValueError(
            "At least 3 daily observations are required."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError("Daily observations must be finite.")

    if np.any(values < 0):
        raise ValueError(
            "Daily observations cannot be negative."
        )

    cumulative = np.cumsum(values)

    if cumulative[-1] <= 0:
        raise ValueError(
            "At least one positive observation is required."
        )

    return values, cumulative


def _simulate_sir(N, beta, gamma, initial_infected, t):
    """Simulate an SIR trajectory from its initial conditions."""
    if N <= initial_infected or initial_infected <= 0:
        raise ValueError(
            "Population N must exceed the positive initial infected count."
        )

    S0 = N - initial_infected
    I0 = initial_infected
    R0 = 0.0

    return odeint(
        sir_model_ode,
        [S0, I0, R0],
        t,
        args=(N, beta, gamma),
    )


def optimize_sir_parameters(
    daily_cases,
    initial_population=100000,
):
    """
    Fit SIR parameters to cumulative observed counts using
    bounded multi-start optimization.

    IMPORTANT:
    Daily logged locations are not necessarily new infections.
    The fitted parameters are exploratory and must not be interpreted
    epidemiologically without validating that assumption.
    """
    values, actual_cumulative = _validate_daily_counts(daily_cases)
    initial_infected = float(actual_cumulative[0])

    if not np.isfinite(initial_population) or initial_population <= 0:
        raise ValueError("initial_population must be positive.")

    lower_N = max(
        1000.0,
        float(actual_cumulative[-1]) + 1.0,
    )
    upper_N = 1e7

    if lower_N >= upper_N:
        raise ValueError(
            "Observed cumulative count is too large for the "
            "configured population bounds."
        )

    t = np.arange(len(values), dtype=float)

    def loss_function(params):
        N, beta, gamma = params

        try:
            solution = _simulate_sir(
                N, beta, gamma, initial_infected, t
            )
            predicted_cumulative = solution[:, 1] + solution[:, 2]

            if (
                not np.all(np.isfinite(predicted_cumulative))
                or np.any(predicted_cumulative < -1e-6)
            ):
                return 1e30

            errors = predicted_cumulative - actual_cumulative
            return float(np.mean(errors ** 2))

        except (ValueError, FloatingPointError):
            return 1e30

    # Try diverse initial values to reduce dependence on one start.
    population_starts = sorted(set([
        min(max(float(initial_population), lower_N + 1.0), upper_N),
        min(max(float(actual_cumulative[-1] * 1.05), lower_N + 1.0), upper_N),
        min(max(float(actual_cumulative[-1] * 1.5), lower_N + 1.0), upper_N),
        min(max(float(actual_cumulative[-1] * 3.0), lower_N + 1.0), upper_N),
        min(max(float(actual_cumulative[-1] * 10.0), lower_N + 1.0), upper_N),
    ]))

    rate_starts = [
        (0.1, 0.05),
        (0.3, 0.1),
        (0.5, 0.2),
        (0.8, 0.4),
        (1.2, 0.6),
        (1.5, 0.9),
    ]

    candidates = []

    for start_N in population_starts:
        for start_beta, start_gamma in rate_starts:
            result = minimize(
                loss_function,
                x0=[start_N, start_beta, start_gamma],
                method="L-BFGS-B",
                bounds=[
                    (lower_N, upper_N),
                    (0.01, 2.0),
                    (0.001, 1.0),
                ],
                options={"maxiter": 3000, "ftol": 1e-12},
            )

            if result.success and np.isfinite(result.fun) and result.fun < 1e29:
                candidates.append(result)

    if not candidates:
        raise RuntimeError("All SIR optimization attempts failed.")

    # Select the candidate with the lowest objective value.
    best = min(candidates, key=lambda result: result.fun)
    opt_N, opt_beta, opt_gamma = best.x

    return {
        "optimized_N": round(float(opt_N), 2),
        "optimized_beta": round(float(opt_beta), 4),
        "optimized_gamma": round(float(opt_gamma), 4),
        "success": bool(best.success),
        "fit_mse": round(float(best.fun), 4),
        "successful_starts": sum(r.success for r in candidates),
        "total_starts": len(candidates),
    }


def forecast_sir_cumulative(
    daily_cases,
    forecast_horizon,
    population,
    beta,
    gamma,
):
    """
    Continue an SIR trajectory from the beginning of the observed
    series through the requested forecast horizon.

    The returned value is the model's cumulative infected count
    (I + R) at the end of the forecast horizon.

    This is a prototype estimate, not a validated prediction of
    actual disease incidence when inputs are logged locations.
    """
    values, _ = _validate_daily_counts(daily_cases)

    if not isinstance(forecast_horizon, (int, np.integer)):
        raise ValueError("forecast_horizon must be an integer.")

    if forecast_horizon < 0:
        raise ValueError("forecast_horizon cannot be negative.")

    if population <= 0 or beta <= 0 or gamma <= 0:
        raise ValueError(
            "Population, beta, and gamma must be positive."
        )

    initial_infected = float(values[0])

    # Continue the same trajectory used for fitting.
    final_t = len(values) - 1 + forecast_horizon
    t = np.arange(final_t + 1, dtype=float)

    solution = _simulate_sir(
        float(population),
        float(beta),
        float(gamma),
        initial_infected,
        t,
    )

    predicted_cumulative = solution[-1, 1] + solution[-1, 2]

    if not np.isfinite(predicted_cumulative):
        raise RuntimeError("SIR forecast produced a non-finite result.")

    return float(predicted_cumulative)


if __name__ == "__main__":
    sample_cases = [1, 2, 5, 12, 25, 45, 70, 110, 150, 180, 200]

    result = optimize_sir_parameters(sample_cases)
    print("Optimization result:", result)

    if result["success"]:
        forecast = forecast_sir_cumulative(
            daily_cases=sample_cases,
            forecast_horizon=7,
            population=result["optimized_N"],
            beta=result["optimized_beta"],
            gamma=result["optimized_gamma"],
        )

        print("7-day cumulative model estimate:", round(forecast, 2))
