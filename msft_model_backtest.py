import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

np.random.seed(43)
stock = "MSFT"
train_start, train_end = "2020-09-01", "2025-09-01"
test_start, test_end = "2025-09-01", "2026-09-02"
annual_steps, simulations = 252, 10000

def gbm(initial, mu, sigma, years, steps, paths):
    dt = years / steps
    shocks = np.random.standard_normal((steps, paths))
    log_steps = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks
    return initial * np.vstack([np.ones(paths), np.exp(np.cumsum(log_steps, axis=0))])

def heston(initial, mu, v0, kappa, theta, volvol, rho, years, steps, paths):
    dt = years / steps
    prices = np.zeros((steps + 1, paths))
    variances = np.zeros((steps + 1, paths))
    prices[0], variances[0] = initial, v0
    for step in range(1, steps + 1):
        z_stock = np.random.standard_normal(paths)
        z_variance = np.random.standard_normal(paths)
        z_variance = rho * z_stock + np.sqrt(1 - rho**2) * z_variance
        previous = np.maximum(variances[step - 1], 0)
        variances[step] = np.maximum(previous + kappa * (theta - previous) * dt + volvol * np.sqrt(previous * dt) * z_variance, 0)
        prices[step] = prices[step - 1] * np.exp((mu - 0.5 * previous) * dt + np.sqrt(previous * dt) * z_stock)
    return prices

data = yf.download(stock, start=train_start, end=test_end, progress=False, auto_adjust=False)
close_prices = data["Close"][stock] if isinstance(data.columns, pd.MultiIndex) else data["Close"]
close_prices = close_prices.dropna()
close_prices.index = pd.to_datetime(close_prices.index).tz_localize(None)
train, test = close_prices.loc[train_start:train_end], close_prices.loc[test_start:test_end]
if train.empty or test.empty or test.index[-1] < pd.Timestamp("2026-09-01"):
    raise ValueError("MSFT data does not cover the requested periods.")

returns = np.log(train / train.shift(1)).dropna()
daily_var = returns.var()
mu = (returns.mean() + 0.5 * daily_var) * annual_steps
gbm_sigma = returns.std() * np.sqrt(annual_steps)
rolling = (returns.rolling(21).var() * annual_steps).dropna()
v0 = float(rolling.iloc[-1])
v_now, v_next = rolling.iloc[:-1].to_numpy(), rolling.iloc[1:].to_numpy()
intercept, slope = np.linalg.lstsq(np.column_stack([np.ones(len(v_now)), v_now]), v_next, rcond=None)[0]
dt = 1 / annual_steps
if 0 < slope < 1:
    kappa, theta = -np.log(slope) / dt, intercept / (1 - slope)
else:
    kappa, theta = 2.0, float(rolling.mean())
theta = theta if np.isfinite(theta) and theta > 0 else float(rolling.mean())
kappa = kappa if np.isfinite(kappa) and kappa > 0 else 2.0
residuals = v_next - v_now - kappa * (theta - v_now) * dt
valid = v_now > 1e-10
volvol = float(np.std(residuals[valid] / np.sqrt(v_now[valid] * dt), ddof=1))
volvol = volvol if np.isfinite(volvol) and volvol > 0 else 0.01
aligned = returns.loc[rolling.index].iloc[1:].to_numpy()[valid]
stock_shocks = (aligned - mu * dt) / np.sqrt(v_now[valid] * dt)
variance_shocks = residuals[valid] / (volvol * np.sqrt(v_now[valid] * dt))
rho = float(np.corrcoef(stock_shocks, variance_shocks)[0, 1])
rho = float(np.clip(rho if np.isfinite(rho) else 0.0, -0.999, 0.999))

initial = float(train.iloc[-1])
steps = len(test) - 1
years = steps / annual_steps
dates, actual = test.index, test.to_numpy()
gbm_paths = gbm(initial, mu, gbm_sigma, years, steps, simulations)
heston_paths = heston(initial, mu, v0, kappa, theta, volvol, rho, years, steps, simulations)

def path_plot(paths, color, title):
    fig, ax = plt.subplots(figsize=(11, 6))
    for path in paths[:, :200].T:
        ax.plot(dates, path, color=color, alpha=0.08, linewidth=0.8)
    ax.plot(dates, actual, color="black", linewidth=2.5, label="Actual MSFT")
    ax.axhline(initial, color="firebrick", linestyle="--", label="Starting price")
    ax.set(title=title, xlabel="Date", ylabel="Price ($)")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    plt.show()

path_plot(gbm_paths, "royalblue", "GBM Forecast Paths vs Actual MSFT")
fig, ax = plt.subplots(figsize=(11, 6))
ax.hist(gbm_paths[-1], bins=50, color="royalblue", edgecolor="white")
ax.axvline(actual[-1], color="black", linewidth=2.5, label=f"Actual: ${actual[-1]:.2f}")
ax.axvline(np.median(gbm_paths[-1]), color="darkorange", linewidth=2, label="GBM median")
ax.set(title="GBM Terminal-Price Distribution", xlabel="Price on September 1, 2026 ($)", ylabel="Simulation count")
ax.grid(True, linestyle="--", alpha=0.3)
ax.legend()
plt.show()

path_plot(heston_paths, "seagreen", "Heston Forecast Paths vs Actual MSFT")
fig, ax = plt.subplots(figsize=(11, 6))
ax.hist(heston_paths[-1], bins=50, color="seagreen", edgecolor="white")
ax.axvline(actual[-1], color="black", linewidth=2.5, label=f"Actual: ${actual[-1]:.2f}")
ax.axvline(np.median(heston_paths[-1]), color="darkorange", linewidth=2, label="Heston median")
ax.set(title="Heston Terminal-Price Distribution", xlabel="Price on September 1, 2026 ($)", ylabel="Simulation count")
ax.grid(True, linestyle="--", alpha=0.3)
ax.legend()
plt.show()

fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(dates, actual, color="black", linewidth=2.5)
ax.scatter(dates[[0, -1]], actual[[0, -1]], color="firebrick", zorder=3)
ax.set(title="Actual MSFT Price: September 1, 2025 to September 1, 2026", xlabel="Date", ylabel="Price ($)")
ax.grid(True, linestyle="--", alpha=0.3)
fig.autofmt_xdate()
plt.show()

print(f"Calibration: {train.index[0].date()} to {train.index[-1].date()}")
print(f"Test: {test.index[0].date()} to {test.index[-1].date()}")
print(f"Starting price: ${initial:.2f}; actual final price: ${actual[-1]:.2f}")