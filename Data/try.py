import pandas as pd
import numpy as np
from scipy.optimize import minimize

# -----------------------------------
# 1. Load data
# -----------------------------------
df = pd.read_csv("data.csv")

# Hotel attributes only
features = [
    "prop_starrating",
    "prop_review_score",
    "prop_brand_bool",
    "prop_location_score",
    "prop_accesibility_score",
    "prop_log_historical_price",
    "price_usd",
    "promotion_flag"
]

# Keep only needed columns and drop missing rows
df = df[["srch_id", "booking_bool"] + features].dropna().copy()

# -----------------------------------
# 2. Standardize continuous variables
#    (leave binary indicators alone)
# -----------------------------------
binary_cols = ["prop_brand_bool", "promotion_flag"]
continuous_cols = [c for c in features if c not in binary_cols]

for c in continuous_cols:
    mean_c = df[c].mean()
    std_c = df[c].std(ddof=0)
    if std_c > 0:
        df[c] = (df[c] - mean_c) / std_c

# Build arrays
X = df[features].to_numpy(dtype=float)
y = df["booking_bool"].to_numpy(dtype=float)
srch_id = df["srch_id"].to_numpy()

# Convert srch_id to group indices 0,1,2,...
_, group_idx = np.unique(srch_id, return_inverse=True)
n_groups = group_idx.max() + 1

# -----------------------------------
# 3. Negative log-likelihood + gradient
#    MNL with outside option normalized to:
#    u0 = 0  <=>  v0 = 1
# -----------------------------------
def nll_and_grad(theta):
    """
    theta[0] = intercept
    theta[1:] = coefficients on hotel features

    Hotel utility:
        u_j = beta0 + x_j' beta

    No-purchase utility:
        u_0 = 0
    """
    beta0 = theta[0]
    beta = theta[1:]

    # Utility for each hotel row
    u = beta0 + X @ beta

    # Numerical stabilization:
    # For each search, compute max(0, max hotel utility in that search)
    # Including 0 here matters because the outside option has utility 0.
    max_u = np.zeros(n_groups)
    np.maximum.at(max_u, group_idx, u)

    # Stable exponentials for hotel utilities
    exp_shifted = np.exp(u - max_u[group_idx])

    # For each search:
    # denominator = exp(0 - max_u) + sum_j exp(u_j - max_u)
    sum_exp_shifted = np.bincount(group_idx, weights=exp_shifted, minlength=n_groups)
    denom_shifted = np.exp(-max_u) + sum_exp_shifted

    # log denominator = max_u + log(denom_shifted)
    log_denom = max_u + np.log(denom_shifted)

    # Log-likelihood:
    #   sum over chosen hotels of u_j
    #   minus sum over searches of log(1 + sum exp(u_j))
    ll = np.sum(y * u) - np.sum(log_denom)

    # Predicted probability that each hotel is chosen
    p = exp_shifted / denom_shifted[group_idx]

    # Gradient
    resid = y - p
    grad0 = np.sum(resid)
    grad_beta = X.T @ resid

    grad = np.concatenate(([grad0], grad_beta))

    # Return negative log-likelihood and negative gradient
    return -ll, -grad


# -----------------------------------
# 4. Estimate coefficients
# -----------------------------------
theta0 = np.zeros(X.shape[1] + 1)

result = minimize(
    fun=lambda th: nll_and_grad(th),
    x0=theta0,
    jac=True,
    method="L-BFGS-B"
)

# -----------------------------------
# 5. Print results
# -----------------------------------
coef_names = ["intercept"] + features
coefs = pd.Series(result.x, index=coef_names)

print("Optimization success:", result.success)
print("Message:", result.message)
print("\nEstimated coefficients:")
print(coefs.round(6))