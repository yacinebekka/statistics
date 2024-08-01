from scipy.stats import f, ncf, nct, t
import numpy as np

## Check against R implementation -> OK R.A.S

def get_power_f(N, J, alpha, f_squared):
    """
    Calculate the power of an F-test.
    
    N : Total sample size
    J : Number of groups
    alpha : Significance level
    f_squared : Cohen's f-squared, the effect size
    """
    df_between = J - 1
    df_within = N - J
    # Correct non-centrality parameter calculation
    non_centrality = N * f_squared
    
    # Critical F value from the central F-distribution
    f_critical = f.ppf(1 - alpha, df_between, df_within)
    
    # Power calculation using the non-central F-distribution
    power = 1 - ncf.cdf(f_critical, df_between, df_within, non_centrality)

    print(f"df_between: {df_between}, df_within: {df_within}")
    print(f"f_squared: {f_squared}, non_centrality: {non_centrality}")
    print(f"f_critical: {f_critical}, power: {power}")

    return power

def get_power_t(delta, alpha, n, s, two_tail : bool = True):
    """
    Compute power for a two-sample t-test with equal sample sizes and variances.
    delta : minimum difference of mean (effect size)
    alpha : signficicance level
    n : Number of sample in each group (not total)
    s : Pooled SD

    If one-tail we assume that we have H0 : X_bar_1 <=  X_bar_2
    """
    df = 2 * n - 2  # degrees of freedom for two-sample t-test

    if two_tail:
        # Calculate t quantiles for lower and upper critical values
        q_H0_low = t.ppf(alpha / 2, df=df)
        q_H0_high = t.ppf(1 - alpha / 2, df=df)

        # Calculate the noncentrality parameter
        ncp = abs(delta) / (np.sqrt(2) * s / np.sqrt(n))

        # Calculate power using non-central t-distribution
        power = (nct.cdf(q_H0_low, df, ncp) + (1 - nct.cdf(q_H0_high, df, ncp)))

    else:
        q_H0_high = t.ppf(1 - alpha, df=df)
        ncp =(delta  / s) * np.sqrt(n/2)
        power = 1 - nct.cdf(q_H0_high, df, ncp)

    return power

def get_power_welch_t(delta, sd1, sd2, n1, n2, alpha=0.05, two_tail: bool = True):
    """
    Compute power for a two-sample Welch's t-test
    """

    # Calculate the Welch-Satterthwaite degrees of freedom
    df = ((sd1**2 / n1 + sd2**2 / n2)**2) / (((sd1**2 / n1)**2 / (n1 - 1)) + ((sd2**2 / n2)**2 / (n2 - 1)))

    # Non-centrality parameter
    ncp = delta / np.sqrt(sd1**2 / n1 + sd2**2 / n2)

    if two_tail:
        t_critical = t.ppf(1 - alpha / 2, df)
        power = 1 - nct.cdf(t_critical, df, ncp)

    else:
        t_critical = t.ppf(1 - alpha, df)
        power = 1 - nct.cdf(t_critical, df, ncp)

    return power
