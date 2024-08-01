from scipy.stats import f, ncf, nct

## !!! To be reworked and checked against know implementation


def get_power_f(N, J, alpha, f_squared):
    """
    Calculate the power of an F-test.
    
    Parameters:
    N (int): Total sample size
    J (int): Number of groups
    alpha (float): Significance level
    f_squared (float): Cohen's f-squared, the effect size
    
    Returns:
    float: Power of the F-test
    """
    df_between = J - 1
    df_within = N - J
    # Correct non-centrality parameter calculation
    non_centrality = N * f_squared * df_between / J
    
    # Critical F value from the central F-distribution
    f_critical = f.ppf(1 - alpha, df_between, df_within)
    
    # Power calculation using the non-central F-distribution
    power = 1 - ncf.cdf(f_critical, df_between, df_within, non_centrality)

    print(f"df_between: {df_between}, df_within: {df_within}")
    print(f"f_squared: {f_squared}, non_centrality: {non_centrality}")
    print(f"f_critical: {f_critical}, power: {power}")

    return power

def get_power_t(delta, alpha, n, s):
    """
    Compute power for a two-sample t-test with equal sample sizes and variances
    """
    df = 2 * n - 2  # degrees of freedom for two-sample t-test

    # Calculate t quantiles for lower and upper critical values
    q_H0_low = t.ppf(alpha / 2, df=df)
    q_H0_high = t.ppf(1 - alpha / 2, df=df)

    # Calculate the noncentrality parameter
    ncp = abs(delta) / (np.sqrt(2) * s / np.sqrt(n))

    # Calculate power using non-central t-distribution
    power = (nct.cdf(q_H0_low, df, ncp) + (1 - nct.cdf(q_H0_high, df, ncp)))
    
    return power

def get_power_welch_t(alpha=0.05, power=0.90, mu1=0, mu2=0, sigma1=1, sigma2=1, n2n1_ratio=1):
    """
    Compute power for a two-sample Welch's t-test
    """
    delta = mu1 - mu2

    def calculate_power(n1, n2, delta, sigma1, sigma2, alpha):
        s_pooled = np.sqrt((sigma1**2 / n1) + (sigma2**2 / n2))
        d = delta / s_pooled
        df = ((sigma1**2 / n1 + sigma2**2 / n2)**2 /
              ((sigma1**2 / n1)**2 / (n1 - 1) + (sigma2**2 / n2)**2 / (n2 - 1)))

        t_crit = t.ppf(1 - alpha / 2, df)
        # Non-centrality parameter
        non_centrality = d * np.sqrt(n1 * n2 / (n1 + n2))
        # Calculate power using the non-central t-distribution
        power = 2 * (1 - nct.cdf(t_crit, df, non_centrality))

        return power

    n1 = 10  # Initial guess
    n2 = math.ceil(n1 * n2n1_ratio)
    while True:
        calc_power = calculate_power(n1, n2, delta, sigma1, sigma2, alpha)
        if calc_power >= power:
            break
        n1 += 1
        n2 = math.ceil(n1 * n2n1_ratio)

    return n1, n2