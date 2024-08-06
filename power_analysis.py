from scipy.stats import f, ncf, nct, t
import numpy as np

## Check against R implementation -> OK R.A.S

def get_power_t(delta: float, alpha: float, n: int, s: float, two_tail : bool = True):
    """
    Compute power for a two-sample t-test with equal sample sizes and variances.
    delta : minimum difference of mean (effect size)
    alpha : signficicance level
    n : Number of sample in each group (not total)
    s : Pooled SD

    If one-tail we assume that we have H0 : X_bar_1 <=  X_bar_2
    """
    df = 2 * n - 2  # degrees of freedom for two-sample t-test
    ncp = abs(delta) / (s * np.sqrt(2/n))

    if two_tail:
        t_critical_upper = t.ppf(alpha / 2, df=df)
        t_critical_lower = t.ppf(1 - alpha / 2, df=df)
        power = nct.cdf(t_critical_lower, df, ncp) + (1 - nct.cdf(t_critical_upper, df, ncp))

    else:
        t_critical_upper = t.ppf(1 - alpha, df=df)
        t_critical_lower = None
        power = 1 - nct.cdf(t_critical_upper, df, ncp)

    return {
        'delta': delta,
        'alpha': alpha,
        'n': n,
        'ncp': ncp,
        'df': df,
        't_critical_upper': t_critical_upper,
        't_critical_lower': t_critical_upper,  # This seems to be a typo. It should probably be t_critical_lower if it exists.
        'power': power
    }


def get_power_welch_t(delta: float, sd1: float, sd2: float, n1: int, n2: int, alpha=0.05, two_tail: bool = True):
    """
    Compute power for a two-sample Welch's t-test
    delta : minimum difference of mean (effect size)
    alpha : significance level
    n1 : Number of observations in sample 1
    n2 : Number of observations in sample 2
    sd1 : Sample standard deviation for sample 1    
    sd2 : Sample standard deviation for sample 2    

    If one-tail we assume that we have H0 : X_bar_1 <=  X_bar_2
    """

    # Calculate the Welch-Satterthwaite degrees of freedom
    df = ((sd1**2 / n1 + sd2**2 / n2)**2) / (((sd1**2 / n1)**2 / (n1 - 1)) + ((sd2**2 / n2)**2 / (n2 - 1)))

    # Non-centrality parameter
    ncp = delta / np.sqrt(sd1**2 / n1 + sd2**2 / n2)

    if two_tail:
        t_critical_upper = t.ppf(1 - alpha / 2, df)
        t_critical_lower = t.ppf(alpha / 2, df)
        power = 1 - nct.cdf(t_critical_upper, df, ncp) + nct.cdf(t_critical_lower, df, non_centrality)

    else:
        t_critical_upper = t.ppf(1 - alpha, df)
        t_critical_lower = None
        power = 1 - nct.cdf(t_critical_upper, df, ncp)

    return {
        'delta': delta,
        'alpha': alpha,
        'n1': n1,
        'n2': n2,
        'ncp': ncp,
        'df': df,
        't_critical_upper': t_critical_upper,
        't_critical_lower': t_critical_lower,
        'power': power
    }


def get_power_contrast_t_test(delta: float, alpha: float, sample_variance: float, N: int, J: int, two_tail: bool = True):
    """
    Calculate the power for a linear contrast t-test
    delta : minimum difference of mean (effect size)
    alpha : significance level
    sample_variance : Sample variance
    N : Total number of observations
    J : Number of groups

    If one-tail we assume that we have H0 : X_bar_1 <=  X_bar_2
    """
    df = N - J  # Degrees of freedom within groups

    n = N / J

    SS_resid = sample_variance * N

    variance_contrast = (SS_resid / df) * (1/n + 1/n)  # Variance of the contrast
    SE_gamma = np.sqrt(variance_contrast)  # Standard error for the contrast
    ncp = delta / SE_gamma  # ncp parameter

    if two_tail:
        t_critical_upper = t.ppf(1 - alpha / 2, df)
        t_critical_lower = t.ppf(alpha / 2, df)
        power = 1 - nct.cdf(t_critical_upper, df, ncp) + nct.cdf(t_critical_lower, df, ncp)
    else:
        t_critical_upper = t.ppf(1 - alpha, df)
        power = 1 - nct.cdf(t_critical_upper, df, ncp)

    return {
        'delta': delta,
        'alpha': alpha,
        'sample_variance': sample_variance,
        'n': n,
        'ncp': ncp,
        'df': df,
        't_critical_upper': t_critical_upper,
        't_critical_lower': t_critical_lower,
        'power': power
    }


def get_power_anova_f_test(deltas: float, sample_variance: float, J: int, N: int, alpha=0.05):
    '''
    Calculate power for ANOVA F-test with equal sample size across groups

    deltas : List of deviations (delta) of group means from the grand mean (alpha_i for each group)
    sample_variance : Sample variance
    J : Number of groups
    N : Total number of observations
    '''

    n = N / J

    sum_delta_squared = sum(deltas**2 for delta in deltas)
    f2 = sum_delta_squared / (J * sample_variance)
    ncp = J * n * f2

    df1 = J - 1
    df2 = (J * n) - J

    f_critical = f.ppf(1 - alpha, df1, df2)
    
    power = 1 - ncf.cdf(f_critical, df1, df2, ncp)

    return {
        'deltas': deltas,
        'sample_variance': sample_variance,
        'f2': f2,
        'n': n,
        'ncp': ncp,
        'df1': df1,
        'df2': df2,
        'f_critical': f_critical,
        'power': power
    }


def get_power_two_way_anova_main_effect_f_test():
    pass


def get_power_two_way_anova_interaction_term_f_test_():
    pass


def get_power_two_way_anova_contrast_t_test():
    pass


def get_power_full_vs_reduced_model_f_test():
    pass



## Add references
