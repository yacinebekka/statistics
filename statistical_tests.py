import csv
import statistics
import math
from scipy.stats import t, f

## Check against R implementation -> OK R.A.S.

def load_csv(filepath: str):
    data = {}  # Initialize an empty dictionary to store the data
    with open(filepath, mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)  # Use DictReader to read the CSV into a dictionary
        headers = csv_reader.fieldnames  # Extract headers from the CSV file
        if headers:
            for header in headers:
                data[header] = []  # Initialize a list for each header
            for row in csv_reader:
                for header in headers:
                    value = row[header]
                    # Attempt to convert numeric values to int or float
                    if value.isdigit():  # Checks if value can be converted to int
                        data[header].append(int(value))
                    else:
                        try:
                            # Try converting to float if possible
                            float_value = float(value)
                            data[header].append(float_value)
                        except ValueError:
                            # Leave as string if it cannot be converted
                            data[header].append(value)
    return data


def t_test_mean_diff_pooled_var(sample_1: list, sample_2: list, two_tail: bool = True, alpha: float=0.05):
	'''
	T-test for difference of mean, with unknown variance but assuming equal variance across samples.

	If one-tail we assume that we have H0 : X_bar_1 <=  X_bar_2
	'''
	x_bar_1 = statistics.mean(sample_1)
	x_bar_2 = statistics.mean(sample_2)
	mean_diff = x_bar_1 - x_bar_2

	sample_var_1 = statistics.variance(sample_1)
	sample_var_2 = statistics.variance(sample_2)

	n1 = len(sample_1)
	n2 = len(sample_2)

	pooled_variance = (((n1 -1) * sample_var_1) + ((n2 -1) * sample_var_2)) / (n1 + n2 - 2)
	se_mean_diff = math.sqrt(pooled_variance * ((1/n1) + (1/n2)))

	t_statistic = mean_diff / se_mean_diff
	df = n1 + n2 - 2

	if two_tail:
		p_value = 2 * t.sf(abs(t_statistic), df)
		t_critical = stats.t.ppf(1 - alpha/2, df)
		margin_of_error = t_critical * se_mean_diff
		lower_confidence_bound = diff_means - margin_of_error
		upper_confidence_bound = diff_means + margin_of_error

	else:
		p_value = t.sf(t_statistic, df)  # One-tailed test: p-value for mean1 > mean2
		t_critical = stats.t.ppf(1 - alpha, df)
		margin_of_error = t_critical * se_mean_diff
		lower_confidence_bound = diff_means - margin_of_error
		upper_confidence_bound = None

	return {
		'x_bar_1': x_bar_1, 
		'x_bar_2': x_bar_2, 
		'mean_diff': mean_diff, 
		'lower_confidence_bound': lower_confidence_bound, 
		'upper_confidence_bound': upper_confidence_bound, 
		'se_mean_diff': se_mean_diff, 
		't_statistic': t_statistic, 
		'p_value': p_value, 
		'df': df
	}


def f_test(sample_1: list, sample_2: list, alpha: float=0.05):
	'''
	F-test for ratio of variance

	By convention for the ratio we put the largest sample variance at the numerator and we reject when the test statistic is above the critical value
	'''
	sample_var_1 = statistics.variance(sample_1)
	sample_var_2 = statistics.variance(sample_2)

	if sample_var_1 >= sample_var_2:
		larger_var = sample_var_1
		smaller_var = sample_var_2
		df1 = len(sample_1) - 1
		df2 = len(sample_2) - 1
	else:
		larger_var = sample_var_2
		smaller_var = sample_var_1
		df1 = len(sample_2) - 1
		df2 = len(sample_1) - 1

	f_statistic = larger_var / smaller_var

    f_upper = f.ppf(1 - alpha/2, df1, df2)
    f_lower = f.ppf(alpha/2, df1, df2)
    upper_confidence_bound = f_statistic / f_upper
    lower_confidence_bound = f_statistic / f_lower

	p_value = f.sf(F_statistic, df1, df2)  # sf is the survival function, equivalent to 1 - cdf

	return {
		'larger_var': larger_var,
		'smaller_var': smaller_var,
		'f_statistic': f_statistic,
		'upper_confidence_bound': upper_confidence_bound,
		'lower_confidence_bound': lower_confidence_bound,
		'p_value': p_value,
		'df1': df1,
		'df2': df2
	}


def levene_test(sample_1: list, sample_2: list, alpha: float=0.05):
	'''
	Levene's test for equal variances usign mean as a centering point	
	'''
	# Calculate means
	x_bar_1 = statistics.mean(sample_1)
	x_bar_2 = statistics.mean(sample_2)

	# Sample sizes
	n1 = len(sample_1)
	n2 = len(sample_2)

	# Calculate Z values (deviations from the mean)
	Z1 = [abs(x - x_bar_1) for x in sample_1]
	Z2 = [abs(x - x_bar_2) for x in sample_2]

	# Calculate the mean of Z values
	Z_bar1 = statistics.mean(Z1)
	Z_bar2 = statistics.mean(Z2)

	# Calculate the overall mean of Z values
	overall_Z_mean = statistics.mean(Z1 + Z2)

	numerator = n1 * (Z_bar1 - overall_Z_mean)**2 + n2 * (Z_bar2 - overall_Z_mean)**2
	denominator = sum((z - Z_bar1)**2 for z in Z1) + sum((z - Z_bar2)**2 for z in Z2)

	# Degrees of freedom
	df_between = 1  # k - 1 = 2 - 1
	df_within = n1 + n2 - 2  # Total n - k

	# Calculate the Levene's statistic
	f_statistic = (numerator / df_between) / (denominator / df_within)

    f_upper = f.ppf(1 - alpha/2, df_between, df_within)
    f_lower = f.ppf(alpha/2, df_between, df_within)
    upper_confidence_bound = f_statistic / f_upper
    lower_confidence_bound = f_statistic / f_lower

	p_value = f.sf(f_statistic, df_between, df_within)

	return {
		'f_statistic': f_statistic,
		'upper_confidence_bound': upper_confidence_bound,
		'lower_confidence_bound': lower_confidence_bound,
		'p_value': p_value,
		'df_between': df_between,
		'df_within': df_within
	}


def welch_t_test(sample_1: list, sample_2: list, two_tail: bool = True, alpha: float=0.05):
	'''
	Welch's t-test for the means of two independent samples, with unknown pop. variance and not equal variance across samples

	If one-tail we assume that we have H0 : X_bar_1 <=  X_bar_2
	'''
	# Calculate means
	x_bar_1 = statistics.mean(sample_1)
	x_bar_2 = statistics.mean(sample_2)
	mean_diff = x_bar_1 - x_bar_2

	# Calculate variances
	sample_var_1 = statistics.variance(sample_1)
	sample_var_2 = statistics.variance(sample_2)

	n1 = len(sample_1)
	n2 = len(sample_2)

	# Calculate Welch's t-statistic
	se_mean_diff = math.sqrt((sample_var_1 / n1) + (sample_var_2 / n2))
	t_statistic = mean_diff / se_mean_diff

	# Calculate the degrees of freedom
	df = ((sample_var_1 / n1) + (sample_var_2 / n2))**2 / (((sample_var_1 / n1)**2 / (n1 - 1)) + ((sample_var_2 / n2)**2 / (n2 - 1)))

	# Calculate the p-value
	if two_tail:
			p_value = 2 * t.sf(abs(t_statistic), df)  # Two-tailed test
			t_critical = stats.t.ppf(1 - alpha/2, df)
			margin_of_error = t_critical * se_mean_diff
			lower_confidence_bound = diff_means - margin_of_error
			upper_confidence_bound = diff_means + margin_of_error
	else:
			p_value = t.sf(t_statistic, df)  # One-tailed test: p-value for mean1 > mean2
			t_critical = stats.t.ppf(1 - alpha, df)
			margin_of_error = t_critical * se_mean_diff
			lower_confidence_bound = diff_means - margin_of_error
			upper_confidence_bound = None

	return {
		'x_bar_1': x_bar_1,
		'x_bar_2': x_bar_2,
		'mean_diff': mean_diff,
		'lower_confidence_bound': lower_confidence_bound,
		'upper_confidence_bound': upper_confidence_bound,
		'se_mean_diff': se_mean_diff,
		't_statistic': t_statistic,
		'p_value': p_value,
		'df': df
	}


def t_test_contrasts(sample_1: list, sample_2: list, sample_variance: float, N: int, two_tail: bool = True, alpha: float=0.05):
	'''
	T-test for linear contrast with two groups in one-way ANOVA.
	sample_1 : Sample data for group 1
	sample_2 : Sample data for group 2
	SS_resid : Residual sum of squares from the ANOVA
	N : Total number of observations across all groups
	two_tail : Whether to perform a two-tailed test. Default is True.

	If one-tail we assume that we have H0 : X_bar_1 <=  X_bar_2

	'''
	# Calculate means
	x_bar_1 = statistics.mean(sample_1)
	x_bar_2 = statistics.mean(sample_2)
	mean_diff = x_bar_1 - x_bar_2

	# Sample sizes
	n1 = len(sample_1)
	n2 = len(sample_2)

	# Degrees of freedom
	J = 2  # Since we are comparing two groups
	df = N - J

	SS_resid = sample_variance * N

	# Calculate the standard error of the contrast
	SE_contrast = math.sqrt((SS_resid / df) * ((1/n1) + (1/n2)))

	# Calculate the t statistic
	t_statistic = mean_diff / SE_contrast

	# Calculate the p-value
	if two_tail:
		p_value = 2 * t.sf(abs(t_statistic), df)
		t_critical = stats.t.ppf(1 - alpha/2, df)
		margin_of_error = t_critical * SE_contrast
		lower_confidence_bound = diff_means - margin_of_error
		upper_confidence_bound = diff_means + margin_of_error
	else:
	    p_value = t.sf(t_statistic, df)  # One-tailed test: p-value for mean4 > mean3
		t_critical = stats.t.ppf(1 - alpha, df)
		margin_of_error = t_critical * SE_contrast
		lower_confidence_bound = diff_means - margin_of_error
		upper_confidence_bound = None

	return {
		'x_bar_1': x_bar_1,
		'x_bar_2': x_bar_2,
		'mean_diff': mean_diff,
		'lower_confidence_bound': lower_confidence_bound,
		'upper_confidence_bound': upper_confidence_bound,
		'sample_variance': sample_variance
		'SE_contrast': SE_contrast,
		't_statistic': t_statistic,
		'p_value': p_value,
		'df': df
	}

def anova_f_test(*samples):
	'''
	ANOVA F-test for equal size sample across group
	'''
	
	combined_data = []
	for sample in samples:
		combined_data.extend(sample)

	grand_mean = statistics.mean(combined_data)
	group_mean = [statistics.mean(sample) for sample in samples]

	sample_size = len(samples[0])

	SS_treatment = np.sum([sample_size * (mean - grand_mean)**2 for mean in group_mean])

	SS_residual = 0
	for index, sample in enumerate(samples):
		for observation in sample:
			SS_residual += (observation - group_mean[index])**2

	df1 = len(samples) - 1
	df2 = (sample_size * len(samples)) - len(samples)

	f_statistic = (SS_treatment / df1) / (SS_residual / df2)

	p_value = f.sf(f_statistic, df1, df2)  # sf is the survival function, equivalent to 1 - cdf

	return {
		'SS_treatment': SS_treatment,
		'SS_residual': SS_residual,
		'f_statistic': f_statistic,
		'p_value': p_value,
		'df1': df1,
		'df2': df2
	}
