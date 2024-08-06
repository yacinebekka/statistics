from scipy.stats import t, f
import numpy as np

## Check against R implementation -> OK R.A.S.

def t_test_pooled_var(sample_1: np.array, sample_2: np.array, two_tail: bool = True, alpha: float=0.05):
	'''
	T-test for difference of mean, with unknown variance but assuming equal variance across samples.

	If one-tail we assume that we have H0 : X_bar_1 <=  X_bar_2
	'''
	x_bar_1 = np.mean(sample_1)
	x_bar_2 = np.mean(sample_2)
	mean_diff = x_bar_1 - x_bar_2

	sample_var_1 = np.var(sample_1, ddof=1)
	sample_var_2 = np.var(sample_2, ddof=1)

	n1 = len(sample_1)
	n2 = len(sample_2)

	pooled_variance = (((n1 -1) * sample_var_1) + ((n2 -1) * sample_var_2)) / (n1 + n2 - 2)
	se_mean_diff = np.sqrt(pooled_variance * ((1/n1) + (1/n2)))

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


def f_test(sample_1: np.array, sample_2: np.array, alpha: float=0.05):
	'''
	F-test for ratio of variance

	By convention for the ratio we put the largest sample variance at the numerator and we reject when the test statistic is above the critical value
	'''
	sample_var_1 = np.var(sample_1, ddof=1)
	sample_var_2 = np.var(sample_2, ddof=1)

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


def levene_test(*samples: np.array, alpha: float=0.05):
	'''
	Levene's test for equal variances usign mean as a centering point	
	'''
	k = len(samples)

	# Calculate means
	group_means = [np.mean(sample) in samples]
	grand_mean = np.mean([observation for sample in samples for observation in sample])

	# Sample sizes
	n_i = [len(sample) for sample in samples]
	N = sum(n_i)

	# Calculate Z values (deviations from the mean)
	Z = [[abs(observation - group_mean[i]) for observation in sample] for i, sample in enumerate(samples)]

	# Calculate the mean of Z values
	Z_bar = [np.mean(z) for z in Z]

	# Calculate the numerator (between-group variability)
	SS_between = np.sum([ni[i] * (Z_bar[i] - overall_median)**2 for i in range(k)])
	
	# Calculate the denominator (within-group variability)
	SS_within = np.sum([np.sum([(z - Z_bar[i])**2 for z in Z[i]]) for i in range(k)])

	# Degrees of freedom
	df_between = k - 1
	df_within = N - k

	# Calculate the Levene's statistic
	f_statistic = (SS_between / df_between) / (SS_within / df_within)

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


def welch_t_test(sample_1: np.array, sample_2: np.array, two_tail: bool = True, alpha: float=0.05):
	'''
	Welch's t-test for the means of two independent samples, with unknown pop. variance and not equal variance across samples

	If one-tail we assume that we have H0 : X_bar_1 <=  X_bar_2
	'''
	# Calculate means
	x_bar_1 = np.mean(sample_1)
	x_bar_2 = np.mean(sample_2)
	mean_diff = x_bar_1 - x_bar_2

	# Calculate variances
	sample_var_1 = np.var(sample_1, ddof=1)
	sample_var_2 = np.var(sample_2, ddof=1)

	n1 = len(sample_1)
	n2 = len(sample_2)

	# Calculate Welch's t-statistic
	se_mean_diff = np.sqrt((sample_var_1 / n1) + (sample_var_2 / n2))
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


def t_test_contrasts(sample_1: np.array, sample_2: np.array, two_tail: bool = True, alpha: float=0.05):
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
	x_bar_1 = np.mean(sample_1)
	x_bar_2 = np.mean(sample_2)
	mean_diff = x_bar_1 - x_bar_2

	sample_variance = np.var(sample_1, ddof=1) # We assume equal variance across sample

	# Sample sizes
	n1 = len(sample_1)
	n2 = len(sample_2)
	N = n1 + n2

	# Degrees of freedom
	J = 2  # Since we are comparing two groups
	df = N - J

	SS_resid = sample_variance * N

	# Calculate the standard error of the contrast
	SE_contrast = np.sqrt((SS_resid / df) * ((1/n1) + (1/n2)))

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

def anova_f_test(*samples: np.array):
	'''
	ANOVA F-test assume equal size sample across group
	'''
	
	combined_data = []
	for sample in samples:
		combined_data.extend(sample)

	grand_mean = np.mean(combined_data)
	group_mean = [np.mean(sample) for sample in samples]

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


def two_way_anova_f_test(sample_array: np.array):
	'''
	Two-way ANOVA F-test for main effect, assume equal sample size for all group
	
	Expect a Numpy 2-D array as an input such that :

	array([('A1', 'B1', 20.), ('A1', 'B1', 21.),
	       ('A1', 'B2', 19.), ('A2', 'B1', 18.),
	       ('A2', 'B2', 17.), ('A2', 'B2', 15.),
	       ('A1', 'B1', 22.), ('A1', 'B1', 20.),
	       ('A1', 'B2', 21.), ('A2', 'B1', 15.),
	       ('A2', 'B2', 14.), ('A2', 'B2', 15.)],
	      dtype=[('FactorA', '<U10'), ('FactorB', '<U10'), ('Response', '<f8')])

	'''
	factor_a_levels = np.unique(sample_array['FactorA'])
	factor_b_levels = np.unique(sample_array['FactorB'])

	J = len(factor_a_levels)
	K = len(factor_b_levels)

	group_means = {}
	grand_total = 0
	sample_sizes = []
	main_effects_means = {}
	interactions_means = {}

	for a in factor_a_levels:
		for b in factor_b_levels:
			filter_mask = (sample_array['FactorA'] == a) & (sample_array['FactorB'] == b)
			group_data = sample_array[filter_mask]['Response']
			mean = np.mean(group_data)
			group_means[(a, b)] = mean
			grand_total += np.sum(group_data)
			sample_sizes.append(len(group_data))

	N = np.sum(sample_sizes)
	n = N / (J * K)

	grand_mean = grand_total / N

	df_A = J - 1  # Degrees of freedom for Factor A
	df_B = K - 1  # Degrees of freedom for Factor B
	df_interaction = df_A * df_B  # Interaction term
	df_total = N - 1  # Total degrees of freedom
	df_resid = df_total - df_A - df_B - df_interaction  # Residual degrees of freedom
	df_resid_reduced = df_total - df_A - df_B

	SS_A = 0
	for a in factor_a_levels:
		filter_mask = (sample_array['FactorA'] == a)
		responses = sample_array['Response'][filter_mask]
		SS_A += n * (np.mean(responses) - grand_mean)**2
		main_effects_means[a] = np.mean(responses)

	SS_B = 0
	for b in factor_b_levels:
		filter_mask = (sample_array['FactorB'] == b)
		responses = sample_array['Response'][filter_mask]
		SS_B += n * (np.mean(responses) - grand_mean)**2
		main_effects_means[b] = np.mean(responses)

	SS_interaction = 0
	for a in factor_a_levels:
		for b in factor_b_levels:
			filter_mask_ab = (sample_array['FactorA'] == a) & (sample_array['FactorB'] == b)
			filter_mask_a = (sample_array['FactorA'] == a)
			filter_mask_b = (sample_array['FactorB'] == b)

			group_data = sample_array['Response'][filter_mask_ab]
			factor_a_data = sample_array['Response'][filter_mask_a]
			factor_b_data = sample_array['Response'][filter_mask_b]

			SS_interaction += n * (np.mean(group_data) - np.mean(factor_a_data) - np.mean(factor_b_data) + grand_mean)**2

	SS_resid = 0
	SS_resid_reduced = 0

	for observation in sample_array:
		a = observation['FactorA']
		b = observation['FactorB']
		response = observation['Response']

		filter_mask_ab = (sample_array['FactorA'] == a) & (sample_array['FactorB'] == b)
		group_data = sample_array['Response'][filter_mask_ab]

		predicted_full = group_means[(a, b)]
		predicted_reduced = main_effects_means_a[a] + main_effects_means_b[b] - grand_mean

		SS_resid += (observation - predicted_full)**2
		SS_resid_reduced += (observation - predicted_reduced)**2

	f_statistic_A = SS_A / df_A / SS_resid / df_resid
	f_statistic_B = SS_B / df_B / SS_resid / df_resid
	f_statistic_interaction = SS_interaction / df_interaction / SS_resid / df_resid
	f_statistic_reduced = (SS_resid_reduced - SS_resid) / (df_resid - df_resid_reduced) / SS_resid / df_resid

	p_value_A = f.sf(f_statistic_A, df_A, df_resid)  # sf is the survival function, equivalent to 1 - cdf
	p_value_B = f.sf(f_statistic_B, df_B, df_resid) 
	p_value_interaction = f.sf(f_statistic_interaction, df_interaction, df_resid)
	p_value_reduced = f.sf(f_statistic_reduced, df_resid, df_resid - df_resid_reduced)

	return {
		'factor_a' : {
			'SS' : SS_A,
			'f_statistic' : f_statistic_A,
			'p_value' : p_value_A,
			'df' : df_A
		},
		'factor_b' : {
			'SS' : SS_B,
			'f_statistic' : f_statistic_B,
			'p_value' : p_value_B,
			'df' : df_B
		},
		'interactions' : {
			'SS' : SS_interaction,
			'f_statistic' : f_statistic_interaction,
			'p_value' : p_value_interaction,
			'df' : df_interaction
		},
		'residuals' : {
			'SS' : SS_resid,
			'df' : df_resid
		},
		'reduced_model' : {
			'SS_resid' : SS_resid_reduced,
			'f_statistic' : f_statistic_reduced,
			'p_value' : p_value_reduced,
			'df_resid' : df_resid_reduced,
		},
		'descriptive' : {
			'grand_mean' : grand_mean,
			'group_mean' : group_mean,
			'main_effects_means' : main_effects_means,
			'interactions_means' : interactions_means
		}
	}

def two_way_anova_contrast_t_test(x_bar_1, x_bar_2, SS_resid, df_resid, n, v, two_tail: bool = True, alpha: float=0.05):
	'''
	T-test for linear contrast with two groups in one-way ANOVA.

	n = sample size per group (assuming equal sample size across all groups)
	v = number of group inside contrasts

	If one-tail we assume that we have H0 : X_bar_1 <=  X_bar_2

	'''
	# Calculate means
	mean_diff = x_bar_1 - x_bar_2

	# Calculate the standard error of the contrast
	SE_contrast = np.sqrt((SS_resid / df_resid) * ((1/n) * v))

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


## Add references
