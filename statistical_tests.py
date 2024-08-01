import csv
import statistics
import math
from scipy.stats import t, f

## !!! To be reworked and checked against know implementation


def load_csv(filepath):
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

def t_test_mean_diff_pooled_var(sample_1: list, sample_2: list, two_tail: bool = True):
	'''
	T-test for difference of mean, with unknown variance but asusming equal variance across samples.

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

	T_statistic = mean_diff / se_mean_diff
	df = n1 + n2 - 2

	if two_tail:
		p_value = 2 * t.sf(abs(T_statistic), df)
	else:
		p_value = t.sf(T_statistic, df)  # One-tailed test: p-value for mean1 > mean2

	return (x_bar_1, x_bar_2, mean_diff, se_mean_diff, T_statistic, p_value, df)

def f_test(sample_1: list, sample_2: list):
	'''
	F-test for difference of variance

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

	F_statistic = larger_var / smaller_var

	print(larger_var)
	print(smaller_var)

	p_value = f.sf(F_statistic, df1, df2)  # sf is the survival function, equivalent to 1 - cdf

	return (sample_var_1, sample_var_2, F_statistic, p_value, df1, df2)

def levene_test(sample_1: list, sample_2: list):
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
	W_statistic = (numerator / df_between) / (denominator / df_within)

	p_value = f.sf(W_statistic, df_between, df_within)

	return (W_statistic, p_value, df_between, df_within)


def welch_t_test(sample_1, sample_2, two_tail: bool = True):
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
	T_statistic = (x_bar_1 - x_bar_2) / math.sqrt((sample_var_1 / n1) + (sample_var_2 / n2))

	# Calculate the degrees of freedom
	df = ((sample_var_1 / n1) + (sample_var_2 / n2))**2 / (((sample_var_1 / n1)**2 / (n1 - 1)) + ((sample_var_2 / n2)**2 / (n2 - 1)))

	# Calculate the p-value
	if two_tail:
			p_value = 2 * t.sf(abs(T_statistic), df)  # Two-tailed test
	else:
			p_value = t.sf(T_statistic, df)  # One-tailed test: p-value for mean1 > mean2

	return x_bar_1, x_bar_2, mean_diff, T_statistic, p_value, df