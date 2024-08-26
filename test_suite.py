import statistical_tests as st
import power_analysis as pa
import unittest
import numpy as np

## Test against output from R implementation of the tests

class TestSuite(unittest.TestCase):
	def test_welch_t_test(self):
		'''
		Corresponding R code :

		# Two-tail

		# Create sample data for two groups
		group1 <- c(20, 22, 21, 19, 23)
		group2 <- c(28, 30, 27, 29, 26)

		# Perform a two-sample t-test using pooled variance
		test_result <- t.test(group1, group2, var.equal = TRUE)

		# Print the results of the t-test
		print(test_result)

		data:  group1 and group2
		t = -7, df = 8, p-value = 0.0001126
		alternative hypothesis: true difference in means is not equal to 0
		95 percent confidence interval:
		 -9.306004 -4.693996
		sample estimates:
		mean of x mean of y 
		21        28 

		# One-tail

		data:  group1 and group2
		t = -7, df = 8, p-value = 0.9999
		alternative hypothesis: true difference in means is greater than 0
		95 percent confidence interval:
		 -8.859548       Inf
		sample estimates:
		mean of x mean of y 
		       21        28

		'''

		sample1 = [20, 22, 21, 19, 23]
		sample2 = [28, 30, 27, 29, 26]

		result = st.welch_t_test(sample1, sample2, two_tail=True, alpha=0.05)

		self.assertEqual(round(result['x_bar_1'], 0), 21)
		self.assertEqual(round(result['x_bar_2'], 0), 28)
		self.assertEqual(round(result['lower_confidence_bound'], 4) , round(-9.3060004, 4))
		self.assertEqual(round(result['upper_confidence_bound'], 4), round(-4.693996, 4))
		self.assertEqual(round(result['t_statistic'], 0), -7)
		self.assertEqual(round(result['p_value'], 4), round(0.0001126, 4))
		self.assertEqual(round(result['df'], 0), 8)

		result = st.welch_t_test(sample1, sample2, two_tail=False, alpha=0.05)

		self.assertEqual(round(result['x_bar_1'], 0), 21)
		self.assertEqual(round(result['x_bar_2'], 0), 28)
		self.assertEqual(round(result['lower_confidence_bound'], 4), round(-8.859548, 4))
		self.assertEqual(result['upper_confidence_bound'], None)
		self.assertEqual(round(result['t_statistic'], 0), -7)
		self.assertEqual(round(result['p_value'], 4), round(0.9999, 4))
		self.assertEqual(round(result['df'], 0), 8)

	def test_f_test(self):
		'''
		# Create sample data for two groups
		group2 <- c(12, 22, 21, 19, 23)
		group1 <- c(28, 30, 27, 29, 42)

		# Perform a two-sample t-test using pooled variance
		test_result <- var.test(group1, group2, alternative="two.sided")

		# Print the results of the t-test
		print(test_result)

		F test to compare two variances

		data:  group1 and group2
		F = 1.9534, num df = 4, denom df = 4, p-value = 0.5326
		alternative hypothesis: true ratio of variances is not equal to 1
		95 percent confidence interval:
		  0.2033799 18.7611801
		sample estimates:
		ratio of variances 
		          1.953368 
		'''
		sample1 = [12, 22, 21, 19, 23]
		sample2 = [28, 30, 27, 29, 42]

		result = st.f_test(sample1, sample2,  alpha=0.05)

		self.assertEqual(round(result['f_statistic'], 4), 1.9534)
		self.assertEqual(round(result['upper_confidence_bound'], 4) , round(0.2033799, 4))
		self.assertEqual(round(result['lower_confidence_bound'], 4), round(18.7611801, 4))
		self.assertEqual(round(result['p_value'], 4), 0.5326, 4)
		self.assertEqual(round(result['df1'], 0), 4)
		self.assertEqual(round(result['df2'], 0), 4)


	def test_levene_test(self):
		'''
		if (!require(car)) {
			install.packages("car")
			library(car)
		}
				
		sample1 = c(12, 22, 21, 19, 23)
		sample2 = c(28, 30, 27, 29, 42)
		sample3 = c(12, 25, 30, 29, 48)

		data <- data.frame(
			values = c(sample1 , sample2 , sample3 ),
			group = factor(rep(c("Group1", "Group2", "Group3"), each = 50))
		)

		levene_result <- leveneTest(values ~ group, data = data, center = mean)
		print(levene_result)

		Levene's Test for Homogeneity of Variance (center = mean)
		       Df F value Pr(>F)
		group   2  0.0659 0.9363
		      147     

		'''
		sample1 = [12, 22, 21, 19, 23]
		sample2 = [28, 30, 27, 29, 42]
		sample3 = [12, 25, 30, 29, 48]

		result = st.levene_test(sample1, sample2, sample3,  alpha=0.05)

		self.assertEqual(round(result['f_statistic'], 4), 1.0408)
		self.assertEqual(round(result['p_value'], 3), 0.383)
		self.assertEqual(round(result['df_between'], 0), 2)
		self.assertEqual(round(result['df_within'], 0), 12)

	def test_anova_f_test(self):
		'''
		sample1 = c(12, 22, 21, 19, 23)
		sample2 = c(28, 30, 27, 29, 42)
		sample3 = c(12, 25, 30, 29, 48)

		data <- data.frame(
		values = c(sample1 , sample2 , sample3 ),
		group = factor(rep(c("Group1", "Group2", "Group3"), each = 5))
		)

		# Fit the ANOVA model using aov()
		anova_model <- aov(values ~ group, data = data)

		# Display the summary ANOVA table
		summary(anova_model)

		             Df Sum Sq Mean Sq F value Pr(>F)
		group         2     39   19.45   0.223    0.8
		Residuals   147  12798   87.06     
	
		'''
		sample1 = [12, 22, 21, 19, 23]
		sample2 = [28, 30, 27, 29, 42]
		sample3 = [12, 25, 30, 29, 48]

		result = st.anova_f_test(sample1, sample2, sample3)

		self.assertEqual(round(result['f_statistic'], 3), 2.608)
		self.assertEqual(round(result['p_value'], 3), 0.115)
		self.assertEqual(round(result['df1'], 0), 2)
		self.assertEqual(round(result['df2'], 0), 12)
		self.assertEqual(round(result['SS_treatment'], 1), 388.9)
		self.assertEqual(round(result['SS_residual'], 1), 894.8)

	def test_two_way_anova_f_test(self):
		'''
		scores = c(12, 22, 21, 19, 23,
		28, 30, 27, 29, 42,
		12, 25, 30, 29, 48,
		14, 13, 15, 13, 17)

		treatment <- factor(rep(c("Treatment1", "Treatment2"), each=5))
		gender <- factor(rep(c("Male", "Female"), times=5))

		data <- data.frame(scores, treatment, gender)

		anova_model <- aov(scores ~ treatment * gender, data = data)

		# Display the ANOVA table
		summary(anova_model)

				                 Df Sum Sq Mean Sq F value Pr(>F)
		treatment         1    8.4    8.45   0.074  0.788
		gender            1    4.8    4.80   0.042  0.840
		treatment:gender  1   12.0   12.03   0.106  0.749
		Residuals        16 1815.7  113.48       
		'''
		data = [
			(12, 'Treatment1', 'Male'),
			(22, 'Treatment1', 'Female'),
			(21, 'Treatment1', 'Male'),
			(19, 'Treatment1', 'Female'),
			(23, 'Treatment1', 'Male'),
			(28, 'Treatment2', 'Female'),
			(30, 'Treatment2', 'Male'),
			(27, 'Treatment2', 'Female'),
			(29, 'Treatment2', 'Male'),
			(42, 'Treatment2', 'Female'),
			(12, 'Treatment1', 'Male'),
			(25, 'Treatment1', 'Female'),
			(30, 'Treatment1', 'Male'),
			(29, 'Treatment1', 'Female'),
			(48, 'Treatment1', 'Male'),
			(14, 'Treatment2', 'Female'),
			(13, 'Treatment2', 'Male'),
			(15, 'Treatment2', 'Female'),
			(13, 'Treatment2', 'Male'),
			(17, 'Treatment2', 'Female')
		]

		# Define the data type for each field
		dtype = [('Response', 'f8'), ('FactorA', 'U10'), ('FactorB', 'U10')]
		structured_array = np.array(data, dtype=dtype)

		result = st.two_way_anova_f_test(structured_array)

		self.assertEqual(round(result['factor_a']['f_statistic'], 3), 0.074)
		self.assertEqual(round(result['factor_a']['p_value'], 3), 0.788)
		self.assertEqual(round(result['factor_a']['df'], 0), 1)
		self.assertEqual(round(result['factor_a']['SS'], 2), 8.45)

		self.assertEqual(round(result['factor_b']['f_statistic'], 3), 0.042)
		self.assertEqual(round(result['factor_b']['p_value'], 3), 0.840)
		self.assertEqual(round(result['factor_b']['df'], 0), 1)
		self.assertEqual(round(result['factor_b']['SS'], 1), 4.8)

		self.assertEqual(round(result['interactions']['f_statistic'], 3), 0.106)
		self.assertEqual(round(result['interactions']['p_value'], 3), 0.749)
		self.assertEqual(round(result['interactions']['df'], 0), 1)
		self.assertEqual(round(result['interactions']['SS'], 2), 12.03)		

		self.assertEqual(round(result['residuals']['SS'], 1), 1815.7)
		self.assertEqual(round(result['residuals']['df'], 0), 16)	

		self.assertEqual(round(result['reduced_model']['SS_resid'], 1), 1827.7)
		self.assertEqual(round(result['reduced_model']['df_resid'], 0), 17)	
		self.assertEqual(round(result['reduced_model']['p_value'], 4), 0.7489)	
		self.assertEqual(round(result['reduced_model']['f_statistic'], 3), 0.106)	

	# 	# Analytical power calculation functions

	def test_t_test_power(self):
		'''
		sample1 <- c(10, 12, 10, 13, 14, 10, 11, 12, 14, 13)
		sample2 <- c(14, 15, 15, 16, 17, 15, 16, 17, 16, 15)

		mean1 <- mean(sample1)
		mean2 <- mean(sample2)
		sd1 <- sd(sample1)
		sd2 <- sd(sample2)
		n1 <- length(sample1)
		n2 <- length(sample2)

		pooled_sd <- sqrt(((n1 - 1) * sd1^2 + (n2 - 1) * sd2^2) / (n1 + n2 - 2))

		power_calculation <- power.t.test(n = n1, delta = mean2 - mean1, sd = pooled_sd, sig.level = 0.05, type = "two.sample", alternative = "two.sided")

	    Two-sample t test power calculation 

	              n = 10
	          delta = 1
	             sd = 1.791957
	      sig.level = 0.05
	          power = 0.2182778
	    alternative = two.sided

	    power_calculation <- power.t.test(n = n1, delta = mean2 - mean1, sd = pooled_sd, sig.level = 0.05, type = "two.sample", alternative = "one.sided")

	    Two-sample t test power calculation 

	              n = 10
	          delta = 1
	             sd = 1.791957
	      sig.level = 0.05
	          power = 0.3285556
	    alternative = one.sided 
		'''
		sample1 = [10, 12, 10, 13, 14, 10, 11, 12, 14, 13]
		sample2 = [14, 12, 13, 14, 16, 12, 11, 14, 14, 9]
		n1 = len(sample1)
		n2 = len(sample2)

		mean1 = np.mean(sample1)
		mean2 = np.mean(sample2)
		sample_sd1 = np.std(sample1, ddof=1)
		sample_sd2 = np.std(sample2, ddof=1)
		pooled_sd = np.sqrt(((n1 - 1) * sample_sd1**2 + (n2 - 1) * sample_sd2**2) / (n1 + n2 - 2))

		result = pa.get_power_t((mean2 - mean1), 0.05, 10, pooled_sd, two_tail=True)
		self.assertEqual(round(result['power'], 2), 0.22) # Not exact match starting at 3 decimals

		result = pa.get_power_t((mean2 - mean1), 0.05, 10, pooled_sd, two_tail=False)
		self.assertEqual(round(result['power'], 4), 0.3286)


	def test_welch_t_test_power(self):
		'''
		library(MKpower)

		sample1 <- c(10, 12, 10, 13, 14, 10, 11, 12, 14, 13)
		sample2 <- c(14, 12, 13, 14, 16, 12, 11, 14, 14, 9)

		mean1 <- mean(sample1)
		mean2 <- mean(sample2)
		sd1 <- sd(sample1)
		sd2 <- sd(sample2)
		n1 <- length(sample1)
		n2 <- length(sample2)

		pooled_sd <- sqrt(((n1 - 1) * sd1^2 + (n2 - 1) * sd2^2) / (n1 + n2 - 2))
		d <- (mean1 - mean2) / pooled_sd

		power_calculation = power.welch.t.test(n = 10, delta = (mean2 - mean1), sd1 = sd1, sd2 = sd2, sig.level = 0.05, alternative = "two.sided")

	              n = 10
	          delta = 1
	            sd1 = 1.595131
	            sd2 = 1.969207
	      sig.level = 0.05
	          power = 0.2174527
	    alternative = two.sided

	    power_calculation = power.welch.t.test(n = 10, delta = (mean2 - mean1), sd1 = sd1, sd2 = sd2, sig.level = 0.05, alternative = "two.sided")

	    Two-sample Welch t test power calculation 

	              n = 10
	          delta = 1
	            sd1 = 1.595131
	            sd2 = 1.969207
	      sig.level = 0.05
	          power = 0.3278267
	    alternative = one.sided
		'''

		sample1 = [10, 12, 10, 13, 14, 10, 11, 12, 14, 13]
		sample2 = [14, 12, 13, 14, 16, 12, 11, 14, 14, 9]
		n1 = len(sample1)
		n2 = len(sample2)

		mean1 = np.mean(sample1)
		mean2 = np.mean(sample2)
		sample_sd1 = np.std(sample1, ddof=1)
		sample_sd2 = np.std(sample2, ddof=1)
		pooled_sd = np.sqrt(((n1 - 1) * sample_sd1**2 + (n2 - 1) * sample_sd2**2) / (n1 + n2 - 2))

		result = pa.get_power_welch_t((mean2 - mean1), sample_sd1, sample_sd2, 10, 10, 0.05, two_tail=True)
		self.assertEqual(round(result['power'], 3), 0.218)

		result = pa.get_power_welch_t((mean2 - mean1), sample_sd1, sample_sd2, 10, 10, 0.05, two_tail=False)
		self.assertEqual(round(result['power'], 4), 0.3278)


	def test_anova_f_test_power(self):
		'''
		# Install and load the necessary package
		if (!require(pwr)) install.packages("pwr")
		library(pwr)

		# Define the data arrays
		sample1 <- c(113.70, 94.35, 103.63, 106.32, 104.04, 98.93, 115.11, 99.05, 120.18, 99.37)
		sample2 <- c(118.04, 127.86, 91.11, 102.21, 103.66, 111.35, 102.15, 78.43, 80.59, 118.20)
		sample3 <- c(106.93, 92.18, 108.28, 122.14, 128.95, 105.69, 107.42, 92.36, 114.60, 103.60)

		data <- c(sample1, sample2, sample3)
		group <- factor(rep(c("Group1", "Group2", "Group3"), each = 10))

		# Calculate grand mean and deviations for each group
		grand_mean <- mean(data)
		group_means <- tapply(data, group, mean)
		deltas <- group_means - grand_mean

		# Calculate total variance
		total_variance <- var(data)

		# Calculate sum of squared deviations and f-squared
		sum_delta_squared <- sum(10 * deltas^2)  # Multiply by n (10) in each group
		f_squared <- sum_delta_squared / (length(data) * total_variance)

		# Degrees of freedom
		df_between <- length(group_means) - 1
		df_within <- length(data) - length(group_means)

		# Calculate power using the pwr.f2.test function
		power_result <- pwr.f2.test(u = df_between, v = df_within, f2 = f_squared, sig.level = 0.05)

		# Print results
		print(power_result)

	     Multiple regression power calculation 

	              u = 2
	              v = 27
	             f2 = 0.02638736
	      sig.level = 0.05
	          power = 0.1072853

		'''

		sample1 = [113.70, 94.35, 103.63, 106.32, 104.04, 98.93, 115.11, 99.05, 120.18, 99.37]
		sample2 = [118.04, 127.86, 91.11, 102.21, 103.66, 111.35, 102.15, 78.43, 80.59, 118.20]
		sample3 = [106.93, 92.18, 108.28, 122.14, 128.95, 105.69, 107.42, 92.36, 114.60, 103.60]

		grand_mean = np.mean([np.mean(sample1), np.mean(sample2), np.mean(sample3)])
		deltas = [np.mean(sample1) - grand_mean, np.mean(sample2) - grand_mean, np.mean(sample3) - grand_mean]
		all_data = np.concatenate([sample1, sample2, sample3])
		sample_variance = np.var(all_data, ddof=1)

		result = pa.get_power_anova_f_test(deltas, sample_variance, 3, 30, alpha=0.05)
		self.assertEqual(round(result['power'], 4), 0.1073)


if __name__ == '__main__':
    unittest.main()
