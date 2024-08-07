import statistical_tests as st
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

		self.assertEqual(round(result['x_bar_1'], 4), 21)
		self.assertEqual(round(result['x_bar_2'], 4), 28)
		self.assertEqual(round(result['lower_confidence_bound'], 4) , round(-9.3060004, 4))
		self.assertEqual(round(result['upper_confidence_bound'], 4), round(-4.693996, 4))
		self.assertEqual(round(result['t_statistic'], 4), -7)
		self.assertEqual(round(result['p_value'], 4), round(0.0001126, 4))
		self.assertEqual(round(result['df'], 4), 8)

		result = st.welch_t_test(sample1, sample2, two_tail=False, alpha=0.05)

		self.assertEqual(round(result['x_bar_1'], 4), 21)
		self.assertEqual(round(result['x_bar_2'], 4), 28)
		self.assertEqual(round(result['lower_confidence_bound'], 4), round(-8.859548, 4))
		self.assertEqual(result['upper_confidence_bound'], None)
		self.assertEqual(round(result['t_statistic'], 4), -7)
		self.assertEqual(round(result['p_value'], 4), round(0.9999, 4))
		self.assertEqual(round(result['df'], 4), 8)

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
		self.assertEqual(round(result['p_value'], 4), round(0.5326, 4))
		self.assertEqual(round(result['df1'], 4), 4)
		self.assertEqual(round(result['df2'], 4), 4)


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
		self.assertEqual(round(result['df_between'], 4), 2)
		self.assertEqual(round(result['df_within'], 4), 12)

	# def test_t_test_contrasts(self):
		#pass

	def anova_f_test(self):
		pass

	def two_way_anova_f_test(self):
		pass

	def two_way_anova_f_test(self):
		pass


if __name__ == '__main__':
    unittest.main()
