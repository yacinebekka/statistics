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

	# def test_t_test_contrasts(self):
		#pass

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


if __name__ == '__main__':
    unittest.main()
