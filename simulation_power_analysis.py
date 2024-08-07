import numpy as np
import statistical_test as st


def simulate_t_test_pooled_var_power(mu_1, mu_2, sigma1, sigma2, n, alpha, two_tail, num_simulations):
	'''
	Compute power of a pooled var t-test using simulation

	mu_1 : Mean of sample 1 under H1
	mu_2 : Mean of sample 2 under H1
	sigma1 : Population SD of population represented by sample 1 under H1
	sigma2 : Population SD of population represented by sample 2 under H1

	'''
	rejection = 0

	for _ in range(num_simulations):
		sim_sample_1 = np.random.normal(mu_1, sigma, n)
		sim_sample_2 = np.random.normal(mu_2, sigma, n)
		result = st.t_test_mean_diff_pooled_var(sim_sample_1, sim_sample_2, two_tail=two_tails)

		if result['p_value'] < alpha:
			rejections += 1      

	power = rejections / num_simulations

	return {
		'power' : power,
		'n' : n
	}

def simulate_f_test_power(mu1, mu2, sigma1, sigma2, n, alpha, num_simulations):
	'''
	Compute power of a f-test for ratio of variance using simulation

	mu_1 : Mean of sample 1 under H1
	mu_2 : Mean of sample 2 under H1
	sigma1 : Population SD of population represented by sample 1 under H1
	sigma2 : Population SD of population represented by sample 2 under H1

	'''
	rejection = 0

	for _ in range(num_simulations):
		sim_sample_1 = np.random.normal(mu1, sigma1, n)
		sim_sample_2 = np.random.normal(mu2, sigma2, n)
		result = st.f_test(sim_sample_1, sim_sample_2)

		if result['p_value'] < alpha:
			rejections += 1      

	power = rejections / num_simulations

	return {
		'power' : power,
		'n' : n
	}

def simulate_levene_test_power(group_means, group_sd, n, alpha, num_simulations):
	'''
	Compute power of a Levene test using simulation

	group_means : List of means for each group under H1
	group_sd : Population SD of populations represented by samples under H1

	'''
	rejection = 0

	for _ in range(num_simulations):
		sim_samples = [np.random.normal(mean, sigma, n) for mean, sigma in zip(group_means, group_sd)]

		result = st.levene_test(sim_samples)

		if result['p_value'] < alpha:
			rejections += 1      

	power = rejections / num_simulations

	return {
		'power' : power,
		'n' : n
	}

def simulate_welch_t_test_power(mu_1, mu_2, sigma1, sigma2, n1, n2, alpha, two_tail, num_simulations):
	'''
	Compute power of a Welch t-test using simulation

	mu_1 : Mean of sample 1 under H1
	mu_2 : Mean of sample 2 under H2
	sigma1 : Population SD of population represented by sample 1 under H1
	sigma2 : Population SD of population represented by sample 2 under H1

	'''
	rejection = 0

	for _ in range(num_simulations):
		sim_sample_1 = np.random.normal(mu_1, sigma1, n1)
		sim_sample_2 = np.random.normal(mu_2, sigma2, n2)
		result = st.welch_t_test(sim_sample_1, sim_sample_2, two_tail=two_tails)

		if result['p_value'] < alpha:
			rejections += 1      

	power = rejections / num_simulations

	return {
		'power' : power,
		'n' : n
	}

def simulate_contrast_t_test_power(mu_1, mu_2, sigma, n, alpha, two_tail, num_simulations):
	'''
	Compute power of a one-way anove simple contrast t-test using simulation

	mu_1 : Mean of sample 1 under H1
	mu_2 : Mean of sample 2 under H2
	sigma : Population SD under H1

	'''
	rejection = 0

	for _ in range(num_simulations):
		sim_sample_1 = np.random.normal(mu_1, sigma, n)
		sim_sample_2 = np.random.normal(mu_2, sigma, n)
		result = st.t_test_contrasts(sim_sample_1, sim_sample_2, two_tail=two_tails)

		if result['p_value'] < alpha:
			rejections += 1      

	power = rejections / num_simulations

	return {
		'power' : power,
		'n' : n
	}

def simulate_anova_f_test_power(group_means, sigma, n, alpha, num_simulations):
	'''
	Compute power of a one-way ANOVA f-test using simulation

	group_means : List of means for each group under H1
	sigma : Population SD under H1

	'''
	rejection = 0

	for _ in range(num_simulations):
		sim_samples = [np.random.normal(mean, sigma, n) for mean in group_means]

		result = st.anova_f_test(sim_samples)

		if result['p_value'] < alpha:
			rejections += 1      

	power = rejections / num_simulations

	return {
		'power' : power,
		'n' : n
	}

def simulate_two_way_anova_f_test_power(group_means, sigma, n, alpha, num_simulations):
	'''
	Compute power of a one-way ANOVA f-test using simulation

	group_means : List of means for each group under H1
	sigma : Population SD under H1

	'''
	rejection_factorA = 0
	rejection_factorB = 0
	rejection_interactions = 0
	rejection_reduced = 0

	for _ in range(num_simulations):
		sim_samples = [np.random.normal(mean, sigma, n) for mean in group_means]

		result = st.two_way_anova_f_test(sim_samples)

		if result['factor_a']['p_value'] < alpha:
			rejection_factorA += 1
		if result['factor_b']['p_value'] < alpha:
			rejection_factorB += 1  
		if result['interactions']['p_value'] < alpha:
			rejection_interactions += 1  
		if result['reduced_model']['p_value'] < alpha:
			rejection_reduced += 1  

	power_factor_a = rejection_factorA / num_simulations
	power_factor_b = rejection_factorB / num_simulations
	power_interactions = rejection_interactions / num_simulations
	power_reduced_model = rejection_reduced / num_simulations

	return {
		'power_factor_a' : power_factor_a,
		'power_factor_b' : power_factor_b,
		'power_interactions' : power_interactions,
		'power_reduced_model' : power_reduced_model,
		'n' : n
	}


def simulate_two_way_anova_contrast_t_test(mu1, mu2, sigma, n, v, alpha, two_tail, num_simulations):
	'''
	Compute power of a simple contrasts t-test from two-way ANOVA using simulation

	'''
	rejection = 0

	for _ in range(num_simulations):
		sim_sample_1 = np.random.normal(mu_1, sigma, n)
		sim_sample_2 = np.random.normal(mu_2, sigma, n)

		SS_resid = v * (n - 1)
		SS_resid = 0

		for observation in sim_sample_1:
			SS_resid += (observation - np.mean(sim_sample_1))**2

		for observation in sim_sample_2:
			SS_resid += (observation - np.mean(sim_sample_2))**2

		result = st.two_way_anova_contrast_t_test(mu1, mu2, SS_resid, SS_resid, n, v, two_tail=two_tail, alpha=alpha)

		if result['p_value'] < alpha:
			rejection += 1

	power = rejections / num_simulations

	return {
		'power' : power,
		'n' : n
	}
