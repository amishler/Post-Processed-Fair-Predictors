import copy
from counterfactualEO import *
import pandas as pd
import pickle
import unittest

class TestEstimation(unittest.TestCase):
    data = pd.read_csv('test_data.csv')
    with open('test_values1.pickle', 'rb') as file_in:
        testdict = pickle.load(file_in)
    A = 'A'
    R = 'R'
    outcome = 'phihat'
    theta = testdict['theta']

    def test_risk_coefs(self):
        coefs_phihat = risk_coefs(self.data, self.A, self.R, self.outcome)
        np.testing.assert_array_equal(coefs_phihat, self.testdict['risk_coefs_phihat'])
        coefs_mu0 = risk_coefs(self.data, self.A, self.R, 'mu0')
        np.testing.assert_array_equal(coefs_mu0, self.testdict['risk_coefs_mu0'])

    def test_fairness_coefs(self):
        pos, neg = fairness_coefs(self.data)
        np.testing.assert_array_equal(pos, self.testdict['coefs_pos_phihat'])
        np.testing.assert_array_equal(neg, self.testdict['coefs_neg_phihat'])

        pos, neg = fairness_coefs(self.data, outcome = 'mu0')
        np.testing.assert_array_equal(pos, self.testdict['coefs_pos_mu0'])
        np.testing.assert_array_equal(neg, self.testdict['coefs_neg_mu0'])

    def test_est_risk(self):
        est = est_risk(self.theta, self.data, outcome = 'phihat', ci = 0.95, ci_scale='expit')
        pd.testing.assert_frame_equal(est, self.testdict['risk_df'])
        est = est_risk(self.theta, self.data, outcome = 'phihat', ci = 0.95, ci_scale='logit')
        pd.testing.assert_frame_equal(est, self.testdict['risk_df_logit'])

    def test_est_cFPR(self):
        est = est_cFPR(self.theta, self.data, outcome = 'phihat', ci = 0.95, ci_scale='expit')
        pd.testing.assert_frame_equal(est, self.testdict['cFPR_df'])
        est = est_cFPR(self.theta, self.data, outcome = 'phihat', ci = 0.95, ci_scale='logit')
        pd.testing.assert_frame_equal(est, self.testdict['cFPR_df_logit'])

    def test_est_cFNR(self):
        est = est_cFNR(self.theta, self.data, outcome = 'phihat', ci = 0.95, ci_scale='expit')
        pd.testing.assert_frame_equal(est, self.testdict['cFNR_df'])
        est = est_cFNR(self.theta, self.data, outcome = 'phihat', ci = 0.95, ci_scale='logit')
        pd.testing.assert_frame_equal(est, self.testdict['cFNR_df_logit'])


if __name__ == '__main__':
    unittest.main()
