import sys
import h5py
import numpy as np
from math import sqrt, ceil
from scipy.special import erf
from scipy.stats import scoreatpercentile, norm
from copy import deepcopy
from collections import OrderedDict
from openquake.hazardlib.gsim import get_available_gsims
import smtk.intensity_measures as ims
from openquake.hazardlib import imt
from smtk.strong_motion_selector import SMRecordSelector
import pdb

GSIM_LIST = get_available_gsims()
GSIM_KEYS = set(GSIM_LIST.keys())

#SCALAR_IMTS = ["PGA", "PGV", "PGD", "CAV", "Ia"]
SCALAR_IMTS = ["PGA", "PGV"]

class LLH(Residuals):
    """
    Implements of average sample log-likelihood estimator from
    Scherbaum et al (2009)
    """
    def get_loglikelihood_values(self):
        log_residuals = OrderedDict([(gmpe, np.array([]))
                                      for gmpe in self.gmpe_list])
        llh = OrderedDict([(gmpe, None) for gmpe in self.gmpe_list])
        for gmpe in self.gmpe_list:
            for imtx in self.imts:
                asll = np.log2(norm.pdf(self.residuals[gmpe][imtx]["Total"], 
                               0., 
                               1.0))
                log_residuals[gmpe] = np.hstack([log_residuals[gmpe], asll])
            llh[gmpe] = -(1. / float(len(log_residuals[gmpe]))) * np.sum(log_residuals[gmpe])
        # Get weights
        weights = np.array([2.0 ** -llh[gmpe] for gmpe in self.gmpe_list])
        weights = weights / np.sum(weights)
        model_weights = OrderedDict([
            (gmpe, weights[iloc]) for iloc, gmpe in enumerate(self.gmpe_list)]
            )
        return llh, model_weights

class EDR(Residuals):
    """
    Implements the Euclidean Distance-Based Ranking Method for GMPE selection
    by Kale & Akkar (2013)
    Kale, O., and Akkar, S. (2013) A New Procedure for Selecting and Ranking
    Ground Motion Predicion Equations (GMPEs): The Euclidean Distance-Based
    Ranking Method
    """
    def get_edr_values(self, bandwidth=0.01, multiplier=3.0):
        """
        Calculates the EDR values for each GMPE
        :param float bandwidth:
            Discretisation width
        :param float multiplier:
            "Multiplier of standard deviation (equation 8 of Kale and Akkar)
        """
        edr_values = OrderedDict([(gmpe, {}) for gmpe in self.gmpe_list])
        for gmpe in self.gmpe_list:
            obs, expected, stddev = self._get_gmpe_information(gmpe)
            results = self._get_edr(obs,
                                    expected,
                                    stddev,
                                    bandwidth,
                                    multiplier)
            edr_values[gmpe]["MDE Norm"] = results[0]
            edr_values[gmpe]["sqrt Kappa"] = results[1]
            edr_values[gmpe]["EDR"] = results[2]
        return edr_values

    def _get_gmpe_information(self, gmpe):
        """
        Extract the observed ground motions, expected and total standard
        deviation for the GMPE (aggregating over all IMS)
        """
        obs = np.array([], dtype=float)
        expected = np.array([], dtype=float)
        stddev = np.array([], dtype=float)
        for imtx in self.imts:
            for context in self.contexts:
                obs = np.hstack([obs, np.log(context["Observations"][imtx])])
                expected = np.hstack([expected,
                                      context["Expected"][gmpe][imtx]["Mean"]])
                stddev = np.hstack([stddev,
                                    context["Expected"][gmpe][imtx]["Total"]])
        return obs, expected, stddev

    def _get_edr(self, obs, expected, stddev, bandwidth=0.01, multiplier=3.0):
        """
        Calculated the Euclidean Distanced-Based Rank for a set of
        observed and expected values from a particular GMPE
        """
        nvals = len(obs)
        min_d = bandwidth / 2.
        kappa = self._get_kappa(obs, expected)
        mu_d = obs - expected
        d1c = np.fabs(obs - (expected - (multiplier * stddev)))
        d2c = np.fabs(obs - (expected + (multiplier * stddev)))
        dc_max = ceil(np.max(np.array([np.max(d1c), np.max(d2c)])))
        num_d = len(np.arange(min_d, dc_max, bandwidth))
        mde = np.zeros(nvals)
        for iloc in range(0, num_d):
            d_val = (min_d + (float(iloc) * bandwidth)) * np.ones(nvals)
            d_1 = d_val - min_d
            d_2 = d_val + min_d
            p_1 = norm.cdf((d_1 - mu_d) / stddev) -\
                norm.cdf((-d_1 - mu_d) / stddev)
            p_2 = norm.cdf((d_2 - mu_d) / stddev) -\
                norm.cdf((-d_2 - mu_d) / stddev)
            mde += (p_2 - p_1) * d_val
        inv_n = 1.0 / float(nvals)
        mde_norm = np.sqrt(inv_n * np.sum(mde ** 2.))
        edr = np.sqrt(kappa * inv_n * np.sum(mde ** 2.))
        return mde_norm, np.sqrt(kappa), edr


    def _get_kappa(self, obs, expected):
        """
        Returns the correction factor kappa
        """
        mu_a = np.mean(obs)
        mu_y = np.mean(expected)
        b_1 = np.sum((obs - mu_a) * (expected - mu_y)) / np.sum((obs - mu_a) ** 2.)
        b_0 = mu_y - b_1 * mu_a
        y_c =  expected - ((b_0 + b_1 * obs) - obs)
        de_orig = np.sum((obs - expected) ** 2.)
        de_corr = np.sum((obs - y_c) ** 2.)
        return de_orig / de_corr
 
