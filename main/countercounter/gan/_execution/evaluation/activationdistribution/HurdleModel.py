import numpy as np
import scipy.stats


class HurdleModel():

    def __init__(self, data, dist=None, p_value=None):

        if np.array(data).sum() < 0:
            print("*** ERROR: Must be Positive Data ***")

        self.data = data
        self.filtered_data = self.data[self.data != 0]

        self.distributions_to_try = [
            ('norm', scipy.stats.norm),
            ('gamma', scipy.stats.gamma),
            ('expon', scipy.stats.expon),
            ('alpha', scipy.stats.alpha),
            ('beta', scipy.stats.beta),
            ('cauchy', scipy.stats.cauchy),
            ('chi', scipy.stats.chi),
            ('chi2', scipy.stats.chi2),
            ('anglit', scipy.stats.anglit),
            ('arcsine', scipy.stats.arcsine),
            ('argus', scipy.stats.argus),
            ('betaprime', scipy.stats.betaprime),
            ('bradford', scipy.stats.bradford),
            ('burr', scipy.stats.burr),
            ('crystalball', scipy.stats.crystalball),
            ('dgamma', scipy.stats.dgamma),
            ('dweibull', scipy.stats.dweibull),
            ('erlang', scipy.stats.erlang),
            ('f', scipy.stats.f),
            ('fisk', scipy.stats.fisk),
            ('foldcauchy', scipy.stats.foldcauchy),
            ('gilbrat', scipy.stats.gilbrat),
            ('gompertz', scipy.stats.gompertz),
            ('gumbel_r', scipy.stats.gumbel_r),
            ('gumbel_l', scipy.stats.gumbel_l),
            ('hypsecant', scipy.stats.hypsecant),
            ('invgauss', scipy.stats.invgauss),
            ('johnsonsb', scipy.stats.johnsonsb),
            ('johnsonsu', scipy.stats.johnsonsu),
            ('kappa4', scipy.stats.kappa4),
            ('laplace', scipy.stats.laplace),
            ('levy', scipy.stats.levy),
            ('levy_l', scipy.stats.levy),
            ('maxwell', scipy.stats.maxwell),
            ('mielke', scipy.stats.mielke),
            ('moyal', scipy.stats.moyal),
            ('nakagami', scipy.stats.nakagami),
            ('ncx2', scipy.stats.ncx2),
            ('ncf', scipy.stats.ncf),
            ('rice', scipy.stats.rice),
            ('skewnorm', scipy.stats.skewnorm),
        ]

        if dist is None:
            self.rv, fixed_location = self.__get_dist_type()
        else:
            self.rv, fixed_location = self._from_stats(dist, p_value)

        # Try all PDF options
        if fixed_location:
            self.params = self.rv.fit(self.filtered_data, floc=0)
        else:
            self.params = self.rv.fit(self.filtered_data)

        self.fixed_location = fixed_location

    def __get_dist_type(self):
        locations = ['none', 'floc']

        self.p_values = {f'{test_name} {location}': None for test_name, _ in self.distributions_to_try for location in locations}
        # self.p_values = {'norm none': None, 'gamma none': None, 'expon none': None,
        #                  'norm floc': None, 'gamma floc': None, 'expon floc': None,
        #                  }

        for test_name, rv in self.distributions_to_try:
            for location in locations:
                try:
                    if location == 'none':
                        params = rv.fit(self.filtered_data)
                    elif location == 'floc':
                        params = rv.fit(self.filtered_data, floc=0)

                    self.p_values[test_name + " " + location] = (scipy.stats.kstest(self.filtered_data, test_name, args=params)[1])
                except:
                    self.p_values[test_name + " " + location] = (-1)

        p_values_without_nan = {k: v for k, v in self.p_values.items() if not np.isnan(v)}
        self.max_key = max(p_values_without_nan, key=lambda k: self.p_values[k])
        self.max_p_value = self.p_values[self.max_key]
        dist_type, location = self.max_key.split(" ")

        chosen_rv = None
        for test_name, rv in self.distributions_to_try:
            if test_name == dist_type:
                chosen_rv = rv
                break

        assert chosen_rv is not None

        if location == 'none':
            return chosen_rv, False
        elif location == 'floc':
            return chosen_rv, True

    def __pdf(self, x):
        return self.rv.pdf(x, *self.params)

    def __cdf(self, x):
        return self.rv.cdf(x, *self.params)

    def get_cdf(self, x):
        return self.rv.cdf(x, *self.params)

    def get_expected_value(self):
        return self.rv.mean(*self.params)

    def test_fit(self):
        return scipy.stats.kstest(self.filtered_data, self.rv.name, args=self.params)[1]

    def get_stats(self):
        return self.max_key, self.max_p_value, self.p_values

    def _from_stats(self, dist, p_value):
        dist_name, location = dist.split(" ")

        chosen_rv = None
        for name, scipy_dist in self.distributions_to_try:
            if name == dist_name:
                chosen_rv = scipy_dist
                break

        assert chosen_rv is not None

        self.max_key = dist
        self.max_p_value = p_value
        self.p_values = {}

        if location == 'none':
            return chosen_rv, False
        elif location == 'floc':
            return chosen_rv, True
