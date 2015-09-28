#!/usr/bin/env/python
# LICENSE
#
# Copyright (c) 2010-2014, GEM Foundation, G. Weatherill, M. Pagani,
# D. Monelli.
#
# The Hazard Modeller's Toolkit is free software: you can redistribute
# it and/or modify it under the terms of the GNU Affero General Public
# License as published by the Free Software Foundation, either version
# 3 of the License, or (at your option) any later version.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>

'''
Sets up a simple rupture-site configuration to allow for physical comparison
of GMPEs 
'''
import numpy as np
import pdb
from collections import Iterable, OrderedDict

from openquake.hazardlib import gsim, imt
from openquake.hazardlib.scalerel.wc1994 import WC1994


AVAILABLE_GSIMS = gsim.get_available_gsims()

PARAM_DICT = {'magnitudes': [],
              'distances': [],
              'distance_type': 'rjb',
              'vs30': [],
              'strike': None,
              'dip': None,
              'rake': None,
              'ztor': None,
              'hypocentre_location': (0.5, 0.5),
              'hypo_loc': (0.5, 0.5),
              'msr': WC1994()}

PLOT_UNITS = {'PGA': 'g',
              'PGV': 'cm/s',
              'SA': 'g',
              'IA': 'm/s',
              'CSV': 'g-sec',
              'RSD': 's',
              'MMI': ''}

DISTANCE_LABEL_MAP = {'repi': 'Epicentral Dist.',
                      'rhypo': 'Hypocentral Dist.',
                      'rjb': 'Joyner-Boore Dist.',
                      'rrup': 'Rupture Dist.',
                      'rx': 'Rx Dist.'}

def _check_gsim_list(gsim_list):
    """
    Checks the list of GSIM models and returns an instance of the 
    openquake.hazardlib.gsim class. Raises error if GSIM is not supported in
    OpenQuake
    :param list gsim_list:
        List of GSIM names (str)
    """
    output_gsims = []
    for gsim in gsim_list:
        if not gsim in AVAILABLE_GSIMS.keys():
            raise ValueError('%s Not supported by OpenQuake' % gsim)
        else:
            output_gsims.append(AVAILABLE_GSIMS[gsim]())
    return output_gsims

def _get_imts(imts):
    """
    Reads a list of IMT strings and returns the corresponding 
    openquake.hazardlib.imt class
    :param list imts:
        List of IMTs(str)
    """
    out_imts = []
    for imtl in imts:
        out_imts.append(imt.from_string(imtl))
    return out_imts


class BaseTrellis(object):
    """
    Base class for holding functions related to the trellis plotting
    :param list or np.ndarray magnitudes:
        List of rupture magnitudes
    :param dict distances:
        Dictionary of distance measures as a set of np.ndarrays - 
        {'repi', np.ndarray,
         'rjb': np.ndarray,
         'rrup': np.ndarray,
         'rhypo': np.ndarray}
        The number of elements in all arrays must be equal
    :param list gsims:
        List of instance of the openquake.hazardlib.gsim classes to represent
        GMPEs
    :param list imts:
        List of intensity measures
    :param dctx:
        Distance context as instance of :class:
            openquake.hazardlib.gsim.base.DistancesContext
    :param rctx:
        Rupture context as instance of :class:
            openquake.hazardlib.gsim.base.RuptureContext
    :param sctx:
        Rupture context as instance of :class:
            openquake.hazardlib.gsim.base.SitesContext
    :param int nsites:
        Number of sites
    :param str stddevs:
        Standard deviation types
    :param str filename:
        Name of output file for exporting the figure
    :param str filetype:
        String to indicate file type for exporting the figure
    :param int dpi:
        Dots per inch for export figure
    :param str plot_type:
        Type of plot (only used in distance Trellis)
    :param str distance_type:
        Type of source-site distance to be used in distances trellis
    """

    def __init__(self, magnitudes, distances, gsims, imts, params,
            stddevs="Total", **kwargs):
        """
        """
        # Set default keyword arguments
        
        self.magnitudes = magnitudes
        self.distances = distances
        self.gsims = _check_gsim_list(gsims)
        self.params = params
        self.imts = imts
        self.dctx = None
        self.rctx = None
        self.sctx = None
        self.nsites = 0
        self._preprocess_distances()
        self._preprocess_ruptures()
        self._preprocess_sites()
        self.stddevs = stddevs
        self.get_ground_motion_values()


    def _preprocess_distances(self):
        """
        Preprocesses the input distances to check that all the necessary
        distance types required by the GSIMS are found in the
        DistancesContext()
        """
        self.dctx = gsim.base.DistancesContext()
        required_dists = []
        for gmpe in self.gsims:
            gsim_distances = [dist for dist in gmpe.REQUIRES_DISTANCES]
            for dist in gsim_distances:
                if not dist in self.distances.keys():
                    raise ValueError('GMPE %s requires distance type %s'
                                     % (gmpe.__class.__.__name__, dist))
                if not dist in required_dists:
                    required_dists.append(dist)
        dist_check = False
        for dist in required_dists:
            if dist_check and not (len(self.distances[dist]) == self.nsites):
                raise ValueError("Distances arrays not equal length!")
            else:
                self.nsites = len(self.distances[dist])
                dist_check = True
            setattr(self.dctx, dist, self.distances[dist])
            
        
    def _preprocess_ruptures(self):
        """
        Preprocesses rupture parameters to ensure all the necessary rupture
        information for the GSIMS is found in the input parameters
        """
        self.rctx = []
        if not isinstance(self.magnitudes, list) and not\
            isinstance(self.magnitudes, np.ndarray):
            self.magnitudes = np.array(self.magnitudes)
        # Get all required rupture attributes
        required_attributes = []
        for gmpe in self.gsims:
            rup_params = [param for param in gmpe.REQUIRES_RUPTURE_PARAMETERS]
            for param in rup_params:
                if param == 'mag':
                    continue
                elif not param in self.params.keys():
                    raise ValueError("GMPE %s requires rupture parameter %s"
                                     % (gmpe.__class__.__name__, param))
                elif not param in required_attributes:
                    required_attributes.append(param)
                else:
                    pass
        for mag in self.magnitudes:
            rup = gsim.base.RuptureContext()
            setattr(rup, 'mag', mag)
            for attr in required_attributes:
                setattr(rup, attr, self.params[attr])
            self.rctx.append(rup)

    def _preprocess_sites(self):
        """
        Preprocesses site parameters to ensure all the necessary rupture
        information for the GSIMS is found in the input parameters
        """
        self.sctx = gsim.base.SitesContext()
        required_attributes = []
        for gmpe in self.gsims:
            site_params = [param for param in gmpe.REQUIRES_SITES_PARAMETERS]
            for param in site_params:
                obs = np.log(context["Observations"][imtx])
                mean = context["Expected"][gmpe][imtx]["Mean"]
                total_stddev = context["Expected"][gmpe][imtx]["Total"]
                if not param in self.params.keys():
                    raise ValueError("GMPE %s requires site parameter %s"
                                     % (gmpe.__class__.__name__, param))
                elif not param in required_attributes:
                    required_attributes.append(param)
                else:
                    pass
        for param in required_attributes:
            if isinstance(self.params[param], float):
                setattr(self.sctx, param, 
                        self.params[param] * np.ones(self.nsites, dtype=float))
            
            if isinstance(self.params[param], bool):
                if self.params[param]:
                    setattr(self.sctx, param, self.params[param] * 
                               np.ones(self.nsites, dtype=bool))
                else:
                    setattr(self.sctx, param, self.params[param] * 
                               np.zeros(self.nsites, dtype=bool))
            elif isinstance(self.params[param], Iterable):
                if not len(self.params[param]) == self.nsites:
                    raise ValueError("Length of sites value %s not equal to"
                                     " number of sites %" % (param, 
                                     self.nsites))
                setattr(self.sctx, param, self.params[param])
            else:
                pass

class getgmpe(BaseTrellis):
    """
    Class to generate a plots showing the scaling of a set of IMTs with
    magnitude
    """
    def __init__(self, magnitudes, distances, gsims, imts, params,
            stddevs="Total", **kwargs):
        """ 
        """
        for key in distances.keys():
            if isinstance(distances[key], float):
                distances[key] = np.array([distances[key]])
        super(getgmpe, self).__init__(magnitudes, distances, gsims,
            imts, params, stddevs, **kwargs)

    def get_ground_motion_values(self):
        """
        Runs the GMPE calculations to retreive ground motion values
        :returns:
            Nested dictionary of valuesobs = np.log(context["Observations"][imtx])
                mean = context["Expected"][gmpe][imtx]["Mean"]
                total_stddev = context["Expected"][gmpe][imtx]["Total"]
            {'GMPE1': {'IM1': , 'IM2': },
             'GMPE2': {'IM1': , 'IM2': }}
        """
        gmvs = OrderedDict()
        for gmpe in self.gsims:
            gmvs.update([(gmpe.__class__.__name__, {})])
            for i_m in self.imts:
                gmvs[gmpe.__class__.__name__][i_m] = np.zeros(
                    [len(self.rctx), self.nsites], dtype=float)
                for iloc, rct in enumerate(self.rctx):
                    try:
                        means, _ = gmpe.get_mean_and_stddevs(
                            self.sctx,
                            rct,
                            self.dctx,
                            imt.from_string(i_m),
                            [self.stddevs])

                        gmvs[gmpe.__class__.__name__][i_m][iloc, :] = np.exp(means)
                    except KeyError:
                        gmvs[gmpe.__class__.__name__][i_m] = []
                        break
        self.gmvs = gmvs
