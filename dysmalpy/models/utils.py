# coding=utf8
# Copyright (c) MPE/IR-Submm Group. See LICENSE.rst for license information. 
#
# Utility calculations for models, shared between different model types

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Standard library
import logging

from collections import OrderedDict

# Third party imports
import numpy as np
import jax.numpy as jnp

# Local imports
from dysmalpy.parameters import DysmalParameter


# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DysmalPy')
logger.setLevel(logging.INFO)

import warnings
warnings.filterwarnings("ignore")


def get_light_components_by_tracer(model_set, tracer):

    ncmp_tracer = 0
    lcomps_tracer = OrderedDict()
    # Add all components that match the specific tracer:
    for cmp in model_set.light_components:
        if (model_set.light_components[cmp]):
            if (model_set.components[cmp].tracer == tracer):
                ncmp_tracer += 1
                lcomps_tracer[cmp] = model_set.light_components[cmp]

    # Fallback: no tracer-specifics? Use the mass components:
    if ncmp_tracer == 0:
        for cmp in model_set.light_components:
            if (model_set.light_components[cmp]):
                if (model_set.components[cmp].tracer == 'mass'):
                    lcomps_tracer[cmp] = model_set.light_components[cmp]

    return lcomps_tracer



def get_geom_phi_rad_polar(x, y):
    """
    Calculate polar angle phi from x, y.
    """
    R = jnp.sqrt(x ** 2 + y ** 2)
    # Assume ring is in midplane
    phi_geom_rad = jnp.arcsin(y / R)

    # For x < 0, use pi - arcsin(y/R) instead
    phi_neg_x = jnp.pi - jnp.arcsin(y / R)
    phi_geom_rad = jnp.where(x < 0, phi_neg_x, phi_geom_rad)

    # Handle R == 0 case: set phi to 0
    phi_geom_rad = jnp.where(R == 0, 0., phi_geom_rad)

    return phi_geom_rad

def replace_values_by_refarr(arr, ref_arr, excise_val, subs_val):
    """
    Excise (presumably NaN) values from an array,
    by replacing all entries where ref_arr == excise_value with subs_val, or
       arr[ref_arr==excise_val] = subs_val
    """
    if len(jnp.shape(ref_arr)) == 0:
        if ref_arr == excise_val:
            arr = subs_val
    else:
        arr = jnp.where(ref_arr == excise_val, subs_val, arr)

    return arr


def insert_param_state(state, pn, value=None, fixed=True, tied=False, bounds=None, prior=None):
    """
    Function to insert a DysmalParameter into state for backwards compatibility
    when loading pickles.
    """
    state[pn] = DysmalParameter(default=value, fixed=fixed, tied=tied, bounds=bounds,prior=prior)
    if '_constraints' in state.keys():
        for cnst in ['fixed', 'tied', 'bounds', 'prior']:
            state['_constraints'][cnst][pn] = state[pn].__dict__["_{}".format(cnst)]

    pmdict = {'shape': (), 'orig_unit': None, 'raw_unit': None, 'size': 1}
    ind_pn = len(state['_parameters'])
    pmdict['slice'] = slice(ind_pn, ind_pn+1, None)
    state['_param_metrics'][pn] = pmdict

    state['_parameters'] = np.append(state['_parameters'], value)

    return state


def remove_param_state(state, pn):
    """
    Function to remove a DysmalParameter from state for backwards compatibility
    when loading pickles (if, eg the parameter has been shifted to a plain attribute).
    """
    del state[pn]
    if '_constraints' in state.keys():
        for cnst in ['fixed', 'tied', 'bounds', 'prior']:
            del state['_constraints'][cnst][pn]

    pmdict = {'shape': (), 'orig_unit': None, 'raw_unit': None, 'size': 1}
    ind_pn = len(state['_parameters'])
    pmdict['slice'] = slice(ind_pn, ind_pn+1, None)

    ind_pn = state['_param_metrics'][pn].start
    del state['_param_metrics'][pn]

    orig_params = state['_parameters'][:]
    state['_parameters'] = np.array([])
    for ind, value in enumerate(orig_params):
        if (ind != ind_pn):
            np.append(state['_parameters'], value)

    return state
