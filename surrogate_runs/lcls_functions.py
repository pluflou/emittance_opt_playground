import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl

from lcls.injector_surrogate.injector_surrogate_quads import Surrogate_NN
from lcls.injector_surrogate.sampling_functions import get_ground_truth, get_beamsize

from lcls.configs.ref_config import ref_point
from base import Base
from misc_util import dict_to_namespace


class Lcls(Base):
    """Class for LCLS functions."""

    def set_params(self, params):
        """Set self.params and other attributes."""
        super().set_params(params)
        params = dict_to_namespace(params)

        self.params.name = getattr(params, 'name', 'Lcls')
        self.set_bounds(params)
        self.set_bounds_norm(params)
        self.set_surrogate_model(params)
        self.params.energy = getattr(params, 'energy', 0.135)
        self.params.reference_point = getattr(params, 'reference_point', ref_point)
        self.params.verbose = False

    def set_bounds(self, params):
        """Set bounds for config, quad, and emit."""
        self.params.config_bounds = getattr(params, 'config_bounds',
            [(0.46, 0.485), (-0.01, 0.01), (-0.01, 0.01)]
        )
        self.params.quad_bounds = getattr(params, 'quad_bounds',
            (-6.0, 0.0)
        )
        self.params.emit_bounds = getattr(params, 'emit_bounds',
            (0.5e-6, 5e-6)
        )
        self.params.beamsizes_bounds = getattr(params, 'beamsizes_bounds',
            [(0.0, 5e-4), (0.0, 1e-3)]
        )

    def set_bounds_norm(self, params):
        """Set normalized bounds for config, quad, and emit."""
        self.params.normalize = getattr(params, 'normalize', False)
        self.params.config_bounds_norm = getattr(params, 'config_bounds_norm',
            [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
        )
        self.params.quad_bounds_norm = getattr(params, 'quad_bounds_norm',
            (0.0, 1.0)
        )
        self.params.emit_bounds_norm = getattr(params, 'emit_bounds_norm',
            (-1.0, 1.0)
        )
        self.params.beamsizes_bounds_norm = getattr(params, 'beamsizes_bounds_norm',
            [(-1.0, 1.0), (-1.0, 1.0)]
        )

    def set_surrogate_model(self, params):
        """Set self.surrogate_model."""
        self.surrogate_model = Surrogate_NN(
            model_info_file = 'lcls/configs/model_info.json',
            pv_info_file = 'lcls/configs/pvinfo.json',
        )
        self.surrogate_model.load_saved_model(
            model_path = 'lcls/models/',
            model_name = 'model_OTR2_NA_rms_emit_elu_2021-07-27T19_54_57-07_00',
        )
        self.surrogate_model.load_scaling(
            scalerfilex = 'lcls/data/transformer_x.sav',
            scalerfiley = 'lcls/data/transformer_y.sav',
        )
        self.surrogate_model.take_log_out = False

    def beamsizes_fn(self, config_quad):
        """
        Given config_quad (a list containing three config variables and one quad
        measurement, all floats), return beamsizes (a tuple of two floats).
        """
        #bar_str = '=====' * 20 + '\n'
        #print(bar_str + f'* Calling get_beamsize function on {config_quad}\n')

        beamsizes = get_beamsize(
            self.surrogate_model,
            self.params.reference_point,
            config_quad[0],
            config_quad[1],
            config_quad[2],
            config_quad[3],
        )
        beamsizes = (float(beamsizes[0]), float(beamsizes[1]))

        #print(f'* Beamsizes = {beamsizes}\n' + bar_str)

        return beamsizes

    def beamsizes_list_fn(self, config, quad_list):
        """Return list of beamsizes tuples (one per quad measurement in quad_list)."""
        beamsizes_list = []
        for quad in quad_list:
            config_quad = config + [quad]
            beamsizes = self.beamsizes_fn(config_quad)
            beamsizes_list.append(beamsizes)

        return beamsizes_list

    def rescale_input(self, x, tup1, tup2):
        """
        Transform input from one scale (given by tup1 bounds), to another (given by tup2
        bounds).
        """
        x = (x - tup1[0]) / (tup1[1] - tup1[0])
        x = x * (tup2[1] - tup2[0]) + tup2[0]
        return x

    def normalize_config(self, config):
        """Return config (a list) in normalized space."""
        zipped = zip(config, self.params.config_bounds, self.params.config_bounds_norm)
        config = [self.rescale_input(x, tup, tup_norm) for x, tup, tup_norm in zipped]
        return config

    def unnormalize_config(self, config):
        """Return config (a list) in original space."""
        zipped = zip(config, self.params.config_bounds_norm, self.params.config_bounds)
        config = [self.rescale_input(x, tup_norm, tup) for x, tup_norm, tup in zipped]
        return config

    def normalize_quad(self, quad):
        """Return quad (a scalar) in normalized space."""
        qb, qb_norm = self.params.quad_bounds, self.params.quad_bounds_norm
        quad = self.rescale_input(quad, qb, qb_norm)
        return quad

    def unnormalize_quad(self, quad):
        """Return quad (a scalar) in original space."""
        qb_norm, qb = self.params.quad_bounds_norm, self.params.quad_bounds
        quad = self.rescale_input(quad, qb_norm, qb)
        return quad

    def normalize_config_quad(self, config_quad):
        """Return config_quad (a list) in normalized space."""
        config = self.normalize_config(config_quad[:-1])
        quad = self.normalize_quad(config_quad[-1])
        config_quad = config + [quad]
        return config_quad

    def unnormalize_config_quad(self, config_quad):
        """Return config_quad (a scalar) in original space."""
        config = self.unnormalize_config(config_quad[:-1])
        quad = self.unnormalize_quad(config_quad[-1])
        config_quad = config + [quad]
        return config_quad

    def normalize_beamsizes(self, beamsizes):
        """Return beamsizes (a tuple) in normalized space."""
        lb = list(beamsizes)
        zipped = zip(
            lb, self.params.beamsizes_bounds, self.params.beamsizes_bounds_norm
        )
        beamsizes = tuple(
            self.rescale_input(x, tup, tup_norm) for x, tup, tup_norm in zipped
        )
        return beamsizes

    def unnormalize_beamsizes(self, beamsizes):
        """Return beamsizes (a tuple) in original space."""
        lb = list(beamsizes)
        zipped = zip(
            lb, self.params.beamsizes_bounds_norm, self.params.beamsizes_bounds
        )
        beamsizes = tuple(
            self.rescale_input(x, tup_norm, tup) for x, tup_norm, tup in zipped
        )
        return beamsizes

    def beamsizes_fn_norm(self, config_quad):
        """
        A wrapper of self.beamsizes_fn, where config_quad is a list in the normalized
        space (containing three config variables and one quad measurement, all floats).
        """
        config_quad = self.unnormalize_config_quad(config_quad)
        beamsizes = self.beamsizes_fn(config_quad)
        beamsizes_norm = self.normalize_beamsizes(beamsizes)
        return beamsizes_norm

    def beamsizes_list_fn_norm(self, config, quad_list):
        """
        A wrapper of self.beamsizes_list_fn, which returns list of beamsizes tuples,
        where config and quad_list inputs are in the normalized space.
        """
        config = self.unnormalize_config(config)
        quad_list = [self.unnormalize_quad(quad) for quad in quad_list]
        beamsizes_list = self.beamsizes_list_fn(config, quad_list)
        beamsizes_list_norm = [self.normalize_beamsizes(bs) for bs in beamsizes_list]
        return beamsizes_list_norm


    def set_print_params(self):
        """Set self.print_params."""
        super().set_print_params()
        self.print_params.reference_point = '<reference_point>'
