import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from lcls.injector_surrogate.injector_surrogate_quads import Surrogate_NN
from lcls.injector_surrogate.sampling_functions import get_ground_truth, get_beamsize
from lcls.injector_surrogate.emittance_calc_211216 import get_normemit, adapt_range
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

    def emittance_fn(self, quad_x_list, quad_y_list, beamsize_x_list, beamsize_y_list):
        """
        Return emittance (a tuple of floats), given quad_x_list, quad_y_list (each a
        list of quad measurement floats), and beamsize_x_list, beamsize_y_list (each a
        list of beamsize floats).
        """
        x_arr = np.array(beamsize_x_list)
        y_arr = np.array(beamsize_y_list)

        emittance = get_normemit(
            self.params.energy,
            quad_x_list,
            quad_y_list,
            np.array(beamsize_x_list),
            np.array(beamsize_y_list),
            adapt_ranges = True
        )
        emittance = tuple(float(t) for t in emittance)

        return emittance

    def full_emittance_fn(self, config, quad_x_list, quad_y_list):
        """
        Return emittance (a tuple of floats), given config (a list of three floats) and
        quad_x_list, quad_y_list (each a list of floats).
        """
        # Get beamsize_x_list and beamsize_y_list for the two quad scan lists
        beamsizes_list = self.beamsizes_list_fn(config, quad_x_list)
        beamsize_x_list = [tup[0] for tup in beamsizes_list]

        beamsizes_list = self.beamsizes_list_fn(config, quad_y_list)
        beamsize_y_list = [tup[1] for tup in beamsizes_list]

        # Compute and return emittance
        emittance = self.emittance_fn(
            quad_x_list, quad_y_list, beamsize_x_list, beamsize_y_list
        )
        return emittance

    def emittance_scalar_fn(self, emittance_tuple):
        """Return emittance scalar from tuple via geometric mean."""
        emittance_scalar = np.sqrt(emittance_tuple[0] * emittance_tuple[1])
        return emittance_scalar

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

    def normalize_emit(self, emit):
        """Return emit (a scalar) in normalized space."""
        eb, eb_norm = self.params.emit_bounds, self.params.emit_bounds_norm
        emit = self.rescale_input(emit, eb, eb_norm)
        return emit

    def unnormalize_emit(self, emit):
        """Return emit (a scalar) in original space."""
        eb_norm, eb = self.params.emit_bounds_norm, self.params.emit_bounds
        emit = self.rescale_input(emit, eb_norm, eb)
        return emit

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

    def emittance_fn_norm(
        self, quad_x_list, quad_y_list, beamsize_x_list, beamsize_y_list
    ):
        """
        A wrapper of self.emittance_fn. This returns (unnormalized) emittance (a tuple
        of floats), where quad_x_list, quad_y_list are in the normalized space (each a
        list of quad measurement floats).
        """
        quad_x_list = [self.unnormalize_quad(quad) for quad in quad_x_list]
        quad_y_list = [self.unnormalize_quad(quad) for quad in quad_y_list]
        beamsizes_list = zip(beamsize_x_list, beamsize_y_list)
        beamsizes_list = [self.unnormalize_beamsizes(bs) for bs in beamsizes_list]
        bs_xy_list = [list(bs) for bs in zip(*beamsizes_list)]
        beamsize_x_list, beamsize_y_list = bs_xy_list[0], bs_xy_list[1]
        emittance = self.emittance_fn(
            quad_x_list, quad_y_list, beamsize_x_list, beamsize_y_list
        )
        #### NOTE: returning regular emittance (not normalized emittance)
        return emittance

    def full_emittance_fn_norm(self, config, quad_x_list, quad_y_list):
        """
        A wrapper of self.full_emittance_fn. This returns (unnormalized) emittance (a
        tuple of floats), where config (a list) and quad_x_list, quad_y_list (each a
        list of quad measurement floats) are in the normalized space.
        """
        config = self.unnormalize_config(config)
        quad_x_list = [self.unnormalize_quad(quad) for quad in quad_x_list]
        quad_y_list = [self.unnormalize_quad(quad) for quad in quad_y_list]
        emittance = self.full_emittance_fn(config, quad_x_list, quad_y_list)
        #### NOTE: returning regular emittance (not normalized emittance)
        return emittance

    def plot_line_of_config_scans_norm(
        self,
        config_start,
        config_end,
        num_config,
        quad_x_list,
        quad_y_list,
        fig=None,
        axarr=None,
    ):
        """
        A wrapper of self.plot_line_of_config_scans. This performs plotting when all
        inputs are given in the normalized space.
        """
        config_start = self.unnormalize_config(config_start)
        config_end = self.unnormalize_config(config_end)
        quad_x_list = [self.unnormalize_quad(quad) for quad in quad_x_list]
        quad_y_list = [self.unnormalize_quad(quad) for quad in quad_y_list]
        fig, axarr = self.plot_line_of_config_scans(
            config_start, config_end, num_config, quad_x_list, quad_y_list, fig, axarr
        )
        return fig, axarr

    def plot_list_of_config_scans_norm(
        self, config_list, quad_x_list, quad_y_list, fig=None, axarr=None
    ):
        """
        A wrapper of self.plot_list_of_config_scans. This performs plotting when all
        inputs are given in the normalized space.
        """
        config_list = [self.unnormalize_config(config) for config in config_list]
        quad_x_list = [self.unnormalize_quad(quad) for quad in quad_x_list]
        quad_y_list = [self.unnormalize_quad(quad) for quad in quad_y_list]
        return self.plot_list_of_config_scans(
            config_list, quad_x_list, quad_y_list, fig, axarr
        )

    def plot_line_of_config_scans(
        self,
        config_start,
        config_end,
        num_config,
        quad_x_list,
        quad_y_list,
        fig=None,
        axarr=None,
    ):
        """
        For a line of configs (a set of num_configs configs between config_start and
        config_end), plot beamsizes vs scan (quad_list) for both x and y.
        """
        configs = [list(j) for j in np.linspace(config_start, config_end, num_config)]
        fig, axarr = self.plot_list_of_config_scans(
            configs, quad_x_list, quad_y_list, fig, axarr
        )
        return fig, axarr

    def plot_list_of_config_scans(
        self, config_list, quad_x_list, quad_y_list, fig=None, axarr=None
    ):
        """
        For a list of configs (each a list containing three floats), plot beamsizes vs
        scan (quad_list) for both x and y.
        """
        if fig is None or axarr is None:
            fig, axarr = plt.subplots(1, 2, figsize=(12, 5))
        parameters, s_m = self.get_colorbar_vars(len(config_list))
        emit_list = []
        for config, parameter in zip(config_list, parameters):
            # Get x_sizes from quad_x_list scan
            beamsizes = self.beamsizes_list_fn(config, quad_x_list)
            x_sizes = np.array([tup[0] for tup in beamsizes])

            # Get y_sizes from quad_y_list scan
            beamsizes = self.beamsizes_list_fn(config, quad_y_list)
            y_sizes = np.array([tup[1] for tup in beamsizes])

            # Plot quad_list vs sizes squared
            axarr[0].plot(quad_x_list, x_sizes**2, 'o-', color=s_m.to_rgba(parameter))
            axarr[1].plot(quad_y_list, y_sizes**2, 'o-', color=s_m.to_rgba(parameter))

            # Compute and store emittance of scan
            emit_tup = self.emittance_fn(quad_x_list, quad_y_list, x_sizes, y_sizes)
            emit = np.sqrt(emit_tup[0] * emit_tup[1])
            emit_list.append(f'{emit / 1e-6:.2f}')

            print(f'Param = {parameter}, Emit = {emit_tup}, {emit}')

        # Plot settings
        axarr[0].set(xlabel='quad scan', ylabel='$x$ beamsizes$^2$', title='$x$ beamsizes')
        axarr[1].set(xlabel='quad scan', ylabel='$y$ beamsizes$^2$', title='$y$ beamsizes')
        fig = self.add_colorbar_to_plot(fig, parameters, s_m, emit_list)

        return fig, axarr

    def get_colorbar_vars(self, num_config):
        """Return variables needed for making colorbar in plots."""
        parameters = np.linspace(0, 1, num_config)
        norm = mpl.colors.Normalize(vmin=np.min(parameters), vmax=np.max(parameters))
        c_m = mpl.cm.cool
        s_m = mpl.cm.ScalarMappable(cmap=c_m, norm=norm)
        s_m.set_array([])
        return parameters, s_m

    def add_colorbar_to_plot(self, fig, parameters, s_m, emit_list):
        """Return fig with colorbar added in its own axes."""
        fig.subplots_adjust(right=0.82)
        ax_cbar = fig.add_axes([0.89, 0.15, 0.02, 0.72])
        cbar = fig.colorbar(s_m, cax=ax_cbar, ticks=list(parameters))
        ax_cbar.set_yticklabels(emit_list)
        ax_cbar.set(title='Emittance')
        return fig

    def get_adapted_quad_list(self, quad_list, bs_list, axis, num_points):
        """
        Return adapted list of quad values (scalars, unnormalized), given a list of
        initial quad values (unnormalized), a list of beamsize scalars (unnormalized),
        an axis label (either 'x' or 'y'), and the number of points to use in the
        adapted scan.
        """
        try:
            new_quad_arr = adapt_range(quad_list, bs_list, axis=axis, num_points=num_points)
            new_quad_list = new_quad_arr.tolist()
        except:
            #print("lcls_functions: adapt_range failed. Returning original quad_list.")
            print(".", end="")
            new_quad_list = quad_list
        return new_quad_list

    def set_print_params(self):
        """Set self.print_params."""
        super().set_print_params()
        self.print_params.reference_point = '<reference_point>'
