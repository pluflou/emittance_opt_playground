import os

from .injector_surrogate_quads import Surrogate_NN
from .sampling_functions import get_ground_truth, get_beamsize
from ..configs.ref_config import ref_point

#from emittance_calc import *
#sys.path.append('../configs')
#Sim reference point to optimize around
#from ref_config import ref_point

dirname = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(dirname, 'shift_one_phase.npy')

Model = Surrogate_NN(
    model_info_file = os.path.join(dirname, '../configs/model_info.json'),
    pv_info_file = os.path.join(dirname, '../configs/pvinfo.json'),
)
Model.load_saved_model(
    model_path = os.path.join(dirname, '../models/'),
    model_name = 'model_OTR2_NA_rms_emit_elu_2021-07-27T19_54_57-07_00',
)
Model.load_scaling(
    scalerfilex = os.path.join(dirname, '../data/transformer_x.sav'),
    scalerfiley = os.path.join(dirname, '../data/transformer_y.sav'),
)
Model.take_log_out = False

energy = 0.135

def get_beamsizes(quad, p1=0.5657, p2=-0.01063, p3=-0.01):
    return get_beamsize(Model, ref_point, p1, p2, p3, quad)
