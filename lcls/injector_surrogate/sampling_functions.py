import sys
import numpy as np

#NN Surrogate model
from .injector_surrogate_quads import Model
from lcls.configs.ref_config import ref_point
from lcls.injector_surrogate.injector_surrogate_quads import Surrogate_NN

#input params: solenoid and quads to vary
opt_var_names = ["SOL1:solenoid_field_scale","CQ01:b1_gradient","SQ01:b1_gradient",
                 "QA01:b1_gradient","QA02:b1_gradient","QE01:b1_gradient","QE02:b1_gradient","QE03:b1_gradient",
                 "QE04:b1_gradient"]

def set_surrogate_model():
    """Set surrogate_model."""
    surrogate_model = Surrogate_NN(
        model_info_file = "lcls/configs/model_info.json",
        pv_info_file = "lcls/configs/pvinfo.json",
    )
    surrogate_model.load_saved_model(
        model_path = "lcls/models/",
        model_name = "model_OTR2_NA_rms_emit_elu_2021-07-27T19_54_57-07_00",
    )
    surrogate_model.load_scaling(
        scalerfilex = "lcls/data/transformer_x.sav",
        scalerfiley = "lcls/data/transformer_y.sav",
    )
    surrogate_model.take_log_out = False
    
    return surrogate_model
        
def evaluate(config: list, ref_point=ref_point): 
    """
    D is input space dimensionality
    N is number of sample points
    :param config: input values of opt_var_names, torch.tensor, shape (N, D) 
    returns (1, N) 
    """
    config = np.asarray(config)
    if len(config.shape) == 1:
        config = config.reshape(1,-1)
    N = config.shape[0]
    D = config.shape[1]

    Model = set_surrogate_model()
    
    # point in sim around which to optimize
    ref_point = Model.sim_to_machine(np.asarray(ref_point))
    
    # make input array of length model_in_list (inputs model takes)
    x_in = np.empty((N,len(Model.model_in_list)))
    
    # fill in reference point around which to optimize
    x_in[:,:] = np.asarray(ref_point[0])

    #set solenoid, CQ, SQ, matching quads to values from optimization step
    col = []
    for i in range(D):
        col.append(Model.loc_in[opt_var_names[i]]) #should make col a flat list of indices, e.g. [4, 6, 7]
        
    x_in[:, col] = config[:,:] 
    
    #output predictions
    y_out = Model.pred_machine_units(x_in)

    return objective(y_out, Model)


def objective(y_out, Model):
    """
    y_out has a shape of (M, N, num_outputs)
    """
    # output is dict 
    out = {}
    
    # geometric emittance in transverse plane, units of m.rad
    out["emitx"] = y_out[:,Model.loc_out["norm_emit_x"]] #grab norm_emit_x out of the model
    out["emity"]  = y_out[:,Model.loc_out["norm_emit_y"]] #grab norm_emit_y out of the model
    out["emit_geo_mean"]  = np.sqrt(out["emitx"] * out["emity"]) # geometric mean of the emittance
  
    # beam sizes prediction at OTR2 in units of m
    out["sigma_x"] = y_out[:,Model.loc_out["sigma_x"]] #grab sigma_x out of the model 
    out["sigma_y"] = y_out[:,Model.loc_out["sigma_y"]] #grab sigma_y out of the model 
        
    return out

def get_ground_truth(Model,ref_point,varx,vary,varz,var1,var2,var3,var4,var5,var6):
    """Returns normalized emittance prediction from the surrogate model
       for given settings of SOL1, SQ01, and CQ01 """
    
    #convert to machine units
    ref_point = Model.sim_to_machine(np.asarray(ref_point))

    #make input array of length model_in_list (inputs model takes)
    x_in = np.empty((1,len(Model.model_in_list)))

    #fill in reference point around which to optimize
    x_in[:,:] = np.asarray(ref_point[0])

    #set solenoid, SQ, CQ to values from optimization step
    x_in[:, Model.loc_in[opt_var_names[0]]] = varx
    x_in[:, Model.loc_in[opt_var_names[1]]] = vary
    x_in[:, Model.loc_in[opt_var_names[2]]] = varz

    #set all 6 matching quads to values from optimization step
    x_in[:, Model.loc_in[opt_var_names[3]]] = var1
    x_in[:, Model.loc_in[opt_var_names[4]]] = var2
    x_in[:, Model.loc_in[opt_var_names[5]]] = var3
    x_in[:, Model.loc_in[opt_var_names[6]]] = var4
    x_in[:, Model.loc_in[opt_var_names[7]]] = var5
    x_in[:, Model.loc_in[opt_var_names[8]]] = var6

    #output predictions
    y_out = Model.pred_machine_units(x_in) 
    
    #output is geometric emittance in transverse plane
    emitx = y_out[:,Model.loc_out["norm_emit_x"]] #grab norm_emit_x out of the model
    emity = y_out[:,Model.loc_out["norm_emit_y"]] #grab norm_emit_y out of the model

    print("NOTE this only returns the scalar normalized emittance (not bmag) from the surrogate predictions.")
    return np.sqrt(emitx*emity)

def get_beamsize(Model, ref_point, varx, vary, varz, var1, var2, var3, var4, var5, var6, use_lcls=False):
    """Returns the beamsize (xrms, yrms) prediction [m] from the surrogate model
    for given settings of SOL1, SQ01, CQ01 and scanning quad QE04 """

    if use_lcls:
        # Use function from beam_io
        io = MachineIO("LCLS", "WIRE")
        io.online = True
        # config is now the full injector settings with matching quads
        config = [varx, vary, varz, var1, var2, var3, var4, var5, var6]
        beamsizes_arr = io.get_beamsizes_machine(config)
        return beamsizes_arr

    else:
        # Use surrogate model

        #convert to machine units
        ref_point = Model.sim_to_machine(np.asarray(ref_point))

        #make input array of length model_in_list (inputs model takes)
        x_in = np.empty((1,len(Model.model_in_list)))

        #fill in reference point around which to optimize
        x_in[:,:] = np.asarray(ref_point[0])

        #set solenoid, SQ, CQ to values from optimization step
        x_in[:, Model.loc_in[opt_var_names[0]]] = varx
        x_in[:, Model.loc_in[opt_var_names[1]]] = vary
        x_in[:, Model.loc_in[opt_var_names[2]]] = varz

        #set all 6 matching quads to values from optimization step
        x_in[:, Model.loc_in[opt_var_names[3]]] = var1
        x_in[:, Model.loc_in[opt_var_names[4]]] = var2
        x_in[:, Model.loc_in[opt_var_names[5]]] = var3
        x_in[:, Model.loc_in[opt_var_names[6]]] = var4
        x_in[:, Model.loc_in[opt_var_names[7]]] = var5
        x_in[:, Model.loc_in[opt_var_names[8]]] = var6

        #output predictions
        y_out = Model.pred_machine_units(x_in)

        x_rms = y_out[:,0][0]
        y_rms = y_out[:,1][0]

        return np.array([x_rms, y_rms])
    
def get_beamsize_3d(Model, ref_point, varx, vary, varz, varscan, use_lcls=False):
    """Returns the beamsize (xrms, yrms) prediction [m] from the surrogate model
    for given settings of SOL1, SQ01, CQ01 and scanning quad QE04 """
    opt_var_names = ["SOL1:solenoid_field_scale","SQ01:b1_gradient","CQ01:b1_gradient","QE04:b1_gradient"]
    
    if use_lcls:
        # Use function from beam_io
        beamsizes_arr = get_beamsize_inj([varx, vary, varz], varscan)
        return beamsizes_arr
    else:
        # Use surrogate model
        #convert to machine units
        ref_point = Model.sim_to_machine(np.asarray(ref_point))
        #make input array of length model_in_list (inputs model takes)
        x_in = np.empty((1,len(Model.model_in_list)))
        #fill in reference point around which to optimize
        x_in[:,:] = np.asarray(ref_point[0])
        #set solenoid, SQ, CQ to values from optimization step
        x_in[:, Model.loc_in[opt_var_names[0]]] = varx
        x_in[:, Model.loc_in[opt_var_names[1]]] = vary
        x_in[:, Model.loc_in[opt_var_names[2]]] = varz
        #set quad 525 to values for scan
        x_in[:, Model.loc_in[opt_var_names[3]]] = varscan
        #output predictions
        y_out = Model.pred_machine_units(x_in)
        x_rms = y_out[:,0][0]
        y_rms = y_out[:,1][0]
        return np.array([x_rms, y_rms])

def get_match_emittance_from_scan(config: list, eval_fn=evaluate):
    """Config here is 8D, all model inputs except the scanning quad QE04"""
    from pyemittance.emit_eval_example import eval_emit_surrogate

    out_dict, _ = eval_emit_surrogate(get_bs_model=eval_fn,
                                      config=config,
                                      quad_init=[-7,-4,-1,2],
                                      num_points=5,
                                      calc_bmag=True)

    # return match*emittance
    return out_dict["nemit"], out_dict["nemit_err"], out_dict["bmag_emit"], out_dict["bmag_emit_err"]