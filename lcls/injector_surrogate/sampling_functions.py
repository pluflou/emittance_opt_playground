import sys
import numpy as np

#NN Surrogate model class
from .injector_surrogate_quads import Model
#from pyemittance.beam_io import MachineIO

#input params: solenoid and quads to vary
# NOTE swapped CQ01 and SQ01 to match machine order
opt_var_names = ['SOL1:solenoid_field_scale','CQ01:b1_gradient','SQ01:b1_gradient',
                 'QA01:b1_gradient','QA02:b1_gradient','QE01:b1_gradient','QE02:b1_gradient','QE03:b1_gradient',
                 'QE04:b1_gradient']

#output params: emittance in transverse plane (x & y)
opt_out_names = ['norm_emit_x','norm_emit_y']


def get_ground_truth(Model,ref_point,varx,vary,varz,var1,var2,var3,var4,var5,var6):
    '''Returns normalized emittance prediction from the surrogate model
       for given settings of SOL1, SQ01, and CQ01 '''
    
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
    emitx = y_out[:,Model.loc_out['norm_emit_x']] #grab norm_emit_x out of the model
    emity = y_out[:,Model.loc_out['norm_emit_y']] #grab norm_emit_y out of the model

    print("NOTE this only returns the scalar normalized emittance (not bmag) from the surrogate predictions.")
    return np.sqrt(emitx*emity)

def get_beamsize(Model, ref_point, varx, vary, varz, var1, var2, var3, var4, var5, var6, use_lcls=False):
    '''Returns the beamsize (xrms, yrms) prediction [m] from the surrogate model
    for given settings of SOL1, SQ01, CQ01 and scanning quad QE04 '''

    if use_lcls:
        # Use function from beam_io
        io = MachineIO('LCLS', 'WIRE')
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
    '''Returns the beamsize (xrms, yrms) prediction [m] from the surrogate model
    for given settings of SOL1, SQ01, CQ01 and scanning quad QE04 '''
    opt_var_names = ['SOL1:solenoid_field_scale','SQ01:b1_gradient','CQ01:b1_gradient','QE04:b1_gradient']
    
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