from lcls.injector_surrogate.sampling_functions import evaluate
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 9 inputs: SOL1, CQ1, SQ1, QA1, QA2, QE1, QE2, QE3, QE4

# sim bounds over which NN was trained
bounds = ((0.46, 0.485),(-0.02, 0.02),(-0.02, 0.02),
          (-4, -1),(1, 4),(-7,-1),(-1, 7),(-1, 7),(-7, 1))

# config has to have (N,9) shape, where N is the number of sample points
# values can be outside bounds but results not reliable
config = [[np.random.uniform(t[0],t[1]) for t in bounds],
          [np.random.uniform(t[0],t[1]) for t in bounds]]

out = evaluate(config)

for i in range(len(config)):
    print(f"Example config. {i}: {config[i]}")
    print(f"Transverse emittance in x-plane: {out['emitx'][i]:.3E} m.rad")
    print(f"Transverse emittance in y-plane: {out['emity'][i]:.3E} m.rad")
    print(f"Geometric mean of emittances: {out['emit_geo_mean'][i]:.3E} m.rad")
    print(f"Beamsize in x-plane: {out['sigma_x'][i]:.3E} m")
    print(f"Beamsize in y-plane: {out['sigma_y'][i]:.3E} m")


