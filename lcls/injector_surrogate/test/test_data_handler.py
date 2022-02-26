import numpy as np

import sys

sys.path.append('./src/lcls/injector_surrogate/')
from data_handler import check_symmetry, find_inflection_pnt

xquad = [-6.5, -6, -5.38210006, -4.24154984, -3.71824052, -3.17573903, -1.99949716, -1, 0, 1, 2] # kG, same sign for x and y
xrms = [1.2e-04, 1.47e-04, 1.59568569e-04, 8.75908038e-05, 6.55396626e-05, 4.85704395e-05, 7.99284069e-05, 7.99284069e-05, 6.49284069e-05,6.55396626e-05, 4.85704395e-05]
# xquad = [-6.5, -6, -5.38210006, -4.24154984, -3.71824052, -3.17573903, -1.99949716] # kG, same sign for x and y
# xrms = [1.2e-04, 1.47e-04, 1.59568569e-04, 8.75908038e-05, 6.55396626e-05, 4.85704395e-05, 7.99284069e-05]
yquad = [-5.38210006, -4.24154984, -3.71824052, -3.17573903, -1.99949716] # kG, same sign for x and y
yrms = [9.65669677e-05, 5.61440470e-05, 4.89721770e-05, 4.80127438e-05, 5.86293035e-05]

# xquad = [-5.38210006, -4.24154984, -3.71824052, -3.17573903, -1.99949716]
# yquad = [-5.38210006, -4.24154984, -3.71824052, -3.17573903, -1.99949716]
# xrms = [1.59568569e-04, 8.75908038e-05, 6.55396626e-05, 4.85704395e-05, 7.99284069e-05]
# yrms = [9.65669677e-05, 5.61440470e-05, 4.89721770e-05, 4.80127438e-05, 5.86293035e-05]

def test_check_symmetry(xrms, yrms, xquad, yquad):
    print(check_symmetry(xrms,xquad,"x"))
    print(check_symmetry(yrms,yquad,"y"))

def test_find_inflection_pnt(xrms,yrms,xquad,yquad):
    print(find_inflection_pnt(xrms, xquad, show_plots=True))
    print(find_inflection_pnt(yrms, yquad, show_plots=True))

if __name__ == "__main__":
    test_check_symmetry(xrms, yrms, xquad, yquad)
    test_find_inflection_pnt(xrms, yrms, xquad, yquad)