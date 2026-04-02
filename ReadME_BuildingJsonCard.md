This file contains the instructions to build a consistant set of JSON cards for the simulation to run smoothly

# young modulus 
defined by the keys: "young" and "young_type"
for simple inclusions : sphere, active_sphere and matrix : young type is implicit Constant 
For spheroids : young_type in ['Constant', 'Plateau-Ramp-Plateau', 'Exponential']
    for Constanttype : young -> int or float
    for Plateau-Ramp-Plateau -> "young" : list of lists -> [[x0, y0], [x1, y1]]
    for Exponential -> "young": list of lists -> [[e0, e1, ...], [k0, k1, ...], [l0, l1, ...]] > same lengths lists
    
# orientation 
defined using the polar angles; see function local_basis(theta, phi):
    Compute the local basis vectors using the spherical coordinate system and axisymmetry around the radial direction (e_r).
    Definition and convention used can be found on : https://en.wikipedia.org/wiki/Spherical_coordinate_system
    This is based on the ISO Standard : ISO 80000-2:2019
    
theta and phi in the json cards MUST BE given in degree. Inclusions class automatically translate those into radiant
