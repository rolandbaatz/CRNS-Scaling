"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.

Authors:
Roland Baatz roland.baatz @ zalf.de

Maintainers and contact:
Currently maintained by the authors.

Copyright (C) Leibniz Centre for Agricultural Landscape Research (ZALF)
"""

import numpy as np

def Wd(d, r, bd, y):
    """Wd Weighting function to be applied on samples to calculate weighted impact of 
    soil samples based on depth.

    Parameters
    ----------
    d : float
        depth of sample (cm)
    r : float,int
        radial distance from sensor (m)
    bd : float
        bulk density (g/cm^3)
    y : float
        Soil Moisture from 0.02 to 0.50 in m^3/m^3
    """
    return np.exp(-2 * d / D86(r, bd, y))

def D86(r, bd, y):
    """D86 Calculates the depth of sensor measurement (taken as the depth from which
    86% of neutrons originate)

    Parameters
    ----------
    r : float, int
        radial distance from sensor (m)
    bd : float
        bulk density (g/cm^3)
    y : float
        Soil Moisture from 0.02 to 0.50 in m^3/m^3
    """
    return 1 / bd * (8.321 + 0.14249 * (0.96655 + np.exp(-0.01 * r)) * (20 + y) / (0.0429 + y))

# Given values
depths = [5, 15]  # depths in cm
swc_values = [0.05, 0.15]  # Soil Water Content values
bulk_density = 1.4  # assumed bulk density in g/cm^3 (example value)
radial_distance = 0  # assuming no radial distribution

# Calculate weighted SWC
weighted_swcs = []
for d, swc in zip(depths, swc_values):
    weight = Wd(d, radial_distance, bulk_density, swc)
    weighted_swcs.append(swc * weight)

# Print results
for depth, swc, weighted_swc in zip(depths, swc_values, weighted_swcs):
    print(f"Depth: {depth} cm, SWC: {swc}, Weighted SWC: {weighted_swc}")

# Calculate average weighted SWC
average_weighted_swc = sum(weighted_swcs) / len(weighted_swcs)
print(f"Average Weighted SWC: {average_weighted_swc}")
