import numpy as np
from shapely.geometry import Polygon
from typing import List, Tuple

def distance_to_poly(
    pnt: Tuple, 
    poly: List[Tuple]
):
    """
    
    The function calculates the distance between point and polygon in 2D space in vectorized format using numpy library.
    It works with only valid polygons which do net self-intersect to avoid ambiguous calculations.
    
    Example
    -------

    point = (1.5,4)
    polygon = [(0,0), (1,0), (1.5,1), (1,2), (0,2), (1,1)]
    distance_to_poly(point, polygon)
    
    Parameters
    ----------
    
    point: Tuple
        Tuple of two numbers
    poly: List[Tuple]
        List of apex points of the polygon
    """

    # Chech validity of the polygon to avoid ambiguous situations
    p = Polygon(poly)
    if not p.is_valid:
        raise ValueError(f'Polygon is not valid')

    # Create point and segments of the polygon
    point = np.array(pnt, dtype=float)
    segment_starts = np.array(poly, dtype=float)
    segment_ends = np.roll(segment_starts, 2)

    # Tangent vectors calculation for each segment 
    tangents = np.divide(segment_ends-segment_starts,
                  np.linalg.norm(segment_ends-segment_starts, axis=1).reshape(len(poly),1))

    # Calculation of parallelm components
    parallel_comp_1 = np.multiply((segment_starts - point), tangents).sum(axis=1)
    parallel_comp_2 = np.multiply((point - segment_ends), tangents).sum(axis=1)

    # Clipping negative values
    shift = np.maximum.reduce([parallel_comp_1, parallel_comp_2, np.zeros(len(poly))])

    # Orthogonal part
    orthogonal = np.cross(point-segment_starts, tangents)

    # Returning minumem distance
    return np.hypot(shift, orthogonal).min()
