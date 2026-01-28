"""Target zone site objects for LIBERO environments.

This module defines simple site-based targets used for spatial predicates in
LIBERO task definitions. It relies on robosuite's MJCF utilities and the
`SiteObject` abstraction for representing geometric regions in the simulator.
"""

import re
import numpy as np
import robosuite.utils.transform_utils as T
import pathlib

absolute_path = pathlib.Path(__file__).parent.parent.parent.absolute()
from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import (
    xml_path_completion,
    array_to_string,
    find_elements,
    CustomMaterial,
    add_to_dict,
    RED,
    GREEN,
    BLUE,
)

# from robosuite.models.objects import BoxObject
from libero.libero.envs.objects.site_object import SiteObject

from libero.libero.envs.base_object import (
    register_visual_change_object,
    register_object,
)


@register_object
class TargetZone(SiteObject):
    """Site-based target region used for containment and proximity checks."""

    def __init__(
        self,
        name,
        zone_height=0.007,
        z_offset=0.02,
        rgba=(1, 0, 0, 1),
        joints=None,
        zone_size=(0.15, 0.05),
        zone_centroid_xy=(0, 0),
        # site_type="box",
        # site_pos="0 0 0",
        # site_quat="1 0 0 0",
    ):
        """Initialize a target zone site.

        Args:
            name (str): Unique name for the target zone.
            zone_height (float): Height (z) of the zone.
            z_offset (float): Vertical offset applied to the centroid.
            rgba (tuple): RGBA color for visualization.
            joints (list | None): Optional joint definitions for the zone.
            zone_size (tuple): (x, y) extents of the zone.
            zone_centroid_xy (tuple): (x, y) centroid of the zone.
        """
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()
        self.size = (zone_size[0], zone_size[1], zone_height)
        self.pos = zone_centroid_xy + (z_offset,)
        self.quat = (1, 0, 0, 0)
        super().__init__(
            name=name,
            size=self.size,
            rgba=rgba,
            site_type="box",
            site_pos=array_to_string(self.pos),
            site_quat=array_to_string(self.quat),
        )

    def in_box(self, this_position, this_mat, other_position):
        """Check whether a point lies within the zone's bounding box.

        Args:
            this_position (np.ndarray): World position of the target zone.
            this_mat (np.ndarray): Rotation matrix for the target zone.
            other_position (np.ndarray): World position of the queried object.

        Returns:
            bool: True if the point lies within the axis-aligned bounds.
        """

        total_size = np.abs(this_mat @ self.size)

        ub = this_position + total_size
        lb = this_position - total_size

        lb[2] -= 0.01
        return np.all(other_position > lb) and np.all(other_position < ub)

    def on_top(self, this_position, this_mat, other_position):
        """Check whether a point lies above the zone's upper bound.

        Args:
            this_position (np.ndarray): World position of the target zone.
            this_mat (np.ndarray): Rotation matrix for the target zone.
            other_position (np.ndarray): World position of the queried object.

        Returns:
            bool: True if the point lies above the computed upper bound.
        """

        # The size is rotated into world coordinates and treated as a conservative
        # bounding box for the "above" check.
        total_size = np.abs(this_mat @ self.size)
        ub = this_position + total_size
        return np.all(other_position > ub)
