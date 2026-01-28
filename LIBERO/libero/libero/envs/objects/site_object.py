"""Site-based object utilities for LIBERO environments.

This module provides a lightweight representation of site geometry used to
define spatial regions for predicates and task logic in LIBERO. It assumes
robosuite coordinate conventions and MJCF string serialization utilities.
"""

import os
import numpy as np

from robosuite.utils.mjcf_utils import string_to_array
import robosuite.utils.transform_utils as transform_utils

import pathlib

absolute_path = pathlib.Path(__file__).parent.parent.parent.absolute()


class SiteObject:
    """Geometric site representation used for spatial containment checks."""

    def __init__(
        self,
        name,
        parent_name=None,
        joints=None,
        size=None,
        rgba=None,
        site_type="box",
        site_pos="0 0 0",
        site_quat="1 0 0 0",
        object_properties={},
    ):
        """Initialize a site object definition.

        Args:
            name (str): Unique name for the site.
            parent_name (str | None): Parent body name, if applicable.
            joints (list | None): Joint definitions for the site.
            size (np.ndarray | str | None): Size parameters or MJCF string.
            rgba (tuple | None): RGBA color for visualization.
            site_type (str): MJCF site type (e.g., "box").
            site_pos (str): MJCF position string.
            site_quat (str): MJCF quaternion string.
            object_properties (dict): Additional metadata for the site.
        """
        self.name = name
        self.parent_name = parent_name
        self.joints = joints
        self.site_pos = string_to_array(site_pos)
        self.site_quat = string_to_array(site_quat)
        self.size = size if type(size) is not str else string_to_array(size)
        self.rgba = rgba
        self.site_type = site_type
        self.object_properties = object_properties

    def in_box(self, this_position, this_mat, other_position):
        """Check whether a point lies within this site's bounding box.

        Args:
            this_position (np.ndarray): World position of the site.
            this_mat (np.ndarray): Rotation matrix for the site.
            other_position (np.ndarray): World position of the queried object.

        Returns:
            bool: True if the point lies within the axis-aligned bounds.
        """

        # Treat the rotated size as a conservative axis-aligned bound for
        # containment checks.
        total_size = np.abs(this_mat @ self.size)

        ub = this_position + total_size
        lb = this_position - total_size

        lb[2] -= 0.01
        return np.all(other_position > lb) and np.all(other_position < ub)

    def __str__(self):
        """Return a human-readable description of the site."""
        return (
            f"Object {self.name} : \n geom type: {self.site_type} \n size: {self.size}"
        )

    def under(self, this_position, this_mat, other_position, other_height=0.10):
        """Check whether a point lies above the site within a height window.

        Args:
            this_position (np.ndarray): World position of the site.
            this_mat (np.ndarray): Rotation matrix for the site.
            other_position (np.ndarray): World position of the queried object.
            other_height (float): Height tolerance above the site.

        Returns:
            bool: True if the point lies within the horizontal bounds and above
                the site within the provided height.
        """
        total_size = self.size  # np.abs(this_mat @ self.size)

        delta_position = this_mat @ (other_position - this_position)
        return total_size[2] - 0.005 < delta_position[2] < total_size[
            2
        ] + other_height and np.all(np.abs(delta_position[:2]) < total_size[:2])
