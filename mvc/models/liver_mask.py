"""Coarse liver ROI for MR magnitude images.

This is not a liver segmentation.  Otsu on an abdominal magnitude image
often keeps the bright body wall and can drop the liver, because fat and
liver are different modes.  The ROI below treats dark corners as air, keeps
every brighter tissue class, then removes a physical rim so the sliding
chest and abdominal wall are outside the metric.  The largest remaining
component is the coarse liver.  A spleen, a stomach, or a wall thicker
than the rim can still win; callers that do not set
``mask_kind="coarse_liver"`` never reach this function.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import SimpleITK as sitk


# 20 mm is a typical adult abdominal wall (skin, fat, and muscle).  It is
# large enough to drop the rim Otsu keeps, and smaller than the short axis
# of a normal right lobe, so the liver core remains.  A thinner subject
# keeps the filled body instead of an empty mask; see ``coarse_liver_mask``.
COARSE_LIVER_WALL_MM = 20.0

# Corners of an axial abdominal FOV are air.  8% of the shortest axis is
# wide enough that one bright coil pixel does not move the median, and the
# span is also capped at a quarter of that axis so the four (or eight)
# windows cannot cover the liver.
COARSE_LIVER_CORNER_FRACTION = 0.08

# 1.4826 * MAD is a robust SD.  Five of those above the corner median sits
# over air noise and under liver, fat, and muscle, so a later erosion is
# what removes the wall.  Otsu is free to cut between fat and liver instead.
COARSE_LIVER_NOISE_K = 5.0

# Close gaps the air threshold opens along the capsule (partial-volume
# lines of a few millimetres).  The close is at most half the wall radius,
# so tissue it borrows from outside the skin still lies in the rim that the
# following erosion deletes.  It does not fill a large enclosed stomach;
# hole filling does that, and that limitation is documented on the function.
COARSE_LIVER_CLOSE_MM = 5.0

# Consistency constant between MAD and a Gaussian SD.
_MAD_TO_SIGMA = 1.4826


def require_positive_mm(name, value):
    """Return ``value`` as a finite positive float, or raise."""
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a positive number, got {value!r}")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return numeric


def coarse_liver_mask(image, wall_mm=COARSE_LIVER_WALL_MM):
    """Return a UInt8 mask, 1 on the coarse liver interior.

    ``image`` is a SimpleITK magnitude image in 2D or 3D.  The input is
    not modified.  Non-finite voxels are set darker than every finite
    voxel, so the threshold cannot mark them as tissue.  One that is
    enclosed by tissue is filled later, the same way a vessel is.

    Steps, in order:

    1. Air threshold from the corners when those corners are darker than
       the interior.  Otherwise Otsu, which is the right split when the
       FOV has no air rim (the corner estimate would be tissue).
    2. Morphological close of ``min(5 mm, wall_mm / 2)``.  A radius that
       rounds to zero voxels is skipped.  It is not forced up to one
       voxel: on coarse spacing that voxel would be thicker than the wall
       the next step removes.
    3. Largest face-connected component, then hole filling.  Distant
       bright specks drop out.  Enclosed cavities, including a stomach
       surrounded by tissue, are filled and can remain in the ROI.
    4. Erode by ``wall_mm``.  The radius is at least one voxel, matching
       the Otsu dilation, so a spacing coarser than the wall still pulls
       the mask in.  Voxels outside the image count as background, so a
       mask that fills the FOV still loses a rim.  The largest remaining
       component is the ROI.  A constant image has no air/tissue split:
       Otsu marks no tissue and this raises, rather than calling the
       whole FOV liver.
    5. If that erosion deletes everything, the object is thinner than the
       wall prior.  Return the filled body from step 3.  An empty metric
       mask fails registration; keeping the body still drops air and
       distant specks, and does not pretend the wall was separated.

    The mask is computed in the image's own geometry.  Registration samples
    it at the initial pose, as it does for the Otsu and percentile masks.
    """
    wall_mm = require_positive_mm("liver_wall_mm", wall_mm)
    cleaned = _cleaned_image(image)
    tissue = _tissue_mask(cleaned)
    if not _any(tissue):
        raise ValueError(
            "coarse liver mask found no tissue above the air threshold"
        )
    closed = _close_tissue(tissue, cleaned, wall_mm)
    body = _fill_holes(_largest_component(closed))
    if not _any(body):
        raise ValueError(
            "coarse liver mask found no tissue above the air threshold"
        )
    eroded = _erode(body, cleaned, wall_mm)
    if not _any(eroded):
        return body
    return _largest_component(eroded)


def _cleaned_image(image):
    """Finite copy of ``image``.  Non-finite voxels become darker than tissue."""
    array = np.asarray(sitk.GetArrayFromImage(image), dtype=np.float64)
    finite = np.isfinite(array)
    if not np.any(finite):
        raise ValueError("coarse liver mask needs a finite image")
    if not np.all(finite):
        array = array.copy()
        array[~finite] = float(np.min(array[finite])) - 1.0
    cleaned = sitk.GetImageFromArray(array)
    cleaned.CopyInformation(image)
    return cleaned


def _tissue_mask(image):
    """UInt8 tissue mask.  1 is tissue, 0 is air or background."""
    array = sitk.GetArrayFromImage(image)
    use_corners, threshold = _corner_threshold(array)
    if not use_corners:
        # Same inside/outside values as the dilated Otsu mask: low air is 0,
        # the high class is 1.  On a wall-and-liver histogram that high class
        # can be the wall alone, which is why erosion still follows.
        mask = sitk.OtsuThreshold(image, 0, 1)
        return sitk.Cast(mask, sitk.sitkUInt8)
    tissue = np.asarray(array > threshold, dtype=np.uint8)
    mask = sitk.GetImageFromArray(tissue)
    mask.CopyInformation(image)
    return mask


def _corner_threshold(array):
    """Return ``(use_corners, threshold)``.

    ``use_corners`` is false when the corners are not a darker air rim.
    The threshold is then ignored and the caller uses Otsu.
    """
    span = _corner_span(array.shape)
    if span < 1:
        return False, 0.0
    corners = _finite_values(_corner_values(array, span))
    interior = _finite_values(_interior_values(array, span))
    if corners.size == 0 or interior.size == 0:
        return False, 0.0
    corner_median = float(np.median(corners))
    interior_median = float(np.median(interior))
    # Equal medians mean the corners are not air.  A strict comparison
    # also rejects a flat image, where a corner threshold would be arbitrary.
    if not (corner_median < interior_median):
        return False, 0.0
    mad = float(np.median(np.abs(corners - corner_median)))
    return True, corner_median + COARSE_LIVER_NOISE_K * _MAD_TO_SIGMA * mad


def _corner_span(shape):
    """Width of one corner window, or 0 when the image is too small."""
    if not shape:
        return 0
    shortest = min(int(size) for size in shape)
    # Four windows of this size must leave an interior.  shortest // 4
    # is 0 for an image under 4 voxels wide; the caller then uses Otsu.
    cap = shortest // 4
    if cap < 1:
        return 0
    span = int(round(COARSE_LIVER_CORNER_FRACTION * shortest))
    span = max(1, span)
    return min(span, cap)


def _corner_values(array, span):
    """Voxels in every corner hyper-rectangle, including overlapping ones."""
    windows = [
        (slice(0, span), slice(size - span, size))
        for size in array.shape
    ]
    blocks = [array[choice].ravel() for choice in itertools.product(*windows)]
    if not blocks:
        return np.empty(0, dtype=float)
    return np.concatenate(blocks)


def _interior_values(array, span):
    """Voxels inside the corner rim.  Empty when the rim consumes the image."""
    slices = tuple(slice(span, size - span) for size in array.shape)
    if any(item.start >= item.stop for item in slices):
        return np.empty(0, dtype=float)
    return np.asarray(array[slices]).ravel()


def _finite_values(values):
    values = np.asarray(values, dtype=np.float64).ravel()
    return values[np.isfinite(values)]


def _close_tissue(mask, image, wall_mm):
    close_mm = min(COARSE_LIVER_CLOSE_MM, float(wall_mm) / 2.0)
    radii = _voxel_radii(image, close_mm, minimum=0)
    if all(radius == 0 for radius in radii):
        return mask
    closed = sitk.BinaryMorphologicalClosing(mask, radii)
    return sitk.Cast(closed, sitk.sitkUInt8)


def _erode(mask, image, wall_mm):
    """Pull the mask in by ``wall_mm``.

    ``boundaryToForeground`` defaults to true, which treats voxels outside
    the image as tissue.  A mask that reaches the FOV edge would then not
    shrink.  Outside the image is air, so the boundary is background.
    """
    radii = _voxel_radii(image, wall_mm, minimum=1)
    eroded = sitk.BinaryErode(
        mask,
        radii,
        sitk.sitkBall,
        0,
        1,
        False,
    )
    return sitk.Cast(eroded, sitk.sitkUInt8)


def _voxel_radii(image, radius_mm, minimum):
    """Physical radius converted per axis.  ``minimum`` is the voxel floor."""
    radii = []
    for step in image.GetSpacing():
        spacing = float(step)
        if not (spacing > 0.0):
            raise ValueError(f"spacing must be positive, got {step!r}")
        voxels = int(round(float(radius_mm) / spacing))
        if minimum:
            voxels = max(int(minimum), voxels)
        else:
            voxels = max(0, voxels)
        radii.append(voxels)
    return radii


def _largest_component(mask):
    """Largest face-connected foreground component, as UInt8 0/1."""
    labeled = sitk.ConnectedComponent(mask)
    # Label 1 is the largest object.  An empty mask has no label 1.
    relabeled = sitk.RelabelComponent(labeled, sortByObjectSize=True)
    largest = sitk.Cast(relabeled == 1, sitk.sitkUInt8)
    return largest


def _fill_holes(mask):
    filled = sitk.BinaryFillhole(mask)
    return sitk.Cast(filled, sitk.sitkUInt8)


def _any(mask):
    return int(np.count_nonzero(sitk.GetArrayFromImage(mask))) > 0
