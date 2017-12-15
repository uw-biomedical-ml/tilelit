"""tilelit : make coextensive tiles from massive images."""
import numpy as np

from skimage.measure import ransac
from skimage.feature import match_descriptors, ORB
from skimage import transform as tf
from skimage.filters import gaussian

from dipy.viz.regtools import overlay_images
from dipy.align.imaffine import (MutualInformationMetric,
                                 AffineRegistration)

from dipy.align.transforms import AffineTransform2D


def plot_together(image1, image2):
    """Plot a comparison of two images."""
    new_shape1 = list(image1.shape)
    new_shape2 = list(image2.shape)

    if image1.shape[0] < image2.shape[0]:
        new_shape1[0] = image2.shape[0]
    elif image1.shape[0] > image2.shape[0]:
        new_shape2[0] = image1.shape[0]

    if image1.shape[1] < image2.shape[1]:
        new_shape1[1] = image2.shape[1]
    elif image1.shape[1] > image2.shape[1]:
        new_shape2[1] = image1.shape[1]

    if new_shape1 != image1.shape:
        new_image1 = np.zeros(new_shape1, dtype=image1.dtype)
        new_image1[:image1.shape[0], :image1.shape[1]] = image1
        image1 = new_image1

    if new_shape2 != image2.shape:
        new_image2 = np.zeros(new_shape2, dtype=image2.dtype)
        new_image2[:image2.shape[0], :image2.shape[1]] = image2
        image2 = new_image2

    fig = overlay_images(image1, image2)
    fig.set_size_inches([12, 10])
    return new_image1, new_image2


def initial_alignment(static, moving, gaussian_blur=5):
    """Rough initial alignment of two images to each others.

    Uses RANSAC matched on ORB descriptors.

    Parameters
    ----------
    static : array
        The reference image.
    moving : array
        The moving image.
    gaussian_blur : int/float
        The degree of blurring to apply to the images before detecting and
        extracting ORB descriptors

    Returns
    -------
    img_warp : array
        The moving image warped towards the static image
    affine : array
        The affine transformation for this warping
    """
    descriptor_extractor = ORB()

    descriptor_extractor.detect_and_extract(gaussian(static, gaussian_blur))
    keypoints_left = descriptor_extractor.keypoints
    descriptors_left = descriptor_extractor.descriptors

    descriptor_extractor.detect_and_extract(gaussian(moving, gaussian_blur))
    keypoints_right = descriptor_extractor.keypoints
    descriptors_right = descriptor_extractor.descriptors

    matches = match_descriptors(descriptors_left, descriptors_right,
                                cross_check=True)

    model, inliers = ransac((keypoints_left[matches[:, 0]],
                             keypoints_right[matches[:, 1]]),
                            tf.AffineTransform,
                            min_samples=8,
                            residual_threshold=10,
                            max_trials=5000)

    img_warp = tf.warp(moving, model.inverse)
    return img_warp, model.params


def fine_alignment(static, moving, starting_affine=None):
    """Use mutual information to align two images.

    Parameters
    ----------
    static : array
        The reference image.
    moving : array
        The moving image.
    starting_affine : array
        A proposed initial transformation

    Returns
    -------
    img_warp : array
        The moving image warped towards the static image
    affine : array
        The affine transformation for this warping

    """
    metric = MutualInformationMetric()
    reggy = AffineRegistration(metric=metric)
    transform = AffineTransform2D()
    affine = reggy.optimize(static, moving, transform, None,
                            starting_affine=starting_affine)
    img_warp = affine.transform(moving)
    return img_warp, affine.affine
