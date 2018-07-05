"""tilelit : make coextensive tiles from massive images."""
import os.path as op
import numpy as np

import skimage.io as sio
from skimage.measure import ransac
from skimage.feature import match_descriptors, ORB
from skimage import transform as tf
from skimage.filters import gaussian

from dipy.viz.regtools import overlay_images
from dipy.align.imaffine import (MutualInformationMetric,
                                 AffineRegistration)

from dipy.align.transforms import AffineTransform2D

import tifffile


def tiff_resize(fname, factor=10):
    """
    Resize a massive tiff file by a factor.

    Paramters
    ---------
    fname : str
        The name of the image file to read

    factor : int
        The compression factor (default: 10)
    """
    massive = sio.imread(fname)
    return tf.resize(massive,
                     (massive.shape[0]//factor,
                      massive.shape[1]//factor))


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


def register_case(case, chan=2, factor=10):
    """ """
    he_file = '/home/ubuntu/data/%s_HE.tiff'%case
    er_file = '/home/ubuntu/data/%s_ER.tiff'%case

    im_he = tiff_resize(he_file, factor=factor)
    im_stain = tiff_resize(er_file, factor=factor)
    
    img_stain_bw = im_stain[..., chan]
    img_he_bw = im_he[..., chan]
    
    # Plot before registration
    im1, im2 = plot_together(1-img_he_bw, 1-(img_stain_bw/img_stain_bw.max()))
    # Register
    img_final, aff_final = fine_alignment(img_stain_bw, img_he_bw, starting_affine=np.eye(3))
    # Plot again after registration
    new_image1, new_image2 = plot_together(1-(img_final/img_final.max()), 1-(img_stain_bw/img_stain_bw.max()))
    
    return aff_final


def make_tiles(case, bucket, aff, box_size=(280, 280)):
    im_stain_orig = tifffile.imread('/home/ubuntu/data/%s_ER.tiff'%case)
    im_he_orig = tifffile.imread('/home/ubuntu/data/%s_HE.tiff'%case)
    ii = 1
    for y_corner in np.arange(0, im_stain_orig.shape[0], box_size[0]):
        for x_corner in np.arange(0, im_stain_orig.shape[1], box_size[1]):
            y_center = y_corner + box_size[0] // 2
            x_center = x_corner + box_size[1] // 2
            x_coords, y_coords = np.meshgrid(np.arange(x_center - box_size[1] // 2, x_center + box_size[1] // 2),
                                             np.arange(y_center - box_size[0] // 2, y_center + box_size[0] // 2))

            coords = np.array(list(zip(x_coords.ravel(), y_coords.ravel(), np.ones(np.product(box_size))))).astype(int)
            if not (np.any(coords[:, 0] > im_stain_orig.shape[0]) or 
                    np.any(coords[:, 1] > im_stain_orig.shape[1])):
                im_stain_tile = im_stain_orig[(coords[:, 0], coords[:, 1])].reshape(box_size + (-1, ))
                trans_coords = (np.round(np.dot(aff, coords.T))).T.astype(int)
                if not (np.any(trans_coords < 0) or 
                        np.any(trans_coords[:, 0] >= im_he_orig.shape[0]) or 
                        np.any(trans_coords[:, 1] >= im_he_orig.shape[1])):
                    im_he_tile = im_he_orig[(trans_coords[:, 0], trans_coords[:, 1])].reshape(box_size + (-1, ))
                    mean_tile = im_he_tile[:, :, :3].mean()
                    if mean_tile < 240:
                        local_he_trans = './tile-he-%s.png'%case
                        tifffile.imsave(local_he_trans, im_he_tile)                
                        remote_he = op.join("BreastDeepLearning/RU-18-160-sections-8-12/TIFF/tiles/%s/HE/"%case,
                                            'tile%08d.png' % ii)
                        bucket.upload_file(local_he_trans, remote_he)
                        local_stain ='./tile-er-%s.png'%case
                        tifffile.imsave(local_stain, im_stain_tile)
                        remote_stain = op.join("BreastDeepLearning/RU-18-160-sections-8-12/TIFF/tiles/%s/ER/"%case, 
                                               'tile%08d.png' % ii)
                        if not np.mod(ii, 50):
                            print(ii)
                        bucket.upload_file(local_stain, remote_stain)
                        ii = ii + 1                    