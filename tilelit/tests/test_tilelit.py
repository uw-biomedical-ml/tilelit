from __future__ import absolute_import, division, print_function
import os.path as op
import numpy as np
import pandas as pd
import numpy.testing as npt
import tilelit as ti
import skimage.data as skd
from skimage import transform as tf


def test_tile():
    static = skd.coins()
    ty = np.random.randn() * 0.05 * static.shape[0]
    tx = np.random.randn() * 0.05 * static.shape[1]
    rot = np.random.randn() * np.pi/20
    affine = tf.AffineTransform(rotation=rot, translation=(tx, ty))
    moving = tf.warp(static, affine.inverse)
    img_initial, aff_initial = \
        ti.initial_alignment(static, moving, gaussian_blur=5)
    img_final, aff_final = \
        ti.fine_alignment(static, img_initial, starting_affine=aff_initial)

    npt.assert_almost_equal(aff_final, affine.params)
