#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2019-01-24
# @Filename: simulate_guide_archive.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2019-02-07 11:34:12

import multiprocessing
import pathlib

import numpy
import pandas

from cherno.tools.fitting import filter_gprobes, guider_fit


n_samples = 5

data_path = pathlib.Path('~/sdss09/QA/guider_tests').expanduser()

observatory = 'APO'
platescale = 217.7358 if observatory == 'APO' else 330.275


def create_simple_mesh(radius=1.5, m_width=None, platescale=217.7358):
    """Creates a simple mesh for the field, in plate scale mm."""

    if m_width is None:
        # If no width is defined, uses an adaptive value based on the radius.
        m_width = 0.2 * radius / 1.5 * platescale

    size = radius * platescale

    coords = numpy.meshgrid(numpy.arange(-size, size + m_width, m_width),
                            numpy.arange(-size, size + m_width, m_width))

    return coords


def calculate_distribution_estimator(targets, platescale=217.7358):
    """Returns a metric of how well distributed the targets are on a plate."""

    max_radius = 1.5 * platescale

    targets = targets[['xFocal', 'yFocal']]

    mesh = create_simple_mesh(platescale=platescale)
    x, y = mesh

    efield_vec = numpy.zeros((2, mesh[0].shape[0], mesh[0].shape[1]))

    for __, data in targets.iterrows():
        xf, yf = data
        distances = numpy.sqrt((x - xf)**2 + (y - yf)**2)
        uu = mesh - numpy.array([[[xf, yf]]]).T
        uu /= numpy.sqrt(numpy.sum(uu**2))
        efield_vec += uu / distances ** 2

    efield = numpy.sqrt(numpy.sum(efield_vec**2, axis=0)) / len(targets)

    central_distance = numpy.sqrt(x**2 + y**2)
    efield[central_distance > max_radius] = 0.0

    return numpy.var(efield)


def simulate(df):

    rows = []

    for index, gprobes in df.groupby(level=0):

        gprobes_filt = filter_gprobes(gprobes)
        if len(gprobes_filt) < 10:
            continue

        orig_distribution_estimate = calculate_distribution_estimator(
            gprobes_filt, platescale=platescale)
        orig_fit = guider_fit(gprobes_filt, plugplate_scale=platescale)

        for ii in range(n_samples):

            n_gprobes_sample = numpy.random.randint(3, len(gprobes_filt))
            sample = gprobes_filt.sample(n_gprobes_sample)

            sample_distribution_estimate = calculate_distribution_estimator(
                sample, platescale=platescale)
            sample_fit = guider_fit(sample, plugplate_scale=platescale)

            rows.append([
                index,
                len(gprobes_filt),
                orig_distribution_estimate,
                orig_fit.mRA,
                orig_fit.mDec,
                orig_fit.mRot,
                orig_fit.mScale,
                orig_fit.guideRMS,
                orig_fit.pos_error,
                n_gprobes_sample,
                sample_distribution_estimate,
                sample_fit.mRA,
                sample_fit.mDec,
                sample_fit.mRot,
                sample_fit.mScale,
                sample_fit.guideRMS,
                sample_fit.pos_error])

    df_return = pandas.DataFrame.from_records(
        rows,
        columns=['index',
                 'orig_n_stars',
                 'orig_distribution_estimate',
                 'orig_fit_mRA',
                 'orig_fit_mDec',
                 'orig_fit_mRot',
                 'orig_fit_mScale',
                 'orig_fit_guideRMS',
                 'orig_fit_pos_error',
                 'sample_n_stars',
                 'sample_distribution_estimate',
                 'sample_fit_mRA',
                 'sample_fit_mDec',
                 'sample_fit_mRot',
                 'sample_fit_mScale',
                 'sample_fit_guideRMS',
                 'sample_fit_pos_error'])

    return df_return


bintable = pandas.read_hdf(data_path / f'guider_{observatory.lower()}.h5', 'bintable')

cores = multiprocessing.cpu_count()
data_split = numpy.array_split(bintable, cores)
pool = multiprocessing.Pool(cores)
data = pandas.concat(pool.map(simulate, data_split))
pool.close()
pool.join()

data.to_hdf('guide_simulate.h5', observatory.lower())
