#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2018-11-16
# @Filename: plot_target_density.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2018-11-18 15:02:55

import os
from multiprocessing import Pool

import matplotlib.pyplot
import numpy
import pandas
from matplotlib.colors import LogNorm
from scipy import stats

from sdssdb.peewee.sdss5db import targetdb


assert targetdb.database.connected, 'database not connected.'

target_list_file = 'target_coords.hdf'

if not os.path.exists(target_list_file):

    query = targetdb.Target.select(targetdb.Target.ra, targetdb.Target.dec).tuples()
    ra, dec = zip(*query)

    data = pandas.DataFrame({'ra': ra, 'dec': dec})
    data.to_hdf(target_list_file, 'data')

else:

    data = pandas.read_hdf(target_list_file)


cpus = 2
pool = Pool(processes=cpus)

# data = data.sample(100000)

ra = data.ra
dec = data.dec

org = 0
ra = numpy.remainder(ra + 360 - org, 360)
ra[ra > 180] -= 360
ra = -ra

fig = matplotlib.pyplot.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='mollweide')

kde = stats.gaussian_kde([ra, dec])

ra_grid, dec_grid = numpy.mgrid[-180:181:1, -90:91:1]

data_split = numpy.array_split(numpy.c_[ra_grid.flat, dec_grid.flat].T, cpus, axis=1)
density = pool.map(kde, data_split)
density = numpy.hstack(density).reshape(ra_grid.shape)

numpy.save('target_density.npy', density)

norm = LogNorm(vmin=5e-7, vmax=density.max())
im = ax.pcolormesh(numpy.radians(ra_grid), numpy.radians(dec_grid), density, norm=norm)

fig.colorbar(im, ax=ax, shrink=0.5, label='Density of targets')

tick_labels = numpy.array([150., 120, 90, 60, 30, 0, 330, 300, 270, 240, 210])
tick_labels = numpy.remainder(tick_labels + 360 + org, 360)
tick_labels = numpy.array(tick_labels / 15., int)

tickStr = []
for tick_label in tick_labels[1::2]:
    tickStr.append('')
    tickStr.append('${0:d}^h$'.format(tick_label))

ax.set_xticklabels(tickStr, fontdict={'color': '0.8'})
ax.grid(True)

ax.set_xlabel(r'$\alpha_{2000}$')
ax.set_ylabel(r'$\delta_{2000}$')

fig.tight_layout()
fig.savefig('target_density.png', dpi=300)
