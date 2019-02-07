#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2018-11-02
# @Filename: plot_all_sky.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2018-11-19 21:53:20


import matplotlib.pyplot
import numpy
import pandas
import scipy.interpolate


data_path = './guide_stars_2k_straw_fields_gof_3_11mu_6cam.hdf'

data_apo = pandas.read_hdf(data_path, 'gaia_APO').reset_index()
data_lco = pandas.read_hdf(data_path, 'gaia_LCO').reset_index()
data_lco.field += 10000

data = pandas.concat([data_apo, data_lco])

data_mag = data[data.g_mag == 17.][['ra', 'dec', 'n_valid']]

RA = data_mag.ra
Dec = data_mag.dec

org = 0
RA = numpy.remainder(RA + 360 - org, 360)
RA[RA > 180] -= 360
RA = -RA

ra_grid, dec_grid = numpy.mgrid[-180:180.1:0.1, -90:90.1:0.1]

Ti = scipy.interpolate.griddata((RA, Dec), data_mag.n_valid, (ra_grid, dec_grid))
Ti = numpy.ma.array(Ti, mask=numpy.isnan(Ti))

Ti[Ti > 15] = 15

fig = matplotlib.pyplot.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='mollweide')

im = ax.pcolormesh(numpy.radians(ra_grid), numpy.radians(dec_grid), Ti, vmin=5, vmax=15)

fig.colorbar(im, ax=ax, shrink=0.5, label='Number of guide stars')

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

ax.set_title(r'$\mathrm{6\times 4k\times 4k\,pixels;\;} G<17$')

fig.tight_layout()
fig.savefig('n_guide.png', dpi=300)
