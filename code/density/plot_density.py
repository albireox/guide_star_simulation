#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2018-11-16
# @Filename: plot_density.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2018-11-18 00:08:05


import matplotlib.pyplot
import numpy
import pandas
import scipy.interpolate
from matplotlib.colors import LogNorm


data = pandas.read_hdf('./guide_stars_density.hdf')


def plot_density(RA, Dec, density):

    lon = numpy.linspace(-180., 180., 360)
    lat = numpy.linspace(-90., 90., 180)
    lon_grid, lat_grid = numpy.meshgrid(lon, lat)

    Ti = scipy.interpolate.griddata((RA, Dec), density, (lon_grid, lat_grid))
    Ti = numpy.ma.array(Ti, mask=numpy.isnan(Ti))

    fig = matplotlib.pyplot.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='mollweide')

    norm = LogNorm(vmin=1e3, vmax=1e5)
    im = ax.pcolormesh(numpy.radians(lon_grid), numpy.radians(lat_grid), Ti, norm=norm)
    fig.colorbar(im, ax=ax, shrink=0.5, label=r'Density of stars $\mathregular{[deg^{-1}]}$')

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

    return fig, ax


for mag_cut in data.mag_cut.unique():

    data_mag = data[(data.mag_cut == mag_cut) & (data.density > 0)]
    RA = data_mag.ra
    Dec = data_mag.dec
    density = data_mag.density

    org = 0
    RA = numpy.remainder(RA + 360 - org, 360)
    RA[RA > 180] -= 360
    RA = -RA

    fig, ax = plot_density(RA, Dec, density)
    ax.set_title(f'$G<{mag_cut}$')
    fig.tight_layout()
    fig.savefig(f'guide_stars_g{mag_cut:d}_density.png', dpi=300)

    matplotlib.pyplot.close(fig)
