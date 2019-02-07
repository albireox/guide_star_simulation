#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2018-11-02
# @Filename: plot_guide_stars.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)
#
# @Last modified by: José Sánchez-Gallego (gallegoj@uw.edu)
# @Last modified time: 2019-01-09 14:55:32


import matplotlib.pyplot as plt
import numpy
import pandas
import seaborn


seaborn.set(style='darkgrid', palette='deep', color_codes=True)

chip_size = '1k'
observatory = 'APO'

gaia = pandas.read_hdf(f'./guide_stars_{chip_size}_straw_fields_gof_3_13mu_6cam.hdf',
                       f'gaia_{observatory}')['n_valid']

fig, axes = plt.subplots(2, 1, sharex=False)

for g_mag in gaia.index.levels[2]:

    gaia_mag = gaia.loc[:, :, g_mag]
    gaia_group = gaia_mag.groupby('field')

    all_stars, bins = numpy.histogram(gaia_group.sum(), density=True, bins=range(0, 26))
    min_stars, bins = numpy.histogram(gaia_group.min(), density=True, bins=range(0, 26))

    axes[0].plot(bins[0:-1], numpy.cumsum(all_stars[::-1])[::-1], marker='o',
                 markersize=3, label=f'$\mathrm{{G}} \\leq {g_mag}$', zorder=20)
    axes[1].plot(bins[0:-1], numpy.cumsum(min_stars[::-1])[::-1], marker='o',
                 markersize=3, zorder=20)

axes[0].axvline(x=15, zorder=10, linewidth=1, color='k', linestyle='--')
axes[1].axvline(x=3, zorder=10, linewidth=1, color='k', linestyle='--')

axes[0].set_title('All six GFA combined')
axes[0].set_ylabel('Inverse cumulative probability')
axes[1].set_title('GFA with the fewest stars')
axes[1].set_xlabel('Number of stars')
axes[1].set_ylabel('Inverse cumulative probability')

axes[0].legend()

fig.suptitle('Gaia guide stars. Six {0}$\\times${0} chips. '.format(chip_size) +
             'Pixel size=13$\\mathrm{\\mu}$m. GoF < 3.\n' + observatory,
             fontsize=16)

fig.savefig(f'gaia_{chip_size}_straw_gof_3_{observatory}_13mu_6cam.png')
