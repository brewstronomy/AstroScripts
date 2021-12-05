#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: luca (lucabeale@gmail.com)
"""

import aplpy, numpy as np, matplotlib.pyplot as plt, os.path as path
from PiscesA import constant as con
from astropy.io import fits
from astropy.wcs import WCS
from scipy.optimize import leastsq

# # # preliminaries
# # ignore warnings
import warnings
from astropy.utils.exceptions import AstropyWarning

warnings.filterwarnings('ignore', category=UserWarning, append=True)
warnings.simplefilter('ignore', category=AstropyWarning)

# # fitting function
def gauss2d(pars, x, y):
    x0, y0, sigX, sigY, theta, amp = pars
    theta_rad = np.radians(theta)
    a = np.cos(theta_rad)**2 / (2 * sigX**2) + np.sin(theta_rad)**2 / (2 * sigY**2)
    b = np.sin(2 * theta_rad) / 4 * (1 / sigY**2 - 1 / sigX**2)
    c = np.sin(theta_rad)**2 / (2 * sigX**2) + np.cos(theta_rad)**2 / (2 * sigY**2)
    term1 = a * (x - x0)**2
    term2 = 2 * b * (x - x0) * (y - y0)
    term3 = c * (y - y0)**2
    return amp * np.exp(-(term1 + term2 + term3))

# # plotting function
def plotPA(cubename, cubepath, cubelabel, levels, args_recenter, save, results, simulate, kw_fit, kw_save=None,
           kw_sim=None):
    # extract and parse
    file = path.join(cubepath, cubename)
    data = fits.getdata(file)[0]
    # fit
    X, Y = np.indices(data.shape)
    mask = ~np.isnan(data)
    x, y = X[mask], Y[mask]
    dat = data[mask]
    errorFunc = lambda p, xx, yy, d: gauss2d(p, xx, yy) - d
    popt, pcov, _, _, success = leastsq(errorFunc, **kw_fit, args=(x, y, dat), full_output=True)
    errs = np.sqrt(np.diag(pcov))
    theta, sx, sy = popt[4], popt[2], popt[3]
    PA, ePA = (90 + theta) % 180, errs[4]
    ratio, eratio = sx/sy, sx/sy * np.sqrt((errs[2]/sx)**2+(errs[3]/sy)**2)
    if simulate:
        crnr, Ncycles = kw_sim['corner'], kw_sim['Ncycles']
        seed = kw_sim['seed'] if 'seed' in kw_sim.keys() else 1
        np.random.seed(seed)
        PA_arr, ratio_arr = [], []
        if crnr:
            sX_arr, sY_arr = [], []
        noise_threshold = np.nanmax(errorFunc(popt, x, y, dat))
        for cycle in range(Ncycles):
            data_noise = dat + noise_threshold * np.random.normal(size=dat.shape)
            data_noise[dat == 0] = 0
            popt_noise, pcov_noise, _, _, success = leastsq(errorFunc, x0=popt, args=(x, y, data_noise), full_output=True)
            theta_noise, sx_noise, sy_noise = popt_noise[4], popt_noise[2], popt_noise[3]
            PA_noise = (90 + theta_noise) % 180
            PA_arr.append(PA_noise)
            ratio_noise = sx_noise / sy_noise
            ratio_arr.append(ratio_noise)
            if crnr:
                sX_arr.append(sx_noise)
                sY_arr.append(sy_noise)
        if crnr:
            from corner import corner
            corner(np.array([PA_arr, sX_arr, sY_arr, ratio_arr]).T,
                   labels=['$\\theta_{\\rm PA}$ (deg)', '$b$ (pix)', '$a$ (pix)', '$b/a$'],
                   quantiles=[0.16, 0.50, 0.84], show_titles=True, title_kwargs={'fontsize': 10},
                   truths=[PA, sx, sy, ratio])
            if save:
                savepath, savename = kw_sim['savepath'], kw_sim['savename']
                plt.savefig(path.join(savepath, savename), bbox_inches='tight')
                plt.close()
    # plot
    fig = aplpy.FITSFigure(file)
    fig.recenter(*args_recenter)  # order: ra, dec, width, height
    fig.add_grid()
    fig.grid.set_color('gray')
    fig.show_colorscale(aspect='equal', cmap=con.seq, vmin=1, vmax=np.max(levels), stretch='sqrt')
    fig.show_contour(levels=levels, colors='tab:red', linewidths=1.5, zorder=3)
    # add representative ellipse for PA and axial ratio
    head = fits.getheader(file)
    fwcs = WCS(head).wcs
    pix2deg = np.abs(fwcs.cdelt[0])
    xw, yw = fig.pixel2world(popt[0]+0.5, popt[1]+1.5)
    fig.show_ellipses(xw=[xw], yw=[yw], height=4*pix2deg*sy, width=4*pix2deg*sx, angle=PA,#theta + 90,#-(90-theta),
                      edgecolor='yellow', facecolor='none', linewidth=3, zorder=4)
    fig.add_label(0.97, 0.96,
                  'fit: ($\\theta_{\\rm PA}$, $b/a$) = (%.1f$^\\circ \\pm %.1f^\\circ$, %.2f $\\pm$ %.1g)' % (PA, ePA, ratio, eratio),
                  ha='right', va='center', relative=True)
    if simulate:
        vPA = np.percentile(PA_arr, [16, 50, 84])
        PAest, ePAest = vPA[1], np.mean([vPA[1]-vPA[0], vPA[2]-vPA[1]])
        vratio = np.percentile(ratio_arr, [16, 50, 84])
        ratioest, eratioest = vratio[1], np.mean([vratio[1]-vratio[0], vratio[2]-vratio[1]])
        fig.add_label(0.97, 0.91,
                      'sim: ($\\theta_{\\rm PA}$, $b/a$) = (%.1f$^\\circ \\pm %.1f^\\circ$, %.2f $\\pm$ %.1g)' % (PAest, ePAest, ratioest, eratioest),
                      ha='right', va='center', relative=True)
    fig.add_label(0.03, 0.96, cubelabel, ha='left', va='center', relative=True)
    # save?
    if save:
        savepath, savename = kw_save['savepath'], kw_save['savename']
        plt.savefig(path.join(savepath, savename), bbox_inches='tight')
        plt.close()
    if results:
        return popt, errs

# # # MAIN
# # setup
for i, c in enumerate(con.names):
    levels = np.linspace(1, 1000, 5)
    ar_rctr = [con.ra, con.dec, con.height, con.width]
    shape = [300, 400, 300, 300]
    kw_fit = {'x0': [np.floor(shape[i]/2), np.floor(shape[i]/2), 5, 4, 23, 700]}
    kw_save = {'savepath': con.fullpath + 'Tests/', 'savename': f'PA_{c}_positionAngle.pdf'}
    kw_sim = {'corner': True, 'Ncycles': 5000, 'savepath': con.fullpath + 'Tests/', 'savename': f'PA_{c}_cornerPlot.pdf'}
    plotPA(cubename=f'PA_{c}_mom0_custom.fits', cubepath=con.fullpath + 'Tests/', cubelabel=c,
           levels=levels, args_recenter=ar_rctr, save=True, results=False, simulate=True, kw_fit=kw_fit,
           kw_save=kw_save, kw_sim=kw_sim)