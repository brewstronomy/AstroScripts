#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: luca (lucabeale@gmail.com)
"""

import numpy as np, matplotlib.pyplot as plt, matplotlib.cm as cm, matplotlib.colors as cols
import PiscesA.constant as con
import astropy.units as u
from os import path
from astropy.coordinates import SkyCoord, Distance, Galactocentric
from astropy.table import Table
from mpl_toolkits import mplot3d
from matplotlib.animation import FuncAnimation as anim
import mpl_toolkits.mplot3d.art3d as art3d

# # # preliminaries
fullpath = con.fullpath + 'Local_Void/'
filepath = fullpath + 'Data/'
filename = 'lvol.ecsv'

npix = 1e4
cmap = cm.Spectral_r
csca = cm.ScalarMappable(cmap=cmap, norm=cols.Normalize(vmin=-400, vmax=400))
cfun = lambda x: cols.to_hex(csca.to_rgba(x))

origin = [0], [0]
qwarg = {'angles': 'xy', 'scale_units': 'xy', 'scale': 1, 'width': 0.005}
ckwarg = {'cmap': cmap, 'vmin': -400, 'vmax': 400}

# # # import data
# # nearby galaxies
data = Table.read(path.join(filepath, filename))
names_PGC, names_NSA, names_Other = data['PGC#'], data['NSAID'], data['othername']
alpha, delta, dist, vrad = data['RA'], data['Dec'], data['distance'], data['vhelio']

# # closest galaxy to Pisces A
# NOTE: from Simbad search for galaxies < 10 Mpc within ~10 deg of Pisces A
name_closeGal = 'AGC748778'
coord_closeGal = SkyCoord('00h06m34.3s +15d30m39s', unit=(u.hourangle, u.deg),
                          distance=Distance(value=6.345, unit=u.Mpc),
                          radial_velocity=258*u.km/u.s)  # from Simbad search

# # Pisces A
# main
vrad_PA, dist_PA = 236 * u.km/u.s, Distance(5.64, u.Mpc)
coord_PA = SkyCoord('00h14m46s', '+10d48m47.01s', distance=dist_PA, radial_velocity=vrad_PA)
# NE extension
coord_NE = SkyCoord('00h14m51s', '+10d50m48s', distance=dist_PA)

# # Local Void
dist_LV = Distance(5, u.Mpc)  # avg of galaxies in Rizzi+2017
coord_LV = SkyCoord('18h38m', '18d', distance=dist_LV)

# # # data massaging
# # distance & completeness cuts (credit: E. Tollerud)
K_absmag = data['K'] - (5*np.log10(dist*1e6)-5)
limit_2MASS = 13.5*u.mag-Distance(2**0.5*10*u.Mpc).distmod  # 2MASS absolute magnitude limit

mag_mask = K_absmag < limit_2MASS.value
dist_mask = (dist > 0) & (dist < 15)  # USELESS; completeness cut limits to 10 Mpc by itself (see Tollerud+2016)
mask = dist_mask & mag_mask

names_PGC, names_NSA, names_Other = names_PGC[mask], names_NSA[mask], names_Other[mask]
alpha, delta, dist, vrad, Kabs = alpha[mask], delta[mask], dist[mask], vrad[mask], K_absmag[mask]
names_PGC, names_NSA, names_Other = names_PGC[~vrad.mask], names_NSA[~vrad.mask], names_Other[~vrad.mask]
alpha, delta, dist, vrad, Kabs = alpha[~vrad.mask], delta[~vrad.mask], dist[~vrad.mask], vrad[~vrad.mask], Kabs[~vrad.mask]
coords = SkyCoord(alpha*u.deg, delta*u.deg, distance=Distance(dist, u.Mpc), radial_velocity=vrad*u.km/u.s)

# # coordinate systems
# Cartesian Galactic
coords_galacti_unitCart = coords.galactic.cartesian.without_differentials() / coords.galactic.cartesian.norm()
coord_PA_galacti_unitCart = coord_PA.galactic.cartesian.without_differentials() / coord_PA.galactic.cartesian.norm()
coord_NE_galacti_unitCart = coord_NE.galactic.cartesian.without_differentials() / coord_NE.galactic.cartesian.norm()
coord_LV_galacti_unitCart = coord_LV.galactic.cartesian.without_differentials() / coord_LV.galactic.cartesian.norm()
coord_closeGal_galacti_unitCart = coord_closeGal.galactic.cartesian.without_differentials() / coord_closeGal.galactic.cartesian.norm()
# Cartesian Supergalactic
coords_supergal_cartesian = coords.supergalactic.cartesian
coords_supergal_unitCart = coords_supergal_cartesian.without_differentials() / coords_supergal_cartesian.norm()
X_supergal, Y_supergal, Z_supergal = coords_supergal_cartesian.xyz
coord_PA_supergal_cartesian = coord_PA.supergalactic.cartesian
coord_PA_supergal_unitCart = coord_PA_supergal_cartesian.without_differentials() / coord_PA_supergal_cartesian.norm()
X_PA_supergal, Y_PA_supergal, Z_PA_supergal = coord_PA_supergal_cartesian.xyz
coord_NE_supergal_cartesian = coord_NE.supergalactic.cartesian
coord_NE_supergal_unitCart = coord_NE_supergal_cartesian.without_differentials() / coord_NE_supergal_cartesian.norm()
X_NE_supergal, Y_NE_supergal, Z_NE_supergal = coord_NE_supergal_cartesian.xyz
coord_LV_supergal_cartesian = coord_LV.supergalactic.cartesian
coord_LV_supergal_unitCart = coord_LV_supergal_cartesian.without_differentials() / coord_LV_supergal_cartesian.norm()
X_LV_supergal, Y_LV_supergal, Z_LV_supergal = coord_LV_supergal_cartesian.xyz
coord_closeGal_supergal_cartesian = coord_closeGal.supergalactic.cartesian
coord_closeGal_supergal_unitCart = coord_closeGal_supergal_cartesian.without_differentials() / coord_closeGal_supergal_cartesian.norm()
X_closeGal_supergal, Y_closeGal_supergal, Z_closeGal_supergal = coord_closeGal_supergal_cartesian.xyz

# # velocity transformations
# wrt Galactic Standard of Rest
v_sun_galacto = Galactocentric.galcen_v_sun.to_cartesian()

vrad_GSR = coords.radial_velocity + v_sun_galacto.dot(coords_galacti_unitCart)
vrad_PA_GSR = coord_PA.radial_velocity + v_sun_galacto.dot(coord_PA_galacti_unitCart)
vrad_closeGal_GSR = coord_closeGal.radial_velocity + v_sun_galacto.dot(coord_closeGal_galacti_unitCart)

# wrt Local Sheet (from Tully+2008)
func_VLS_supergal = lambda v, x: (v + np.array([np.dot(c.xyz.value, [234, -31, 214]) for c in np.array(x)]))*u.km/u.s  # for unit vectors in supergalactic frame (should be same as func_VLSkj)
func_VLS = lambda v, x: (v + np.array([np.dot(c.xyz.value, [-26, 317, -8]) for c in np.array(x)]))*u.km/u.s  # for unit vectors in galactic frame; v is heliocentric
func_VLV = lambda v, x: (func_VLS(v, x).value + np.array([np.dot(c.xyz.value, [-222, -130, -10]) for c in np.array(x)]))*u.km/u.s  # v is heliocentric; for galactic frame

v_LS = func_VLS(coords.radial_velocity.value, coords_galacti_unitCart)  # array
v_PA_LS = func_VLS(coord_PA.radial_velocity.value, [coord_PA_galacti_unitCart])[0]  # value
v_closeGal_LS = func_VLS(coord_closeGal.radial_velocity.value, [coord_closeGal_galacti_unitCart])[0]  # value

# peculiar (v_GSR - Hubble flow)
H0 = 74.03*u.km/u.s/u.Mpc  # km/s/Mpc; SHoES collab (Riess+2019)
vpec_LS = v_LS - H0 * coords.distance
vpec_PA_LS = v_PA_LS - H0 * coord_PA.distance
vpec_closeGal_LS = v_closeGal_LS - H0 * coord_closeGal.distance

# wrt Pisces A
vrad_GSR_vector = (vrad_GSR * coords_galacti_unitCart)  # full velocity vector for each galaxy
vrad_PA_GSR_vector = (vrad_PA_GSR * coord_PA_galacti_unitCart)  # full velocity vector of Pisce sA
vrad_closeGal_GSR_vector = (vrad_closeGal_GSR * coord_closeGal_galacti_unitCart)  # full velocity   vector for closest gal

pos_rel_vector = coords.galactic.cartesian.without_differentials() - coord_PA.galactic.cartesian.without_differentials()
pos_rel_unitVec = pos_rel_vector / pos_rel_vector.norm()
vrel = np.array([np.dot(vV - vrad_PA_GSR_vector.get_xyz(), uV) for vV, uV in zip(vrad_GSR_vector.get_xyz().T,
                                                                                 pos_rel_unitVec.get_xyz().T)])*u.km/u.s
pos_closeGal_rel_vector = coord_closeGal.galactic.cartesian.without_differentials() - coord_PA.galactic.cartesian.without_differentials()
pos_closeGal_rel_unitVec = pos_closeGal_rel_vector / pos_closeGal_rel_vector.norm()
vrel_closeGal = np.dot(vrad_closeGal_GSR_vector.get_xyz(), pos_closeGal_rel_unitVec.get_xyz())*u.km/u.s

# # distance transformations
# 3D distance from Pisces A
seps = coord_PA.separation_3d(coords)
seps_nodim = seps.value

# # statistical functions
# general statistics
def quantity_statistic(stat, quant_arr, cond_arr, iter_arr, di=None, percentile_error=False, eper=None):
    if di is not None:
        condlo = lambda cond_elem, it_elem: cond_elem > it_elem
        condhi = lambda cond_elem, it_elem, step: cond_elem <= it_elem + step
    else:
        condlo = lambda x, y: True
        condhi = lambda cond_elem, it_elem, z: cond_elem <= it_elem
    if stat is np.percentile:
        med = [stat([quant_arr[i] for i, cd in enumerate(cond_arr) if (condlo(cd, d) and condhi(cd, d, di))], 50) for d in iter_arr]
        if percentile_error:
            err = [stat([quant_arr[i] for i, cd in enumerate(cond_arr) if (condlo(cd, d) and condhi(cd, d, di))], eper) for d in iter_arr]
            return np.abs(np.array(med) - np.array(err))
        else:
            return np.array(med)
    else:
        lst = [stat([quant_arr[i] for i, cd in enumerate(cond_arr) if (condlo(cd, d) and condhi(cd, d, di))]) for d in iter_arr]
        return np.array(lst)

# # # plotting
# # radial velocity relative to Pisces A in spherical shells
# setup
radii, dr = np.arange(np.floor(seps_nodim.min()), np.ceil(seps_nodim.max())), 1
vrel_med = quantity_statistic(np.median, vrel.value, seps_nodim, radii, dr)
vrel_elo = quantity_statistic(np.percentile, vrel.value, seps_nodim, radii, dr, True, 16)
vrel_ehi = quantity_statistic(np.percentile, vrel.value, seps_nodim, radii, dr, True, 84)
vrel_avg = quantity_statistic(np.mean, vrel.value, seps_nodim, radii, dr)
vrel_len = quantity_statistic(len, vrel.value, seps_nodim, radii, dr)
# plot
fig, ax = plt.subplots()
ax.scatter(seps_nodim, vrel.value, alpha=0.2, edgecolor='gray', label='', zorder=0)
ax.plot(radii+dr/2, vrel_avg, linewidth=2, linestyle='dashed', color='tab:red', label='average', zorder=0)
ax.errorbar(radii+dr/2, vrel_med, yerr=[vrel_elo, vrel_ehi], marker='', color='tab:blue',
            capsize=2.5, label='inner 68\%%', zorder=1)
ax.scatter(radii+dr/2, vrel_med, s=50*vrel_len**0.8, edgecolor='black', linewidths=1, color='tab:blue', label='median', zorder=2)
ax.scatter(coord_closeGal.separation_3d(coord_PA).value, vrel_closeGal.value, s=200, edgecolor='black', linewidths=1,
           color='tab:green', label=name_closeGal, zorder=3)
ax.plot(np.array(sorted(seps_nodim)), H0.value*np.array(sorted(seps_nodim)), linewidth=1.5, linestyle='dashdot',
        color='black', zorder=3, label='H$_0$ = %.2f km/s/Mpc' % (H0.value))
ax.plot(np.array(sorted(seps_nodim)), (H0.value-10)*np.array(sorted(seps_nodim)), linewidth=1, linestyle='dotted',
        color='black', zorder=3, label='H$_0$ $\pm$ 10 km/s/Mpc')
ax.plot(np.array(sorted(seps_nodim)), (H0.value+10)*np.array(sorted(seps_nodim)), linewidth=1, linestyle='dotted',
        color='black', zorder=3, label='')
ax.annotate('Galaxies Per Shell:', (radii[0]+dr/2, 150), ha='left', va='center', fontsize='small')
for i, ngal in enumerate(vrel_len):
    ax.annotate(ngal, (radii[i]+dr/2, 120), ha='center', va='center', fontsize='small')
ax.grid()
ax.set_axisbelow(True)
ax.legend(loc='upper left', ncol=2, framealpha=1, fontsize='small')
ax.set_xlabel('Distance from Pisces A (Mpc)')
ax.set_ylabel('Radial Relative Velocity ($v_{\\rm rel, GSR}$/km s$^{-1}$)')
ax.set_yscale('log')
ax.set_ylim(100, 4200)
ax.set_yticks([100, 200, 400, 600, 800, 1000, 2000, 4000])
ax.set_yticklabels(ax.get_yticks())

plt.savefig(path.join(con.fullpath+'Tests/', 'PA_vrelative_shells.pdf'))
plt.close()
# # peculiar velocity relative to Pisces A in spherical shells
# plot
fig, ax = plt.subplots()
ax.scatter(seps_nodim, (vrel - H0*seps).value, alpha=0.2, edgecolor='gray', label='', zorder=0)
ax.plot(radii+dr/2, vrel_avg - H0.value*(radii+dr/2),
        linewidth=2, linestyle='dashed', color='tab:red', label='average', zorder=0)
ax.errorbar(radii+dr/2, vrel_med - H0.value*(radii+dr/2), yerr=[vrel_elo, vrel_ehi], marker='', color='tab:blue',
            capsize=2.5, label='inner 68\%%', zorder=1)
ax.scatter(radii+dr/2, vrel_med - H0.value*(radii+dr/2), s=50*vrel_len**0.8,
           edgecolor='black', linewidths=1, color='tab:blue', label='median', zorder=2)
ax.scatter(coord_closeGal.separation_3d(coord_PA).value,
           (vrel_closeGal - H0*coord_closeGal.separation_3d(coord_PA)).value, s=200, edgecolor='black', linewidths=1,
           color='tab:green', label=name_closeGal, zorder=3)
ax.annotate('Galaxies Per Shell:', (radii[0]+dr/2, -400), ha='left', va='center', fontsize='small')
for i, ngal in enumerate(vrel_len):
    ax.annotate(ngal, (radii[i]+dr/2, -460), ha='center', va='center', fontsize='small')
ax.grid()
ax.set_axisbelow(True)
ax.legend(loc='upper left', framealpha=1, fontsize='small')
ax.set_xlabel('Distance from Pisces A (Mpc)')
xlim = ax.get_xlim()
ax.plot([1.7, 21.4], [0, 0], color='black', linestyle='dashed', linewidth=1.5, label='', zorder=0)
ax.set_xlim(*xlim)
ax.set_ylabel('Peculiar Relative Velocity ($v_{\\rm rel, GSR} - H_0 \\times d$)')
ax.set_ylim(-500, 600)
ax.set_yticks([-500, -300, -100, 0, 100, 300, 500])
ax.set_yticklabels(['$%i$' % t for t in ax.get_yticks()])

plt.savefig(path.join(con.fullpath+'Tests/', 'PA_vrelative_shells_peculiar.pdf'))
plt.close()

# # radial velocity relative to Pisces A within a spherical volume (steps of 1 Mpc)
# setup
radii = np.arange(3, np.ceil(seps_nodim.max()))
vrel_med = quantity_statistic(np.median, vrel.value, seps_nodim, radii, None)
vrel_elo = quantity_statistic(np.percentile, vrel.value, seps_nodim, radii, None, True, 16)
vrel_ehi = quantity_statistic(np.percentile, vrel.value, seps_nodim, radii, None, True, 84)
vrel_avg = quantity_statistic(np.mean, vrel.value, seps_nodim, radii, None)
vrel_len = quantity_statistic(len, vrel.value, seps_nodim, radii, None)
# plot
fig, ax = plt.subplots()
ax.scatter(seps_nodim, vrel.value, alpha=0.2, edgecolor='gray', label='', zorder=0)
ax.plot(radii, vrel_avg, linewidth=2, linestyle='dashed', color='tab:red', label='average', zorder=0)
ax.errorbar(radii, vrel_med, yerr=[vrel_elo, vrel_ehi], marker='', color='tab:blue',
            capsize=2.5, label='inner 68\%%', zorder=1)
ax.scatter(radii, vrel_med, s=40*vrel_len**0.5, edgecolor='black', linewidths=1, color='tab:blue', label='median', zorder=2)
ax.scatter(coord_closeGal.separation_3d(coord_PA).value, vrel_closeGal.value, s=200, edgecolor='black', linewidths=1,
           color='tab:green', label=name_closeGal, zorder=3)
ax.plot(np.array(sorted(seps_nodim)), H0.value*np.array(sorted(seps_nodim)), linewidth=1.5, linestyle='dashdot',
        color='black', zorder=3, label='H$_0$ = %.2f km/s/Mpc' % (H0.value))
ax.plot(np.array(sorted(seps_nodim)), (H0.value-10)*np.array(sorted(seps_nodim)), linewidth=1, linestyle='dotted',
        color='black', zorder=3, label='H$_0$ $\pm$ 10 km/s/Mpc')
ax.plot(np.array(sorted(seps_nodim)), (H0.value+10)*np.array(sorted(seps_nodim)), linewidth=1, linestyle='dotted',
        color='black', zorder=3, label='')
ax.annotate('Galaxies Per Sphere:', (radii[0], 160), ha='left', va='center', fontsize='small')
for i, ngal in enumerate(vrel_len):
    ax.annotate(ngal, (radii[i], 120), ha='center', va='center', fontsize='small')
ax.grid()
ax.set_axisbelow(True)
ax.legend(loc='upper left', ncol=2, framealpha=1, fontsize='small')
ax.set_xlabel('Distance from Pisces A (Mpc)')
ax.set_ylabel('Radial Relative Velocity ($v_{\\rm rel, GSR}$/km s$^{-1}$)')
ax.set_yscale('log')
ax.set_ylim(100, 4200)
ax.set_yticks([100, 200, 400, 600, 800, 1000, 2000, 4000])
ax.set_yticklabels(ax.get_yticks())

plt.savefig(path.join(con.fullpath+'Tests/', 'PA_vrelative_spheres.pdf'))
plt.close()
# # peculiar velocity relative to Pisces A within a spherical volume (steps of 1 Mpc)
# plot!
fig, ax = plt.subplots()
ax.scatter(seps_nodim, (vrel - H0*seps).value, alpha=0.2, edgecolor='gray', label='', zorder=0)
ax.plot(radii+dr/2, vrel_avg - H0.value*(radii+dr/2),
        linewidth=2, linestyle='dashed', color='tab:red', label='average', zorder=0)
ax.errorbar(radii+dr/2, vrel_med - H0.value*(radii+dr/2), yerr=[vrel_elo, vrel_ehi], marker='', color='tab:blue',
            capsize=2.5, label='inner 68\%%', zorder=1)
ax.scatter(radii+dr/2, vrel_med - H0.value*(radii+dr/2), s=40*vrel_len**0.5,
           edgecolor='black', linewidths=1, color='tab:blue', label='median', zorder=2)
ax.scatter(coord_closeGal.separation_3d(coord_PA).value,
           (vrel_closeGal - H0*coord_closeGal.separation_3d(coord_PA)).value, s=200, edgecolor='black', linewidths=1,
           color='tab:green', label=name_closeGal, zorder=3)
ax.annotate('Galaxies Per Sphere:', (radii[0], -700), ha='left', va='center', fontsize='small')
for i, ngal in enumerate(vrel_len):
    ax.annotate(ngal, (radii[i], -750), ha='center', va='center', fontsize='small')
ax.grid()
ax.set_axisbelow(True)
ax.legend(loc='upper left', framealpha=1, fontsize='small')
ax.set_xlabel('Distance from Pisces A (Mpc)')
xlim = ax.get_xlim()
ax.plot([1.5, 21.4], [0, 0], color='black', linestyle='dashed', linewidth=1.5, label='', zorder=0)
ax.set_xlim(*xlim)
ax.set_ylabel('Peculiar Relative Velocity ($v_{\\rm rel, GSR} - H_0 \\times d$)')
ax.set_ylim(-800, 800)

plt.savefig(path.join(con.fullpath+'Tests/', 'PA_vrelative_spheres_peculiar.pdf'))
plt.close()

# # radial velocity in spherical shells
# setup
radii, dr = np.arange(0.5, np.ceil(dist.max())), 1
vrad_med = quantity_statistic(np.median, vrad_GSR.value, dist, radii, dr)
vrad_elo = quantity_statistic(np.percentile, vrad_GSR.value, dist, radii, dr, True, 16)
vrad_ehi = quantity_statistic(np.percentile, vrad_GSR.value, dist, radii, dr, True, 84)
vrad_avg = quantity_statistic(np.mean, vrad_GSR.value, dist, radii, dr)
vrad_len = quantity_statistic(len, vrad_GSR.value, dist, radii, dr)
# plot
fig, ax = plt.subplots()
ax.scatter(dist, vrad_GSR.value, alpha=0.2, edgecolor='gray', label='', zorder=0)
ax.plot(radii+dr/2, vrad_avg, linewidth=2, linestyle='dashed', color='tab:red', label='average', zorder=0)
ax.errorbar(radii+dr/2, vrad_med, yerr=[vrad_elo, vrad_ehi], marker='', color='tab:blue',
            capsize=2.5, label='inner 68\%%', zorder=1)
ax.scatter(radii+dr/2, vrad_med, s=40*vrad_len**0.8, edgecolor='black', linewidths=1, color='tab:blue', label='median', zorder=2)
ax.scatter(coord_closeGal.distance.value, vrad_closeGal_GSR.value, s=200, edgecolor='black', linewidths=1,
           color='tab:green', label=name_closeGal, zorder=4)
ax.scatter(coord_PA.distance.value, vrad_PA_GSR.value, s=400, marker='*', edgecolor='black', linewidths=1,
           color='tab:orange', label='Pisces A', zorder=4)
ax.plot(dist, H0.value*dist, linewidth=1.5, linestyle='dashdot',
        color='black', zorder=3, label='H$_0$ = %.2f km/s/Mpc' % (H0.value))
ax.plot(dist, (H0.value-10)*dist, linewidth=1, linestyle='dotted',
        color='black', zorder=3, label='H$_0$ $\pm$ 10 km/s/Mpc')
ax.plot(dist, (H0.value+10)*dist, linewidth=1, linestyle='dotted',
        color='black', zorder=3, label='')
ax.annotate('Galaxies Per Shell:', (radii[0]+dr/2, -250), ha='left', va='center', fontsize='small')
for i, ngal in enumerate(vrad_len):
    ax.annotate(ngal, (radii[i]+dr/2, -350), ha='center', va='center', fontsize='small')
ax.grid()
ax.set_axisbelow(True)
ax.legend(loc='upper left', ncol=2, framealpha=1, fontsize='small')
ax.set_xlabel('Distance (Mpc)')
ax.set_ylabel('Radial Velocity ($v_{\\rm GSR}$/km s$^{-1}$)')
ax.set_ylim(-500, 2000)

plt.savefig(path.join(con.fullpath+'Tests/', 'PA_vrad_shells.pdf'))
plt.close()
# # peculiar radial velocity in spherical shells
# plot
fig, ax = plt.subplots()
ax.scatter(dist, vpec_LS.value, alpha=0.2, edgecolor='gray', label='', zorder=0)
ax.plot(radii+dr/2, vrad_avg - H0.value*(radii+dr/2), linewidth=2, linestyle='dashed', color='tab:red', label='average', zorder=0)
ax.errorbar(radii+dr/2, vrad_med - H0.value*(radii+dr/2), yerr=[vrad_elo, vrad_ehi], marker='', color='tab:blue',
            capsize=2.5, label='inner 68\%%', zorder=1)
ax.scatter(radii+dr/2, vrad_med - H0.value*(radii+dr/2), s=40*vrel_len**0.6, edgecolor='black', linewidths=1, color='tab:blue', label='median', zorder=2)
ax.scatter(coord_closeGal.distance.value, vpec_closeGal_LS.value, s=200, edgecolor='black', linewidths=1,
           color='tab:green', label=name_closeGal, zorder=4)
ax.scatter(coord_PA.distance.value, vpec_PA_LS.value, s=400, marker='*', edgecolor='black', linewidths=1,
           color='tab:orange', label='Pisces A', zorder=4)
ax.annotate('Galaxies Per Shell:', (radii[0]+dr/2, -380), ha='left', va='center', fontsize='small')
for i, ngal in enumerate(vrad_len):
    ax.annotate(ngal, (radii[i]+dr/2, -450), ha='center', va='center', fontsize='small')
ax.grid()
ax.set_axisbelow(True)
ax.legend(loc='upper left', ncol=2, framealpha=1, fontsize='small')
ax.set_xlabel('Distance (Mpc)')
xlim = ax.get_xlim()
ax.plot([1.5, 21.4], [0, 0], color='black', linestyle='dashed', linewidth=1.5, label='', zorder=0)
ax.set_xlim(*xlim)
ax.set_ylabel('Peculiar Radial Velocity ($v_{\\rm GSR} - H_0 \\times d$)')
ax.set_ylim(-600, 800)

plt.savefig(path.join(con.fullpath+'Tests/', 'PA_vrad_shells_peculiar.pdf'))
plt.close()

# # velocity with respect to the Local Sheet in spherical shells
# setup
radii, dr = np.arange(0.5, np.ceil(dist.max())), 1
vLS_med = quantity_statistic(np.median, v_LS.value, dist, radii, dr)
vLS_elo = quantity_statistic(np.percentile, v_LS.value, dist, radii, dr, True, 16)
vLS_ehi = quantity_statistic(np.percentile, v_LS.value, dist, radii, dr, True, 84)
vLS_avg = quantity_statistic(np.mean, v_LS.value, dist, radii, dr)
vLS_len = quantity_statistic(len, v_LS.value, dist, radii, dr)
# plot
fig, ax = plt.subplots()
ax.scatter(dist, v_LS.value, alpha=0.2, edgecolor='gray', label='', zorder=0)
ax.plot(radii+dr/2, vLS_avg, linewidth=2, linestyle='dashed', color='tab:red', label='average', zorder=0)
ax.errorbar(radii+dr/2, vLS_med, yerr=[vLS_elo, vLS_ehi], marker='', color='tab:blue',
            capsize=2.5, label='inner 68\%%', zorder=1)
ax.scatter(radii+dr/2, vLS_med, s=40*vrad_len**0.8, edgecolor='black', linewidths=1, color='tab:blue', label='median', zorder=2)
ax.scatter(coord_closeGal.distance.value, v_closeGal_LS.value, s=200, edgecolor='black', linewidths=1,
           color='tab:green', label=name_closeGal, zorder=4)
ax.scatter(coord_PA.distance.value, v_PA_LS.value, s=400, marker='*', edgecolor='black', linewidths=1,
           color='tab:orange', label='Pisces A', zorder=4)
ax.annotate('Galaxies Per Shell:', (radii[0]+dr/2, 10), ha='left', va='center', fontsize='small')
for i, ngal in enumerate(vLS_len):
    ax.annotate(ngal, (radii[i]+dr/2, -20), ha='center', va='center', fontsize='small')
ax.grid()
ax.set_axisbelow(True)
ax.legend(loc='upper left', ncol=2, framealpha=1, fontsize='small')
ax.set_xlabel('Distance (Mpc)')
ax.set_ylabel('Radial Velocity ($v_{\\rm LS}$/km s$^{-1}$)')
ax.set_ylim(-50, 700)

plt.savefig(path.join(con.fullpath+'Tests/', 'PA_vLS_shells.pdf'))
plt.close()

# # peculiar velocity with respect to the Local Sheet in spherical shells
# plot
fig, ax = plt.subplots()
ax.scatter(dist, v_LS.value - H0.value*dist, alpha=0.2, edgecolor='gray', label='', zorder=0)
ax.plot(radii+dr/2, vLS_avg - H0.value*(radii+dr/2), linewidth=2, linestyle='dashed', color='tab:red', label='average', zorder=0)
ax.errorbar(radii+dr/2, vLS_med - H0.value*(radii+dr/2), yerr=[vLS_elo, vLS_ehi], marker='', color='tab:blue',
            capsize=2.5, label='inner 68\%%', zorder=1)
ax.scatter(radii+dr/2, vLS_med - H0.value*(radii+dr/2), s=40*vrad_len**0.8, edgecolor='black', linewidths=1, color='tab:blue', label='median', zorder=2)
ax.scatter(coord_closeGal.distance.value, (v_closeGal_LS - H0*coord_closeGal.distance).value, s=200, edgecolor='black', linewidths=1,
           color='tab:green', label=name_closeGal, zorder=4)
ax.scatter(coord_PA.distance.value, (v_PA_LS - H0*coord_PA.distance).value, s=400, marker='*', edgecolor='black', linewidths=1,
           color='tab:orange', label='Pisces A', zorder=4)
ax.annotate('Galaxies Per Shell:', (radii[0]+dr/2, -1000), ha='left', va='center', fontsize='small')
for i, ngal in enumerate(vLS_len):
    ax.annotate(ngal, (radii[i]+dr/2, -1100), ha='center', va='center', fontsize='small')
ax.grid()
ax.set_axisbelow(True)
ax.legend(loc='upper left', ncol=2, framealpha=1, fontsize='small')
ax.set_xlabel('Distance (Mpc)')
ax.set_ylabel('Peculiar Radial Velocity ($v_{\\rm LS} - H_0 \\times d$)')
ax.set_ylim(-500, 500)

plt.savefig(path.join(con.fullpath+'Tests/', 'PA_vLS_peculiar_shells.pdf'))
plt.close()

# # on sky distribution (Supergalactic, GSR peculiar velocities)
H0 = 74.03  # km/s/Mpc; SHoES collab (Riess+2019)
akwargs = lambda x: {'edgecolor': 'black', 'facecolor': cfun(vpec_PA_LS.value), 'linewidth': 1,
                     'units': 'xy', 'angles': 'xy', 'scale_units': 'xy', 'scale': 1/x}

# Local Sheet definitions (Tully+08)
circ = plt.Circle((0, 0), 7, facecolor='none', edgecolor='black', linewidth=2)
rec1 = plt.Rectangle((-1.5, -7), 3, 14, facecolor='none', edgecolor='black', linewidth=2, label='Local Sheet')
rec2 = plt.Rectangle((-7, -1.5), 14, 3, facecolor='none', edgecolor='black', linewidth=2)

fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots(2, 2, figsize=(10, 10))
fig.subplots_adjust(wspace=0, hspace=0)
# Y vs Z
gals = ax11.scatter(Z_supergal, Y_supergal, c=vpec_LS.value, s=100, edgecolor='black', linewidths=0.5, label='nearby galaxies', zorder=1, **ckwarg)
pa = ax11.scatter(Z_PA_supergal, Y_PA_supergal, c=cfun(vpec_PA_LS.value),
                  s=300, marker='*', edgecolor='black', label='Pisces A', zorder=3, **ckwarg)
ax11.quiver(*origin, Z_LV_supergal, Y_LV_supergal, label='Local Void', zorder=2, **qwarg)
ax11.quiver(Z_PA_supergal, Y_PA_supergal, Z_NE_supergal-Z_PA_supergal, Y_NE_supergal-Y_PA_supergal, label='NEe', zorder=2, **akwargs(2000))
ls = ax11.add_artist(rec1)

ax11.grid()
ax11.set_axisbelow(True)
ax11.tick_params(labelbottom=True)
ax11.set_xlabel('$Z_{\\rm SG}$ (Mpc)')
ax11.set_ylabel('$Y_{\\rm SG}$ (Mpc)')

# Y vs X
im = ax12.scatter(X_supergal, Y_supergal, c=vpec_LS.value, s=100, edgecolor='black', linewidths=0.5, label='nearby galaxies', zorder=1, **ckwarg)
ax12.scatter(X_PA_supergal, Y_PA_supergal, c=cfun(vpec_PA_LS.value), s=300, marker='*', edgecolor='black',
             label='Pisces A', zorder=3, **ckwarg)
#cg = ax12.scatter(X_closeGal_supergal, Y_closeGal_supergal, c=cfun(vpec_closeGal_GSR.value),
#                  s=100, marker='s', edgecolor='black', label=name_closeGal, zorder=3, **ckwarg)
ax12.quiver(*origin, X_LV_supergal, Y_LV_supergal, label='Local Void', zorder=2, **qwarg)
ax12.quiver(X_PA_supergal, Y_PA_supergal, X_NE_supergal-X_PA_supergal, Y_NE_supergal-Y_PA_supergal, label='NEe', zorder=2, **akwargs(1000))
ax12.add_artist(circ)

ax12.grid()
ax12.set_axisbelow(True)
ax12.set_xticklabels([])
ax12.set_yticklabels([])

# colorbar axis
cbar = fig.colorbar(im, ax=ax21, pad=0.01, aspect=30, orientation='horizontal', shrink=0.8)
cbar.ax.set_xlabel(r'Peculiar Velocity $v_{\rm LS} - H_0 d$ (km s$^{-1}$)')
ax21.legend([gals, pa, ls], [l.get_label() for l in [gals, pa, ls]], loc='center')

ax21.axis('off')

# X vs Z
#labax = ax22.twinx()
ax22.scatter(X_supergal, Z_supergal, c='none', zorder=0)
ax22.scatter(X_supergal, Z_supergal, c=vpec_LS.value, s=100, edgecolor='black', linewidths=0.5, label='nearby galaxies', zorder=1, **ckwarg)
ax22.scatter(X_PA_supergal, Z_PA_supergal, c=cfun(vpec_PA_LS.value), s=300, marker='*', edgecolor='black', label='Pisces A', zorder=3, **ckwarg)
#ax22.scatter(X_closeGal_supergal, Z_closeGal_supergal, c=cfun(vpec_closeGal_GSR.value),
#             s=100, marker='s', edgecolor='black', label=name_closeGal, zorder=3, **ckwarg)

ax22.quiver(*origin, X_LV_supergal, Z_LV_supergal, label='Local Void', zorder=2, **qwarg)
ax22.quiver(X_PA_supergal, Z_PA_supergal, X_NE_supergal-X_PA_supergal, Z_NE_supergal-Z_PA_supergal, label='NEe', zorder=2, **akwargs(1000))
ax22.add_artist(rec2)

ax22.grid()
ax22.set_xticks([-10, 0, 10])
ax22.set_xticklabels(['$-10$', '$0$', '$10$'])
ax22.set_yticks([-10, 0, 10])
ax22.set_yticklabels(['$-10$', '$0$', '$10$'])
ax22.set_xlabel('$X_{\\rm SG}$ (Mpc)')
ax22.set_ylabel('$Z_{\\rm SG}$ (Mpc)')  #, rotation=270, labelpad=10)
ax22.set_axisbelow(True)

for a in [ax11, ax12, ax21, ax22]:
    a.set_xlim(-18, 18)
    a.set_ylim(-18, 18)

ax22.set_axisbelow(True)

plt.savefig(path.join(con.fullpath+'Tests/', 'PA_vpec_3Ddistribution.pdf'))
plt.close()

# # velocity distribution in the Local Sheet
# define
maskLS = (X_supergal.value**2+Y_supergal.value**2 <= 7**2) & (np.abs(Z_supergal.value) <= 1.5)
vrad_gal_LS = vrad_GSR.value[maskLS]
vpec_gal_LS = vpec_LS.value[maskLS]# - H0.value*dist[maskLS]
#PA_vpec_gal = vrad_PA_GSR.value - H0*dist_PA.value

# radial velocities
fig, ax = plt.subplots()
ax.hist(vrad_gal_LS, bins='auto', histtype='stepfilled', edgecolor='black', facecolor='tab:blue')
ymin, ymax = ax.get_ylim()
ax.plot([vrad_PA_GSR.value, vrad_PA_GSR.value], [ymin, ymax],
        color='tab:red', linestyle='dashed', linewidth=2, zorder=1)
ax.annotate('\\textbf{Pisces A}', (1.02*vrad_PA_GSR.value, 0.98*ymax), ha='left', va='center', color='tab:red')

ax.grid()
ax.set_axisbelow(True)
ax.set_ylim(ymin, ymax)
ax.set_yticks(ax.get_yticks())
ax.set_yticklabels([int(yt) for yt in ax.get_yticks()])
ax.set_ylabel('Galaxies ($N_{\\rm LS,\\ total} = %i$)' % len(vrad_gal_LS))
ax.set_xticks(range(-200, 901, 100))
ax.set_xlabel('Radial Velocity ($v_{\\rm GSR}$/km s$^{-1}$)')

plt.savefig(path.join(con.fullpath+'Tests/', 'PA_vrad_LS_distribution.pdf'))
plt.close()

# peculiar velocities
fig, ax = plt.subplots()
ax.hist(vpec_gal_LS, bins='auto', histtype='stepfilled', edgecolor='black', facecolor='tab:blue')
ymin, ymax = ax.get_ylim()
ax.plot([vpec_PA_LS.value, vpec_PA_LS.value], [ymin, ymax], color='tab:red', linestyle='dashed', linewidth=2, zorder=1)
ax.annotate('\\textbf{Pisces A}', (0.7*vpec_PA_LS.value, 1), ha='left', va='center', color='tab:red')

ax.grid()
ax.set_axisbelow(True)
ax.set_ylim(0, 25)
ax.set_yticks(ax.get_yticks())
ax.set_yticklabels([int(yt) for yt in ax.get_yticks()])
ax.set_ylabel('Galaxies ($N_{\\rm LS,\\ total} = %i$)' % len(vrad_gal_LS))
ax.set_xlabel('Peculiar Velocity ($v_{\\rm GSR}$/km s$^{-1}$)')

plt.savefig(path.join(con.fullpath+'Tests/', 'PA_vpec_LS_distribution.pdf'))
plt.close()

# # velocity distribution just outside the Local Sheet
# define
#maskoLS = (X_supergal.value**2+Y_supergal.value**2 > 6**2) & (np.abs(Z_supergal.value) > 0.5) & (X_supergal.value**2+Y_supergal.value**2 <= 8**2) & (np.abs(Z_supergal.value) <= 2.5)
#maskoLS = (np.abs(Z_supergal.value) <= 2.5) & (np.abs(Z_supergal.value) >= 0.5)
maskoLS = ((X_supergal.value**2 + Y_supergal.value**2 <= 9**2) & (X_supergal.value**2 + Y_supergal.value**2 >= 5**2)) & ((Z_supergal.value <= 3.5) & (Z_supergal.value >= -0.5))
vrad_gal_oLS = vrad_GSR.value[maskoLS]
vpec_gal_oLS = vpec_LS.value[maskoLS]

# radial velocities
fig, ax = plt.subplots()
ax.hist(vrad_gal_oLS, bins='auto', histtype='stepfilled', edgecolor='black', facecolor='tab:blue')
ymin, ymax = ax.get_ylim()
ax.plot([vrad_PA_GSR.value, vrad_PA_GSR.value], [ymin, 100*ymax],
        color='tab:red', linestyle='dashed', linewidth=2, zorder=1)
ax.annotate('\\textbf{Pisces A}', (1.02*vrad_PA_GSR.value, 0.98*ymax),
            ha='left', va='top', color='tab:red', rotation=270)

ax.grid()
ax.set_axisbelow(True)
ax.set_ylim(ymin, ymax)
ax.set_yticks(ax.get_yticks())
ax.set_yticklabels([int(yt) for yt in ax.get_yticks()])
ax.set_ylabel('Galaxies within 2 Mpc of the LS Boundary ($N_{\\rm total} = %i$)' % len(vrad_gal_oLS))
#ax.set_xticks(range(400, 1401, 100))
ax.set_xlabel('Radial Velocity ($v_{\\rm GSR}$/km s$^{-1}$)')

plt.savefig(path.join(con.fullpath+'Tests/', 'PA_vrad_nearLS_distribution.pdf'))
plt.close()

# peculiar velocities
fig, ax = plt.subplots()
ax.hist(vpec_gal_oLS, bins='auto', histtype='stepfilled', edgecolor='black', facecolor='tab:blue')
ymin, ymax = ax.get_ylim()
ax.plot([vpec_PA_LS.value, vpec_PA_LS.value], [ymin, ymax], color='tab:red', linestyle='dashed', linewidth=2, zorder=1)
ax.annotate('\\textbf{Pisces A}', (1.7*vpec_PA_LS.value, 0.95*ymax),
            ha='left', va='top', color='tab:red', rotation=90)

ax.grid()
ax.set_axisbelow(True)
ax.set_ylim(0, 20)
ax.set_yticklabels(['%i' % yt for yt in ax.get_yticks()])
ax.set_ylabel('Galaxies within 2 Mpc of the LS Boundary ($N_{\\rm total} = %i$)' % len(vrad_gal_oLS))
#ax.set_xticks(range(-200, 801, 100))
ax.set_xlabel('Peculiar Velocity ($v_{\\rm GSR}$/km s$^{-1}$)')

plt.savefig(path.join(con.fullpath+'Tests/', 'PA_vpec_nearLS_distribution.pdf'))
plt.close()

# 3D plot (animated!!!)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def init():
    ax.scatter(0, 0, 0, c='black', s=100, zorder=0, label='Supergalactic Origin')
    ax.scatter(X_supergal[maskoLS], Y_supergal[maskoLS], Z_supergal[maskoLS], c=vpec_gal_oLS, s=100, edgecolor='black', linewidths=0.5,
               label='Galaxies $<2$ Mpc of LS Boundary', zorder=1, **ckwarg)
    ax.scatter(X_PA_supergal, Y_PA_supergal, Z_PA_supergal, c=cfun(vrad_PA_GSR.value - H0*dist_PA.value), s=300, marker='*', edgecolor='black', label='Pisces A', zorder=3, **ckwarg)
    ax.scatter(X_closeGal_supergal, Y_closeGal_supergal, Z_closeGal_supergal, c=cfun(vrad_closeGal_GSR.value - H0*coord_closeGal.distance.value),
               s=30, marker='s', edgecolor='black', label='AGC 748778')
    ax.quiver(X_PA_supergal, Y_PA_supergal, Z_PA_supergal, X_NE_supergal-X_PA_supergal, Y_NE_supergal-Y_PA_supergal, Z_NE_supergal-Z_PA_supergal, edgecolor='black', facecolor=cfun(vrad_PA_GSR.value - H0*dist_PA.value),
              linewidth=1, length=2, arrow_length_ratio=0.5, normalize=True, label='NEe', zorder=2)# **akwargs(2000))
    #       add outline of Local Sheet
    from matplotlib.patches import Circle
    cir_top = Circle((0, 0), 7, facecolor='tab:gray', alpha=0.2, edgecolor='black', linewidth=2, label='')
    cir_bot = Circle((0, 0), 7, facecolor='tab:gray', alpha=0.2, edgecolor='black', linewidth=2, label='')
    ax.add_patch(cir_top)
    art3d.pathpatch_2d_to_3d(cir_top, z=1.5, zdir='z')
    ax.add_patch(cir_bot)
    art3d.pathpatch_2d_to_3d(cir_bot, z=-1.5, zdir='z')
    us = np.linspace(0, 2*np.pi, 50)
    zs = np.linspace(-1.5, 1.5, 2)
    us, zs = np.meshgrid(us, zs)
    xs, ys = 7*np.cos(us), 7*np.sin(us)
    ax.plot_surface(xs, ys, zs, color='tab:gray', alpha=0.2, label='')

    leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), fontsize='small', framealpha=1, ncol=2)
    leg.legendHandles[-2]._sizes = [60]
    ax.set_xlim(-9, 9)
    ax.set_xlabel('X$_{\\rm SG}$ (Mpc)')
    ax.set_ylim(-9, 9)
    ax.set_ylabel('Y$_{\\rm SG}$ (Mpc)')
    ax.set_zlim(-9, 9)
    ax.set_zlabel('Z$_{\\rm SG}$ (Mpc)')
    return fig,

def animate(angle):
    ax.view_init(elev=angle/4-45, azim=angle-180+45)
    return fig,

movie = anim(fig, animate, init_func=init, frames=360, interval=20, blit=True)
movie.save(path.join(con.fullpath + 'Tests/', 'PA_3Ddistribution_animation_closeGal.mp4'), fps=30)
plt.close()
del movie

# 3D plot (single frame!)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(0, 0, 0, c='black', s=100, zorder=0, label='Supergalactic Origin')
ax.scatter(X_supergal[maskoLS], Y_supergal[maskoLS], Z_supergal[maskoLS], c=vpec_gal_oLS, s=100, edgecolor='black', linewidths=0.5,
           label='Galaxies $<2$ Mpc of LS Boundary', zorder=1, **ckwarg)
ax.scatter(X_PA_supergal, Y_PA_supergal, Z_PA_supergal, c=cfun(vrad_PA_GSR.value - H0*dist_PA.value), s=300, marker='*', edgecolor='black', label='Pisces A', zorder=3, **ckwarg)
ax.quiver(X_PA_supergal, Y_PA_supergal, Z_PA_supergal, X_NE_supergal-X_PA_supergal, Y_NE_supergal-Y_PA_supergal, Z_NE_supergal-Z_PA_supergal,
          edgecolor='black', facecolor=cfun(vrad_PA_GSR.value - H0*dist_PA.value),
          linewidth=1, length=2, arrow_length_ratio=0.5, normalize=True, label='NEe', zorder=2)# **akwargs(2000))
#       add outline of Local Sheet
from matplotlib.patches import Circle
cir_top = Circle((0, 0), 7, facecolor='tab:gray', alpha=0.2, edgecolor='black', linewidth=2, label='')
cir_bot = Circle((0, 0), 7, facecolor='tab:gray', alpha=0.2, edgecolor='black', linewidth=2, label='')
ax.add_patch(cir_top)
art3d.pathpatch_2d_to_3d(cir_top, z=1.5, zdir='z')
ax.add_patch(cir_bot)
art3d.pathpatch_2d_to_3d(cir_bot, z=-1.5, zdir='z')
us = np.linspace(0, 2*np.pi, 50)
zs = np.linspace(-1.5, 1.5, 2)
us, zs = np.meshgrid(us, zs)
xs, ys = 7*np.cos(us), 7*np.sin(us)
ax.plot_surface(xs, ys, zs, color='tab:gray', alpha=0.2, label='')

leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), fontsize='small', framealpha=1, ncol=2)
leg.legendHandles[-2]._sizes = [60]
ax.set_xlim(-9, 9)
ax.set_xlabel('X$_{\\rm SG}$ (Mpc)')
ax.set_ylim(-9, 9)
ax.set_ylabel('Y$_{\\rm SG}$ (Mpc)')
ax.set_zlim(-9, 9)
ax.set_zlabel('Z$_{\\rm SG}$ (Mpc)')

ax.view_init(elev=42, azim=-12)

plt.savefig(path.join(con.fullpath+'Tests/', 'PA_3Ddistribution_image.pdf'))
plt.close()

"""
# # galaxy luminosity function (using the classical Vmax method)
volume = 4/3 * np.pi * dist**3  # Mpc^3; assuming h^{-3, -2, -1, 1, 2, 3} ~ 1
limit_app = 13.5  # mag; 2MASS apparent magnitude limit
def Vmax(M, m_lim):
    d = 1e-6 * 10**((m_lim - M + 5)/5)
    return 4/3 * np.pi * d**3
volume_max = Vmax(Kabs, limit_app).data

def phi_est(V, V_max, M, dM):
    bins = np.arange(M.min()-1, M.max()+1, dM)
    centers = (bins[1:] + bins[:-1])/2
    phi = np.zeros_like(centers)
    ephi = np.zeros_like(centers)
    cts = np.zeros_like(centers)
    completenessmask = (V / V_max <= 0.5)
    for i, cen in enumerate(centers):
        binmask = (M > cen - dM/2) & (M <= cen + dM/2)
        totalmask = binmask & completenessmask
        vmax = V_max[totalmask]
        phi[i] = np.sum([1/v for v in vmax])
        cts[i] = len(vmax)
    ephi = phi / np.sqrt(cts)
    \"""
    for i, Mabs in enumerate(M):
        for j, cen in enumerate(centers):
            if (Mabs > cen-dM/2) & (Mabs <= cen+dM/2) & (V[i]/V_max[i] <= 0.5):
                vmax_inv[j] += 1/V_max[i]
                evmax_inv[j] += 1
    evmax_inv = vmax_inv/np.sqrt(evmax_inv)  # Poisson errors??
    \"""
    return centers, phi, ephi
Mbins, phi_M, ephi_M = phi_est(volume[maskLS], volume_max[maskLS], Kabs[maskLS], 0.25)
Mbins, phi_M, ephi_M = Mbins[phi_M > 0], phi_M[phi_M > 0], ephi_M[phi_M > 0]

def Schechter(M, phi_star, M_star, alpha):
    coeff = 0.4 * np.log(10) * phi_star
    term1 = (10**(0.4*(M_star-M)))**(1+alpha)
    term2 = np.exp(-10**(0.4*(M_star-M)))
    return coeff * term1 * term2
from scipy.optimize import curve_fit
guess = [1.16e-2, -23.39, -1.09]  # Kochanek et al. 2001, ApJ, 560, 566
popt, pcov = curve_fit(Schechter, Mbins, phi_M, p0=guess, sigma=ephi_M)

fig, ax = plt.subplots()
err = ax.errorbar(Mbins, phi_M, yerr=ephi_M, fmt='none', ecolor='black', capsize=2.5, zorder=0, label='Poisson Errors')
sca = ax.scatter(Mbins, phi_M, edgecolors='black', linewidths=0.5, zorder=1, label='$\\phi_{\\rm est}(M_{\\rm K_{\\rm s}})$')
ref = ax.plot(Mbins, Schechter(Mbins, 1.16e-6, -23.39, -1.09), color='black', linewidth=1, linestyle='dashed', zorder=0,
              label='Kochanek+2019:\n\
                     $\\phi_\star$ = %.1f $\\times 10^{-2}$ Mpc$^{-3}$\n\
                     M$_{\\rm K_{\\rm s}\star} = %.1f$ mag\n\
                     $\\alpha = %.1f$' % (1.16, -23.39, -1.09))
fit = ax.plot(Mbins, Schechter(Mbins, *popt), color='tab:red', linewidth=1.5, zorder=0,
              label='Galaxies in the LS:\n\
                     $\\phi_\star$ = %.1f $\\times 10^{-9}$ Mpc$^{-3}$\n\
                     M$_{\\rm K_{\\rm s}\star} = %.1f$ mag\n\
                     $\\alpha = %.1f$' % (popt[0]*1e9, popt[1], popt[2]))

lH = [err, fit[0], sca, ref[0]]
leg = ax.legend(lH, [l.get_label() for l in lH], markerfirst=False,
                loc='upper right', framealpha=1, fontsize='xx-small', ncol=2)
ax.grid()
ax.set_axisbelow(True)
ax.invert_xaxis()
ax.set_yscale('log')
ax.set_ylim(5e-10, 1e-2)
ax.set_xlabel('M$_{\\rm K_{\\rm s}}$ (mag)')
ax.set_ylabel('$\\phi$(M) or $10^{-4} \\times \\phi$(M)')

# 3D plot (alpha, dec, dist)
#       find galaxy closest to "origin"
idx_origin = np.where(dist == dist.min())[0][0]
alpha_origin, delta_origin, dist_origin = alpha[idx_origin], delta[idx_origin], dist[idx_origin]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(alpha_origin, delta_origin, dist_origin, c='black', s=100, zorder=0, label='Supergalactic `Origin\'')
ax.scatter(alpha[maskoLS], delta[maskoLS], dist[maskoLS], c=vpec_gal_oLS, s=100, edgecolor='black', linewidths=0.5,
           label='Galaxies $<2$ Mpc of LS Boundary', zorder=1, **ckwarg)
ax.scatter(PAcoord.ra.deg, PAcoord.dec.deg, dist_PA.value, c=cfun(vrad_PA_GSR.value - H0*dist_PA.value), s=300,
           marker='*', edgecolor='black', label='Pisces A', zorder=3, **ckwarg)
ax.quiver(NEcoord.ra.deg, NEcoord.dec.deg, NEcoord.distance.value,
          NEcoord.ra.deg-PAcoord.ra.deg, NEcoord.dec.deg-PAcoord.dec.deg, 0,
          edgecolor='black', linewidth=1, length=25, arrow_length_ratio=0.1, normalize=True, label='NEe', zorder=2)


leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), fontsize='small', framealpha=1, ncol=2)
leg.legendHandles[-1]._sizes = [60]
ax.set_xlim(360, 0)
ax.set_xlabel('$\\alpha$ (deg)')
ax.set_ylabel('$\\delta$ (deg)')
ax.set_zlabel('Distance (Mpc)')

# 3D plot (Aitoff projection)
#       adjust positions for wraparound
ra_origin, dec_origin = np.pi - coords[idx_origin].ra.wrap_at(180*u.deg).radian, coords[idx_origin].dec.radian
ra_oLS, dec_oLS = coords[maskoLS].ra.wrap_at(180*u.deg).radian, coords[maskoLS].dec.radian
#       supergalactic plane
l_SG, b_SG = np.linspace(0, 360, 10000), np.zeros(10000)
planecoord = SkyCoord(l_SG, b_SG, unit=u.deg, frame='supergalactic')
planera, planedec = planecoord.icrs.ra.wrap_at(180*u.deg).radian, planecoord.icrs.dec.radian
#       Eridanus Void
#alpha_EV, dec_EV = np.concatenate((np.linspace(23.6666666, 24, 500), np.linspace(0, 1.73, 500))), np.linspace(-7, 7, 1000)
#EVcoord = SkyCoord(alpha_EV, dec_EV, unit=(u.hourangle, u.deg), frame='icrs')
#EVra, EVdec = EVcoord.icrs.ra.wrap_at(180*u.deg).radian, EVcoord.icrs.dec.radian

RAlos_EV, RAhis_EV = 23.66666*np.ones(1000), 1.75*np.ones(1000)
DEClos_EV, DEChis_EV = -7*np.ones(1000), 7*np.ones(1000)
EVb1 = SkyCoord(RAlos_EV, DEClos_EV, unit=(u.hourangle, u.deg))
EVb1_ra, EVb1_dec = EVb1.icrs.ra.wrap_at(180*u.deg).radian, EVb1.icrs.dec.radian
EVb2 = SkyCoord(RAlos_EV, DEChis_EV, unit=(u.hourangle, u.deg))
EVb2_ra, EVb2_dec = EVb2.icrs.ra.wrap_at(180*u.deg).radian, EVb2.icrs.dec.radian
EVb3 = SkyCoord(RAhis_EV, DEClos_EV, unit=(u.hourangle, u.deg))
EVb3_ra, EVb3_dec = EVb3.icrs.ra.wrap_at(180*u.deg).radian, EVb3.icrs.dec.radian
EVb4 = SkyCoord(RAhis_EV, DEChis_EV, unit=(u.hourangle, u.deg))
EVb4_ra, EVb4_dec = EVb4.icrs.ra.wrap_at(180*u.deg).radian, EVb4.icrs.dec.radian

#       local sheet
xls = np.linspace(-7, 7, 10000)
yls = lambda x: float(f'{x}1') * np.sqrt(7**2 - xls**2)
zls = lambda x: float(f'{x}1') * 1.5 * np.ones(10000)

def framecoords(xx, yy, zz, unit=u.Mpc, frame='supergalactic', reptype='cartesian'):
    cds = SkyCoord(xx, yy, zz, unit=unit, frame=frame, representation_type=reptype)
    RA, DEC = cds.icrs.ra.wrap_at(180*u.deg).radian, cds.icrs.dec.radian
    return RA, DEC

fig = plt.figure()
ax = fig.add_subplot(111, projection='aitoff')
ax.scatter(planera, planedec, c='tab:blue', s=1, zorder=0, label='Supergalactic Plane')
sheetra, sheetdec = framecoords(xls, yls('+'), zls('+'))
ax.scatter(sheetra, sheetdec, c='tab:red', s=5, zorder=0, label='Local Sheet')
sheetra, sheetdec = framecoords(xls, yls('-'), zls('+'))
ax.scatter(sheetra, sheetdec, c='tab:red', s=5, zorder=0, label='')
sheetra, sheetdec = framecoords(xls, yls('+'), zls('-'))
ax.scatter(sheetra, sheetdec, c='tab:red', s=5, zorder=0, label='')
sheetra, sheetdec = framecoords(xls, yls('-'), zls('-'))
ax.scatter(sheetra, sheetdec, c='tab:red', s=5, zorder=0, label='')
ax.plot([EVb1_ra[0], EVb3_ra[0]], [EVb3_dec[0], EVb3_dec[0]], linewidth=2, zorder=0, color='tab:purple', label='Eridanus Void')
ax.plot([EVb1_ra[0], EVb3_ra[0]], [EVb4_dec[0], EVb4_dec[0]], linewidth=2, zorder=0, color='tab:purple', label='')
ax.plot([EVb1_ra[0], EVb1_ra[0]], [EVb3_dec[0], EVb4_dec[0]], linewidth=2, zorder=0, color='tab:purple', label='')
ax.plot([EVb3_ra[0], EVb3_ra[0]], [EVb3_dec[0], EVb4_dec[0]], linewidth=2, zorder=0, color='tab:purple', label='')

ax.scatter(ra_origin, dec_origin, c='black', s=100, zorder=0, label='Supergalactic `Origin\'')
ax.scatter(coords.ra.wrap_at(180*u.deg).radian, coords.dec.radian, s=1, alpha=0.5, edgecolor='black', zorder=0,
           label='All Nearby Galaxies')
ax.scatter(ra_oLS, dec_oLS, s=dist[maskoLS]**2.5, c=vpec_GSR.value[maskoLS], edgecolor='black', linewidths=0.5,
           label='Galaxies $<2$ Mpc of LS Boundary', zorder=1, **ckwarg)
ax.scatter(coord_PA.ra.wrap_at(180*u.deg).radian, coord_PA.dec.radian, s=coord_PA.distance.value**2.5, c=cfun(vpec_PA_GSR.value), marker='*',
           edgecolor='black', label='Pisces A', zorder=3, **ckwarg)

ax.grid()
ax.set_axisbelow(True)
ax.set_xticks([15*x*(np.pi/180) for x in [-12, -6, 0, 6, 12]])
ax.set_xticklabels(['$%i^{\\rm h}$' % (x*(180/np.pi)/15) for x in ax.get_xticks()])
ax.set_xlabel('Right Ascension (J2000)', labelpad=15)
ax.set_ylabel('Declination (J2000)', labelpad=5)
leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), fontsize='xx-small', ncol=2)
for mrkr in leg.legendHandles:
    mrkr._sizes = [100]
leg.legendHandles[-2]._facecolors = 'tab:yellow'
"""
