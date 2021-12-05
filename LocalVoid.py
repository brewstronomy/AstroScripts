#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: luca (lucabeale@gmail.com)
"""

import numpy as np
import matplotlib.pyplot as plt
import PiscesA.constant as con
import astropy.units as u
from os import path
from astropy.coordinates import SkyCoord, Distance
from astropy.table import Table
from astropy.coordinates.matrix_utilities import rotation_matrix
from astropy.visualization import simple_norm
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import gaussian_filter

# # preliminaries
# following Calibrated_Context.ipynb from github.com/eteq/piscdwarfs_hst
filepath = con.fullpath + 'Local_Void/Data/'
filename = 'lvol.ecsv'

npix = 1e4

# # import data
data = Table.read(path.join(filepath, filename))
alpha, delta, dist, vrad = data['RA'], data['Dec'], data['distance'], data['vhelio']

#PAd = np.array([5.64, 5.64 - 0.13, 5.64 + 0.15])
#PA_dist = Distance(PAd, u.Mpc)  # includes inner 68%
PA_dist = Distance(5.64, u.Mpc)
PAcoord = SkyCoord('00h14m46s', '+10d48m47.01s', distance=PA_dist)

#PBd = np.array([8.89, 8.89 - 0.85, 8.89 + 0.75])
#PB_dist = Distance(PBd, u.Mpc)  # includes inner 68%
PB_dist = Distance(8.89, u.Mpc)
PBcoord = SkyCoord('01h19m11.7s', '+11d07m18.22s', distance=PB_dist)

LV_dist = Distance(5, u.Mpc)  # avg of galaxies in Rizzi+2017
LVcoord = SkyCoord('18h38m', '18d', distance=LV_dist)

NEcoord = SkyCoord('00h14m51s', '+10d50m48s', distance=PA_dist)
#NEcoord = SkyCoord('00h14m57s', '+10d54m00s', distance=PA_dist)

# # coordinate transformations
# apply distance and completness limit cuts
dist_mask = (dist > 0) & (dist < 15)

K_absmag = data['K'] - (5*np.log10(data['distance']*1e6)-5)
limit_2MASS = 13.5*u.mag-Distance(2**0.5*10*u.Mpc).distmod
mag_mask = K_absmag < limit_2MASS.value
mask = dist_mask & mag_mask

alpha, delta, dist, vrad = alpha[mask], delta[mask], dist[mask], vrad[mask]

coords = SkyCoord(alpha*u.deg, delta*u.deg, distance=Distance(dist, u.Mpc))
X, Y, Z = coords.supergalactic.cartesian.xyz

PA_X, PA_Y, PA_Z = PAcoord.supergalactic.cartesian.xyz
PB_X, PB_Y, PB_Z = PAcoord.supergalactic.cartesian.xyz
PA_vrad, PB_vrad = 236, 615

LV_X, LV_Y, LV_Z = LVcoord.supergalactic.cartesian.xyz
NE_X, NE_Y, NE_Z = NEcoord.supergalactic.cartesian.xyz

"""
rot1 = rotation_matrix(-LVcoord.galactic.b.value, 'y')
rot2 = rotation_matrix(LVcoord.galactic.l.value, 'z')
rot_matrix = rot1 @ rot2

Z, X, Y = np.dot(rot_matrix, [X, Y, Z])
PA_Z, PA_X, PA_Y = np.dot(rot_matrix, PAcoord.galactic.cartesian.xyz)
PB_Z, PB_X, PA_Y = np.dot(rot_matrix, PBcoord.galactic.cartesian.xyz)
LV_Z, LV_X, LV_Y = np.dot(rot_matrix, LVcoord.galactic.cartesian.xyz)
NE_Z, NE_X, NE_Y = np.dot(rot_matrix, NEcoord.galactic.cartesian.xyz)
"""

# # plotting
# X vs Y
pts = np.array([X, Y]).T
delaunay, voronoi = Delaunay(pts), Voronoi(pts)
simp = delaunay.simplices

# for each point p, select simplices where p is a vertex
idx = []  # indices of the simplex array
psimp = []  # array of simplices per point
for i in range(len(pts)):
    idx.append(np.where(simp == i)[0])
    psimp.append(pts[simp[idx[i]]])

# calculate the area of the contiguous Voronoi cell
area = []  # in pix^2
for i in range(len(pts)):
    a = 0
    for s in psimp[i]:
        mat = np.insert(s.T, 2, [1, 1, 1], axis=0)
        a += 0.5 * np.abs(np.linalg.det(mat))  # triangle area given vertices
    area.append(a)

# assign density as inverse area
rho = [1/a for a in area]  # in pix^-2; len = num. of simplices
rho = gaussian_filter(rho, 0.5)

# use a piecewise linear interpolation to estimate the density field
density = LinearNDInterpolator(delaunay, rho, fill_value=np.nan)
xs, ys = np.linspace(X.value.min(), X.value.max(), len(X)), np.linspace(Y.value.min(), Y.value.max(), len(Y))
xx, yy = np.meshgrid(xs, ys, indexing='ij')
Rho = density(xx, yy)

# also plot critical points
sec_grad_x = np.diff(Rho, n=1, axis=0)
sec_grad_y = np.diff(Rho, n=1, axis=1)
cp = []
# starts from 1 because diff function gives a forward difference
for i in range(1, len(sec_grad_x)-1):
    for j in range(1, len(sec_grad_y)-1):
        # check when the difference changes its sign
        if ((sec_grad_x[i-1, j] < 0) != (sec_grad_x[i-1+1, j] < 0)) and ((sec_grad_y[i, j-1] < 0) != (sec_grad_y[i, j-1+1] < 0)):
            cp.append([xs[i], ys[j]])
cp = np.array(cp)

norm = simple_norm(Rho, 'log', percent=99)

fig, ax = plt.subplots()
ax.triplot(pts[:, 0], pts[:, 1], delaunay.simplices, linewidth=0.5,
           c=con.colors[1], zorder=1)
p = ax.imshow(Rho.T, cmap='bone', norm=norm, zorder=0, aspect='auto',
              extent=[X.value.min(), X.value.max(), Y.value.min(), Y.value.max()])
ax.scatter(X, Y, c='black', s=10, edgecolor='none', label='nearby galaxies',
           zorder=2)
ax.scatter(PA_X, PA_Y, c=con.colors[-2], s=300, marker='*', edgecolor='black',
           label='Pisces A', zorder=2)
ax.legend(loc='lower right', fontsize='x-small', framealpha=1)
ax.set_xlim(-10, 10)
ax.set_xticks(np.arange(-8, 9, 4))
ax.set_xlabel('$X_{\\rm SG}$ (Mpc)')
ax.set_ylim(-10, 10)
ax.set_yticks(np.arange(-8, 9, 4))
ax.set_ylabel('$Y_{\\rm SG}$ (Mpc)')
savepath = con.fullpath + 'Local_Void/Plots'
savename = 'CosmicWeb_Del+Density_XvsY.pdf'
plt.savefig(path.join(savepath, savename))
plt.close()
'''
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.triplot(pts[:, 0], pts[:, 1], delaunay.simplices, linewidth=0.5,
            c=con.colors[1], zorder=1)
ax1.scatter(*cp.T, c='red', s=10, edgecolor='none', zorder=2)
ax1.set_title('Delaunay Triangulation')
voronoi_plot_2d(voronoi, ax=ax2, line_colors='purple', line_width=0.5,
                show_points=False, show_vertices=False)
ax2.set_title('Voronoi Diagram')
for a in [ax1, ax2]:
    p = a.imshow(Rho.T, cmap='bone', norm=norm, zorder=0, aspect='auto',
                 extent=[X.value.min(), X.value.max(), Y.value.min(), Y.value.max()])
    a.scatter(X, Y, c='black', s=10, edgecolor='none', label='nearby galaxies',
              zorder=2)
    a.scatter(PA_X, PA_Y, c=con.colors[-2], s=300, marker='*', edgecolor='black',
              label='Pisces A', zorder=2)
#    a.scatter(PB_X, PB_Y, c=con.colors[-2], s=300, marker='*', edgecolor='black',
#              label='Pisces A', zorder=2)

    a.legend(loc='lower right', fontsize='x-small', framealpha=1)
    a.set_xlim(-10, 10)
    a.set_xticks(np.arange(-8, 9, 4))
    a.set_xlabel('$x_{\\rm LV}$ (Mpc)')
    a.set_ylim(-10, 10)
    a.set_yticks(np.arange(-8, 9, 4))
ax1.set_ylabel('$y_{\\rm LV}$ (Mpc)')
ax2.set_yticklabels([])
fig.subplots_adjust(wspace=0)

plt.colorbar(p)
'''

# X vs Z
pts = np.array([X, Z]).T
delaunay, voronoi = Delaunay(pts), Voronoi(pts)
simp = delaunay.simplices

# for each point p, select simplices where p is a vertex
idx = []  # indices of the simplex array
psimp = []  # array of simplices per point
for i in range(len(pts)):
    idx.append(np.where(simp == i)[0])
    psimp.append(pts[simp[idx[i]]])

# calculate the area of the contiguous Voronoi cell
area = []  # in pix^2
for i in range(len(pts)):
    a = 0
    for s in psimp[i]:
        mat = np.insert(s.T, 2, [1, 1, 1], axis=0)
        a += 0.5 * np.abs(np.linalg.det(mat))
    area.append(a)

# assign density as inverse area
rho = [1/a for a in area]  # in pix^-2
#rho = gaussian_filter(rho, 0.2)

# use a piecewise linear interpolation to estimate the density field
density = LinearNDInterpolator(delaunay, rho, fill_value=0)
xs, ys = np.linspace(X.value.min(), X.value.max(), 1e3), np.linspace(Z.value.min(), Z.value.max(), 1e3)
xx, yy = np.meshgrid(xs, ys, indexing='ij')
Rho = density(xx, yy)
Rho = gaussian_filter(Rho, sigma=5)

norm = simple_norm(Rho, 'log', percent=99)

fig, ax = plt.subplots()
ax.triplot(pts[:, 0], pts[:, 1], delaunay.simplices, linewidth=1, c=con.colors[1], zorder=1)
p = ax.imshow(Rho.T, cmap='bone', norm=norm, zorder=0, aspect='auto',
              extent=[X.value.min(), X.value.max(), Z.value.min(), Z.value.max()])
#voronoi_plot_2d(voronoi, ax=ax, line_colors='black', line_width=1, show_points=False, show_vertices=False, zorder=1)
ax.scatter(X, Z, c='black', s=30, edgecolor='none', label='nearby galaxies', zorder=2)
ax.scatter(PA_X, PA_Z, c=con.colors[-2], s=600, marker='*', edgecolor='black',
           label='Pisces A', zorder=3)
ax.quiver(PA_X.value, PA_Z.value, NE_X.value-PA_X.value, NE_Z.value-PA_Z.value,
          label='NEext', zorder=2, angles='xy', scale_units='xy', scale=2e-3,
          color=con.colors[-2], edgecolor='black', linewidth=0.5)
leg = ax.legend(loc='lower right', fontsize='small', framealpha=1, markerfirst=False)
marker = leg.legendHandles[1]
marker._sizes = [400]
ax.set_xlim(-7, 7)
#ax.set_xticks(np.arange(-8, 9, 4))
ax.set_xlabel('$X_{\\rm SG}$ (Mpc)')
ax.set_ylim(-7, 7)
#ax.set_yticks(np.arange(-8, 9, 4))
ax.set_ylabel('$Z_{\\rm SG}$ (Mpc)')

savepath = con.fullpath + 'Tests/'
savename = 'PA_CosmicWeb_XvsZ.pdf'
plt.savefig(path.join(savepath, savename))
plt.close()

# Y vs Z
pts = np.array([Y, Z]).T
delaunay, voronoi = Delaunay(pts), Voronoi(pts)
simp = delaunay.simplices

# for each point p, select simplices where p is a vertex
idx = []  # indices of the simplex array
psimp = []  # array of simplices per point
for i in range(len(pts)):
    idx.append(np.where(simp == i)[0])
    psimp.append(pts[simp[idx[i]]])

# calculate the area of the contiguous Voronoi cell
area = []  # in pix^2
for i in range(len(pts)):
    a = 0
    for s in psimp[i]:
        mat = np.insert(s.T, 2, [1, 1, 1], axis=0)
        a += 0.5 * np.abs(np.linalg.det(mat))
    area.append(a)

# assign density as inverse area
rho = [1/a for a in area]  # in pix^-2

# use a piecewise linear interpolation to estimate the density field
density = LinearNDInterpolator(delaunay, rho, fill_value=np.nan)
xs, ys = np.linspace(Y.value.min(), Y.value.max(), 1e3), np.linspace(Z.value.min(), Z.value.max(), 1e3)
xx, yy = np.meshgrid(xs, ys, indexing='ij')
Rho = density(xx, yy)

norm = simple_norm(Rho, 'log', percent=99)

fig, ax = plt.subplots()
ax.triplot(pts[:, 0], pts[:, 1], delaunay.simplices, linewidth=0.5,
           c=con.colors[1], zorder=1)
p = ax.imshow(Rho.T, cmap='bone', norm=norm, zorder=0, aspect='auto',
              extent=[Y.value.min(), Y.value.max(), Z.value.min(), Z.value.max()])
ax.scatter(Y, Z, c='black', s=10, edgecolor='none', label='nearby galaxies',
           zorder=2)
ax.scatter(PA_Y, PA_Z, c=con.colors[-2], s=300, marker='*', edgecolor='black',
           label='Pisces A', zorder=2)
ax.legend(loc='lower right', fontsize='x-small', framealpha=1)
ax.set_xlim(-10, 10)
ax.set_xticks(np.arange(-8, 9, 4))
ax.set_xlabel('$Y_{\\rm SG}$ (Mpc)')
ax.set_ylim(-10, 10)
ax.set_yticks(np.arange(-8, 9, 4))
ax.set_ylabel('$Z_{\\rm SG}$ (Mpc)')
savepath = con.fullpath + 'Local_Void/Plots'
savename = 'CosmicWeb_Del+Density_YvsZ.pdf'
plt.savefig(path.join(savepath, savename))
plt.close()