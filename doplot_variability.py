import numpy as np
import matplotlib.pyplot as pl
import healpy as hp
import pickle

def graticule(ax):
    ax.axhline(0.0, color='k')
    for i in range(2):
        ax.axhline(30+30*i, color='k', linestyle=':')
        ax.axhline(-30-30*i, color='k', linestyle=':')
    for i in range(12):
        ax.axvline(-180+30*i, color='k', linestyle=':')

def avg_map(T, inclination, n=20):
    proj_fun = lambda x,y,z : hp.vec2pix(2**4, x, y, z, nest=True)
    projector = hp.projector.OrthographicProj(xsize=200, half_sky=True)

    angles = np.linspace(0,360,n)
    for i in range(n):
        if (i == 0):
            tmp = projector.projmap(T, proj_fun, rot=(angles[i], inclination, 0))
        else:
            tmp += projector.projmap(T, proj_fun, rot=(angles[i], inclination, 0))

    return tmp / n

pl.close('all')

with open('variability_sini.pk', 'rb') as handle:
    T, output, angles = pickle.load(handle)

output /= 1000.0
T /= 1000.0

proj_fun = lambda x,y,z : hp.vec2pix(2**4, x, y, z, nest=True)
proj_fun_hd = lambda x,y,z : hp.vec2pix(2**4, x, y, z, nest=True)
projector = hp.projector.CartesianProj() 

fig, ax = pl.subplots(nrows=4, ncols=4, figsize=(15,7), sharex='col', sharey=True)

median = np.median(output, axis=1)
pct = np.percentile(output, [10, 90], axis=1)
idr = pct[1, :] - pct[0, :]

target_vmin = np.min(T)
target_vmax = np.max(T)

for i in range(4):
    tmp = projector.projmap(median[i, :], proj_fun_hd) 
    mapp = ax[i, 0].imshow(tmp, cmap=pl.cm.inferno, interpolation='none', origin='lower', 
        extent=[-180, 180, -90, 90], vmin=np.min(tmp), vmax=np.max(tmp))    
    ax[i, 0].set_yticks([-90,0,90])
    ax[i, 0].set_xticks([-180,-90,0,90,180])   
    ax[i, 0].set_title(f'i={angles[i]}') 
    if (i == 0):
        ax[i, 0].text(-170, 65, 'Median', fontweight='demibold')
    
    fig.colorbar(pl.cm.ScalarMappable(norm=pl.Normalize(vmin=target_vmin, vmax=target_vmax), cmap=pl.cm.inferno), ax=ax[i, 0], label='T [kK]')

    tmp = projector.projmap(pct[0, i, :], proj_fun_hd) 
    mapp = ax[i, 1].imshow(tmp, cmap=pl.cm.inferno, interpolation='none', origin='lower', 
        extent=[-180, 180, -90, 90], vmin=np.min(tmp), vmax=np.max(tmp))
    ax[i, 1].set_yticks([-90,0,90])    
    ax[i, 1].set_xticks([-180,-90,0,90,180])
    if (i == 0):
        ax[i, 1].text(-170, 65, '10%', fontweight='demibold', color='black')
    fig.colorbar(pl.cm.ScalarMappable(norm=pl.Normalize(vmin=np.min(tmp), vmax=np.max(tmp)), cmap=pl.cm.inferno), ax=ax[i, 1], label='T [kK]')    

    tmp = projector.projmap(pct[1, i, :], proj_fun_hd) 
    mapp = ax[i, 2].imshow(tmp, cmap=pl.cm.inferno, interpolation='none', origin='lower', 
        extent=[-180, 180, -90, 90], vmin=np.min(tmp), vmax=np.max(tmp))
    ax[i, 2].set_yticks([-90,0,90])    
    ax[i, 2].set_xticks([-180,-90,0,90,180])
    if (i == 0):
        ax[i, 2].text(-170, 65, '90%', fontweight='demibold', color='black')
    fig.colorbar(pl.cm.ScalarMappable(norm=pl.Normalize(vmin=np.min(tmp), vmax=np.max(tmp)), cmap=pl.cm.inferno), ax=ax[i, 2], label='T [kK]')    

    tmp = projector.projmap(pct[1, i, :] - pct[0, i, :], proj_fun_hd) 
    mapp = ax[i, 3].imshow(tmp, cmap=pl.cm.inferno, interpolation='none', origin='lower', 
        extent=[-180, 180, -90, 90], vmin=np.min(tmp), vmax=np.max(tmp))
    ax[i, 3].set_yticks([-90,0,90])    
    ax[i, 3].set_xticks([-180,-90,0,90,180])
    if (i == 0):
        ax[i, 3].text(-170, 65, 'IDR', fontweight='demibold', color='white')
    fig.colorbar(pl.cm.ScalarMappable(norm=pl.Normalize(vmin=np.min(tmp), vmax=np.max(tmp)), cmap=pl.cm.inferno), ax=ax[i, 3], label='T [kK]')    

fig.supylabel('Latitude [deg]', y=0.6)
fig.supxlabel('Longitude [deg]', y=0)

fig.subplots_adjust(left=0.07, right=0.83, top=0.95, bottom=0.25)

ax = fig.add_axes([0.20, 0.07, 0.50, 0.14])
tmp = projector.projmap(T, proj_fun) 
mapp = ax.imshow(tmp, cmap=pl.cm.inferno, interpolation='none', origin='lower', 
    extent=[-180, 180, -90, 90], vmin=np.min(T), vmax=np.max(T))

ax.set_yticks([-90,0,90])
ax.set_xticks([-180,-90,0,90,180])            
fig.colorbar(pl.cm.ScalarMappable(norm=pl.Normalize(vmin=target_vmin, vmax=target_vmax), cmap=pl.cm.inferno), ax=ax, label='T [kK]')
ax.text(-170, 65, 'Target', fontweight='demibold', color='black')

for i in range(4):
    ax = fig.add_axes([0.85, 0.79-i*0.18, 0.15, 0.15])
    ax.imshow(avg_map(T, 90-angles[i], n=90),  extent=[-90, 90, -90, 90], cmap=pl.cm.inferno, interpolation='none', origin='lower', aspect='equal')
    ax.set_axis_off()
    
pl.savefig('figs/variability_sini.png')
pl.savefig('figs/variability_sini.pdf')

