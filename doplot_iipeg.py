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

pl.close('all')
with open('iipeg_step1.pk', 'rb') as handle:
    output, output_hd, stokesi, stokesi_median, median_stokesi, wl, phase, obs = pickle.load(handle)

with open('iipeg_step2.pk', 'rb') as handle:
    output_step2, output_hd_step2, stokesi_step2, stokesi_median_step2, median_stokesi_step2, wl_step2, phase_step2, obs_step2 = pickle.load(handle)

with open('iipeg_step4.pk', 'rb') as handle:
    output_step4, output_hd_step4, stokesi_step4, stokesi_median_step4, median_stokesi_step4, wl_step4, phase_step4, obs_step4 = pickle.load(handle)

# with open('validation.pk', 'rb') as handle:
    # output, target, sini, velocity, nangles = pickle.load(handle)

savefig = True

median = np.median(output_hd, axis=0)
mad = np.std(output_hd, axis=0)
pct = np.percentile(output_hd, [10, 90], axis=0)
idr = pct[1, :] - pct[0, :]

median_step2 = np.median(output_hd_step2, axis=0)
mad_step2 = np.std(output_hd_step2, axis=0)
pct_step2 = np.percentile(output_hd_step2, [10, 90], axis=0)
idr_step2 = pct_step2[1, :] - pct_step2[0, :]

median_step4 = np.median(output_hd_step4, axis=0)
mad_step4 = np.std(output_hd_step4, axis=0)
pct_step4 = np.percentile(output_hd_step4, [10, 90], axis=0)
idr_step4 = pct_step4[1, :] - pct_step4[0, :]

proj_fun = lambda x,y,z : hp.vec2pix(16, x, y, z, nest=True)
projector = hp.projector.CartesianProj() 

which = 2  #[0,1,2]

if (which == 0):
    fig, ax = pl.subplots(nrows=3, ncols=5, figsize=(12,8))
    for i in range(5):
        pl.axes(ax[0,i])
        hp.orthview(median, nest=True, hold=True, title=f'{i*0.2:3.1f}', cmap=pl.cm.inferno, half_sky=True, rot=(i*360/5.,0,0), min=np.min(pct), max=np.max(pct), cbar=False)
        if (i == 0):
            pl.text(-1.5, 0.9, 'Median', fontsize='large')
        if (i == 2):
            pl.text(-0.3, 1.5, r'N$_\mathrm{obs}$=12', fontsize='large')
        hp.visufunc.graticule()
        
        pl.axes(ax[1,i])
        hp.orthview(pct[0, :], nest=True, hold=True, title='', cmap=pl.cm.inferno, half_sky=True, rot=(i*360/5.,0,0), min=np.min(pct), max=np.max(pct), cbar=False)
        hp.visufunc.graticule()
        if (i == 0):
            pl.text(-1.5, 0.9, '10%', fontsize='large')

        pl.axes(ax[2,i])
        hp.orthview(pct[1, :], nest=True, hold=True, title='', cmap=pl.cm.inferno, half_sky=True, rot=(i*360/5.,0,0), min=np.min(pct), max=np.max(pct), cbar=False)
        hp.visufunc.graticule()
        if (i == 0):
            pl.text(-1.5, 0.9, '90%', fontsize='large')


    fig.subplots_adjust(bottom=0.1)
    cbar_ax = fig.add_axes([0.35, 0.08, 0.30, 0.02])
    fig.colorbar(pl.cm.ScalarMappable(norm=pl.Normalize(vmin=np.min(median), vmax=np.max(median)), cmap=pl.cm.inferno), cax=cbar_ax, orientation='horizontal', label='Temperature [K]')
    

    if (savefig):
        pl.savefig('figs/iipeg_median_idr.png')
        pl.savefig('figs/iipeg_median_idr.pdf', bbox_inches='tight')
    
    #----------------------------------
    fig, ax = pl.subplots(nrows=3, ncols=5, figsize=(12,8))
    for i in range(5):
        pl.axes(ax[0,i])
        hp.orthview(median_step2, nest=True, hold=True, title=f'{i*0.2:3.1f}', cmap=pl.cm.inferno, half_sky=True, rot=(i*360/5.,0,0), min=np.min(pct_step2), max=np.max(pct_step2), cbar=False)
        if (i == 0):
            pl.text(-1.5, 0.9, 'Median', fontsize='large')
        if (i == 2):
            pl.text(-0.3, 1.5, r'N$_\mathrm{obs}$=6', fontsize='large')
        hp.visufunc.graticule()

        pl.axes(ax[1,i])
        hp.orthview(pct_step2[0, :], nest=True, hold=True, title='', cmap=pl.cm.inferno, half_sky=True, rot=(i*360/5.,0,0), min=np.min(pct_step2), max=np.max(pct_step2), cbar=False)
        hp.visufunc.graticule()
        if (i == 0):
            pl.text(-1.5, 0.9, '10%', fontsize='large')

        pl.axes(ax[2,i])
        hp.orthview(pct_step2[1, :], nest=True, hold=True, title='', cmap=pl.cm.inferno, half_sky=True, rot=(i*360/5.,0,0), min=np.min(pct_step2), max=np.max(pct_step2), cbar=False)
        hp.visufunc.graticule()
        if (i == 0):
            pl.text(-1.5, 0.9, '90%', fontsize='large')


    fig.subplots_adjust(bottom=0.1)
    cbar_ax = fig.add_axes([0.35, 0.08, 0.30, 0.02])
    fig.colorbar(pl.cm.ScalarMappable(norm=pl.Normalize(vmin=np.min(median), vmax=np.max(median)), cmap=pl.cm.inferno), cax=cbar_ax, orientation='horizontal', label='Temperature [K]')

    
    if (savefig):
        pl.savefig('figs/iipeg_median_idr_step2.png')
        pl.savefig('figs/iipeg_median_idr_step2.pdf', bbox_inches='tight')

    #----------------------------------
    fig, ax = pl.subplots(nrows=3, ncols=5, figsize=(12,8))
    for i in range(5):
        pl.axes(ax[0,i])
        hp.orthview(median_step4, nest=True, hold=True, title=f'{i*0.2:3.1f}', cmap=pl.cm.inferno, half_sky=True, rot=(i*360/5.,0,0), min=np.min(pct_step4), max=np.max(pct_step4), cbar=False)
        if (i == 0):
            pl.text(-1.5, 0.9, 'Median', fontsize='large')
        if (i == 2):
            pl.text(-0.3, 1.5, r'N$_\mathrm{obs}$=3', fontsize='large')
        hp.visufunc.graticule()

        
        pl.axes(ax[1,i])
        hp.orthview(pct_step4[0, :], nest=True, hold=True, title='', cmap=pl.cm.inferno, half_sky=True, rot=(i*360/5.,0,0), min=np.min(pct_step4), max=np.max(pct_step4), cbar=False)
        hp.visufunc.graticule()
        if (i == 0):
            pl.text(-1.5, 0.9, '10%', fontsize='large')

        pl.axes(ax[2,i])
        hp.orthview(pct_step4[1, :], nest=True, hold=True, title='', cmap=pl.cm.inferno, half_sky=True, rot=(i*360/5.,0,0), min=np.min(pct_step4), max=np.max(pct_step4), cbar=False)
        hp.visufunc.graticule()
        if (i == 0):
            pl.text(-1.5, 0.9, '90%', fontsize='large')
                    


    fig.subplots_adjust(bottom=0.1)
    cbar_ax = fig.add_axes([0.35, 0.08, 0.30, 0.02])
    fig.colorbar(pl.cm.ScalarMappable(norm=pl.Normalize(vmin=np.min(median), vmax=np.max(median)), cmap=pl.cm.inferno), cax=cbar_ax, orientation='horizontal', label='Temperature [K]')

    if (savefig):
        pl.savefig('figs/iipeg_median_idr_step4.png')
        pl.savefig('figs/iipeg_median_idr_step4.pdf', bbox_inches='tight')


if (which == 1):
    #-------------- 12 obs
    fig, ax = pl.subplots(nrows=4, ncols=4, figsize=(12,8), sharex=True, sharey=True)
    for i in range(16):
        tmp = projector.projmap(output_hd[i, :], proj_fun) 
        ax.flat[i].imshow(tmp, cmap=pl.cm.inferno, interpolation='none', origin='lower', extent=[-180, 180, -90, 90])
        ax.flat[i].set_yticks([-90,0,90])
        ax.flat[i].set_xticks([-180,-90,0,90,180])
        
        graticule(ax.flat[i])

    ax.flat[0].set_title(r'N$_\mathrm{obs}$=12')
    fig.subplots_adjust(top=0.85)
    cbar_ax = fig.add_axes([0.35, 0.93, 0.30, 0.02])
    fig.colorbar(pl.cm.ScalarMappable(norm=pl.Normalize(vmin=np.min(output_hd[0:17,:]), vmax=np.max(output_hd[0:17,:])), cmap=pl.cm.inferno), cax=cbar_ax, orientation='horizontal', label='Temperature [K]')

    if (savefig):
        pl.savefig('figs/iipeg_samples.png')
        pl.savefig('figs/iipeg_samples.pdf', bbox_inches='tight')

    #-------------- 6 obs
    fig, ax = pl.subplots(nrows=4, ncols=4, figsize=(12,8), sharex=True, sharey=True)
    for i in range(16):
        tmp = projector.projmap(output_hd_step2[i, :], proj_fun) 
        ax.flat[i].imshow(tmp, cmap=pl.cm.inferno, interpolation='none', origin='lower', extent=[-180, 180, -90, 90])
        ax.flat[i].set_yticks([-90,0,90])
        ax.flat[i].set_xticks([-180,-90,0,90,180])
        
        graticule(ax.flat[i])
    
    ax.flat[0].set_title(r'N$_\mathrm{obs}$=6')
    fig.subplots_adjust(top=0.85)
    cbar_ax = fig.add_axes([0.35, 0.93, 0.30, 0.02])
    fig.colorbar(pl.cm.ScalarMappable(norm=pl.Normalize(vmin=np.min(output_hd_step2[0:17,:]), vmax=np.max(output_hd_step2[0:17,:])), cmap=pl.cm.inferno), cax=cbar_ax, orientation='horizontal', label='Temperature [K]')

    if (savefig):
        pl.savefig('figs/iipeg_samples_step2.png')
        pl.savefig('figs/iipeg_samples_step2.pdf', bbox_inches='tight')

    #-------------- 3 obs
    fig, ax = pl.subplots(nrows=4, ncols=4, figsize=(12,8), sharex=True, sharey=True)
    for i in range(16):
        tmp = projector.projmap(output_hd_step4[i, :], proj_fun) 
        ax.flat[i].imshow(tmp, cmap=pl.cm.inferno, interpolation='none', origin='lower', extent=[-180, 180, -90, 90])
        ax.flat[i].set_yticks([-90,0,90])
        ax.flat[i].set_xticks([-180,-90,0,90,180])
        
        graticule(ax.flat[i])
    
    ax.flat[0].set_title(r'N$_\mathrm{obs}$=3')
    fig.subplots_adjust(top=0.85)
    cbar_ax = fig.add_axes([0.35, 0.93, 0.30, 0.02])
    fig.colorbar(pl.cm.ScalarMappable(norm=pl.Normalize(vmin=np.min(output_hd_step4[0:17,:]), vmax=np.max(output_hd_step4[0:17,:])), cmap=pl.cm.inferno), cax=cbar_ax, orientation='horizontal', label='Temperature [K]')

    if (savefig):
        pl.savefig('figs/iipeg_samples_step4.png')
        pl.savefig('figs/iipeg_samples_step4.pdf', bbox_inches='tight')

if (which == 2):

    origin = [5980, 6000, 6020]
    labels = ['5987.1', '6003.0', '6024.1']

    #-------------- 12 obs
    fig, ax = pl.subplots(nrows=1, ncols=3, figsize=(10,18))
    for region in range(3):
        for i in range(len(phase)):
            ind = len(phase) - 1 - i        
            ax[region].errorbar(wl[region] - origin[region], obs[region][ind, :] + i*0.2, yerr=1e-3, color='C0')
            for j in range(100):
                ax[region].plot(wl[region] - origin[region], stokesi[j][region][ind, :] + i*0.2, color='C1', alpha=0.05)
            ax[region].plot(wl[region] - origin[region], stokesi_median[region][ind, :] + i*0.2, color='C2', alpha=1.0)
            ax[region].plot(wl[region] - origin[region], median_stokesi[region][ind, :] + i*0.2, color='C3', alpha=1.0)
            if (region == 0):
                ax[region].text(6.52, 1.03+i*0.2, f'{phase[ind]/(2.0*np.pi):5.3f}')

    for region in range(3):
        ax[region].set_xlabel(fr'$\lambda+${origin[region]} [$\AA$]')
        ax[region].set_title(fr'Fe I {labels[region]} $\AA$')
    
    if (savefig):
        pl.savefig('figs/iipeg_spectra.png')
        pl.savefig('figs/iipeg_spectra.pdf', bbox_inches='tight')

    #-------------- 6 obs
    fig, ax = pl.subplots(nrows=1, ncols=3, figsize=(10,18))
    for region in range(3):
        for i in range(len(phase_step2)):
            ind = len(phase_step2) - 1 - i        
            ax[region].errorbar(wl[region] - origin[region], obs_step2[region][ind, :] + i*0.2, yerr=1e-3, color='C0')
            for j in range(100):
                ax[region].plot(wl[region] - origin[region], stokesi_step2[j][region][ind, :] + i*0.2, color='C1', alpha=0.05)
            ax[region].plot(wl[region] - origin[region], stokesi_median_step2[region][ind, :] + i*0.2, color='C2', alpha=1.0)
            ax[region].plot(wl[region] - origin[region], median_stokesi_step2[region][ind, :] + i*0.2, color='C3', alpha=1.0)
            if (region == 0):
                ax[region].text(6.52, 1.03+i*0.2, f'{phase_step2[ind]/(2.0*np.pi):5.3f}')

    for region in range(3):
        ax[region].set_xlabel(fr'$\lambda+${origin[region]} [$\AA$]')
        ax[region].set_title(fr'Fe I {labels[region]} $\AA$')

    if (savefig):
        pl.savefig('figs/iipeg_spectra_step2.png')
        pl.savefig('figs/iipeg_spectra_step2.pdf', bbox_inches='tight')

    #-------------- 3 obs
    fig, ax = pl.subplots(nrows=1, ncols=3, figsize=(10,18))
    for region in range(3):
        for i in range(len(phase_step4)):
            ind = len(phase_step4) - 1 - i        
            ax[region].errorbar(wl[region] - origin[region], obs_step4[region][ind, :] + i*0.2, yerr=1e-3, color='C0')
            for j in range(100):
                ax[region].plot(wl[region] - origin[region], stokesi_step4[j][region][ind, :] + i*0.2, color='C1', alpha=0.05)
            ax[region].plot(wl[region] - origin[region], stokesi_median_step2[region][ind, :] + i*0.2, color='C2', alpha=1.0)
            ax[region].plot(wl[region] - origin[region], median_stokesi_step4[region][ind, :] + i*0.2, color='C3', alpha=1.0)
            if (region == 0):
                ax[region].text(6.52, 1.03+i*0.2, f'{phase_step4[ind]/(2.0*np.pi):5.3f}')

    for region in range(3):
        ax[region].set_xlabel(fr'$\lambda+${origin[region]} [$\AA$]')
        ax[region].set_title(fr'Fe I {labels[region]} $\AA$')

    if (savefig):
        pl.savefig('figs/iipeg_spectra_step4.png')
        pl.savefig('figs/iipeg_spectra_step4.pdf', bbox_inches='tight')

    pl.show()    