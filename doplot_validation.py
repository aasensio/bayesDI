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

with open('validation.pk', 'rb') as handle:
    output, target, sini, velocity, nangles, stokesi, stokes_median, stokes_ae, obs, phase = pickle.load(handle)

output /= 1000.0
target /= 1000.0

savefig = True

median = np.median(output, axis=1)
mad = np.std(output, axis=1)
pct = np.percentile(output, [10, 90], axis=1)
idr = pct[1, :] - pct[0, :]

proj_fun = lambda x,y,z : hp.vec2pix(2**4, x, y, z, nest=True)
projector = hp.projector.CartesianProj() 

which = 3

if (which == 0):
    
    ind = 113
    fig, ax = pl.subplots(nrows=4, ncols=5, figsize=(11,6), sharex=True, sharey=True)

    target_vmin = np.min(target[ind, :])
    target_vmax = np.max(target[ind, :])    

    for i in range(16):
        tmp = projector.projmap(output[ind, i, :], proj_fun) 
        ax.flat[i].imshow(tmp, cmap=pl.cm.inferno, interpolation='none', origin='lower', 
            extent=[-180, 180, -90, 90], vmin=target_vmin, vmax=target_vmax)
        ax.flat[i].set_yticks([-90,0,90])
        ax.flat[i].set_xticks([-180,-90,0,90,180])            

    tmp = projector.projmap(median[ind, :], proj_fun)     
    ax.flat[-4].imshow(tmp, cmap=pl.cm.inferno, interpolation='none', origin='lower', 
        extent=[-180, 180, -90, 90], vmin=target_vmin, vmax=target_vmax)
    ax.flat[-4].text(45, -70, f'Median',fontweight='demibold')

    tmp = projector.projmap(pct[0, ind, :], proj_fun)     
    ax.flat[-3].imshow(tmp, cmap=pl.cm.inferno, interpolation='none', origin='lower', 
        extent=[-180, 180, -90, 90], vmin=target_vmin, vmax=target_vmax)
    ax.flat[-3].text(100, -70, f'10%',fontweight='demibold', color='white')

    tmp = projector.projmap(pct[1, ind, :], proj_fun)     
    ax.flat[-2].imshow(tmp, cmap=pl.cm.inferno, interpolation='none', origin='lower', 
        extent=[-180, 180, -90, 90], vmin=target_vmin, vmax=target_vmax)
    ax.flat[-2].text(100, -70, f'90%',fontweight='demibold')

    tmp = projector.projmap(target[ind, :], proj_fun)     
    ax.flat[-1].imshow(tmp, cmap=pl.cm.inferno, interpolation='none', origin='lower', 
        extent=[-180, 180, -90, 90], vmin=target_vmin, vmax=target_vmax)
    ax.flat[-1].text(50, -70, f'Target',fontweight='demibold')
    
    fig.subplots_adjust(left=0.08, right=0.95, top=0.84)
    cbar_ax = fig.add_axes([0.35, 0.93, 0.30, 0.02])
    fig.colorbar(pl.cm.ScalarMappable(norm=pl.Normalize(vmin=target_vmin, vmax=target_vmax), cmap=pl.cm.inferno), cax=cbar_ax, orientation='horizontal', label='T [kK]')

    fig.supxlabel('Longitude [deg]')
    fig.supylabel('Latitude [deg]')
    
    # if (savefig):
        # pl.savefig('figs/validation_samples.png')
        # pl.savefig('figs/validation_samples.pdf', bbox_inches='tight')


if (which == 1):
    loop = 0

    for j in range(3):
        fig, ax = pl.subplots(nrows=5, ncols=6, figsize=(17,8), sharex=True, sharey=True)        

        target_vmin = np.min(target[6*j:6*j+6, :])
        target_vmax = np.max(target[6*j:6*j+6, :])

        idr_vmin = np.min(idr[6*j:6*j+6, :])
        idr_vmax = np.max(idr[6*j:6*j+6, :])
        
        for i in range(6):
            tmp = projector.projmap(target[loop, :], proj_fun) 
            ax[0,i].imshow(tmp, cmap=pl.cm.inferno, interpolation='none', origin='lower', 
                extent=[-180, 180, -90, 90], vmin=target_vmin, vmax=target_vmax)
            ax[0,i].set_yticks([-90,0,90])
            ax[0,i].set_xticks([-180,-90,0,90,180])
            ax[0,i].set_title(rf'i={np.arcsin(sini[loop])*180/np.pi:5.2f}')
            if (i == 0):
                ax[0,i].set_ylabel(f'Target')

            tmp = projector.projmap(median[loop, :], proj_fun) 
            ax[1,i].imshow(tmp, cmap=pl.cm.inferno, interpolation='none', origin='lower', 
                extent=[-180, 180, -90, 90], vmin=target_vmin, vmax=target_vmax)
            ax[1,i].set_yticks([-90,0,90])
            ax[1,i].set_xticks([-180,-90,0,90,180])
            ax[1,i].set_title(f'N$_p$={nangles[loop]}')
            if (i == 0):
                ax[1,i].set_ylabel(f'Median')
            

            tmp = projector.projmap(idr[loop, :], proj_fun) 
            ax[2,i].imshow(tmp, cmap=pl.cm.inferno, interpolation='none', origin='lower', 
                extent=[-180, 180, -90, 90], vmin=idr_vmin, vmax=idr_vmax)
            ax[2,i].set_yticks([-90,0,90])
            ax[2,i].set_xticks([-180,-90,0,90,180])
            ax[2,i].set_title(rf'v sini={velocity[loop]*80*sini[loop]:5.2f}')
            if (i == 0):
                ax[2,i].set_ylabel(f'IDR')            

            tmp = projector.projmap(pct[0, loop, :], proj_fun) 
            ax[3,i].imshow(tmp, cmap=pl.cm.inferno, interpolation='none', origin='lower', 
                extent=[-180, 180, -90, 90], vmin=target_vmin, vmax=target_vmax)
            ax[3,i].set_yticks([-90,0,90])
            ax[3,i].set_xticks([-180,-90,0,90,180])            
            if (i == 0):
                ax[3,i].set_ylabel(f'10%')

            tmp = projector.projmap(pct[1, loop, :], proj_fun) 
            ax[4,i].imshow(tmp, cmap=pl.cm.inferno, interpolation='none', origin='lower', 
                extent=[-180, 180, -90, 90], vmin=target_vmin, vmax=target_vmax)
            ax[4,i].set_yticks([-90,0,90])
            ax[4,i].set_xticks([-180,-90,0,90,180])            
            if (i == 0):
                ax[4,i].set_ylabel(f'90%')
            
            for k in range(5):
                graticule(ax[k, i])

            if (j <= 1):
                ax[0,i].xaxis.set_visible(False)
                ax[1,i].xaxis.set_visible(False)
                ax[2,i].xaxis.set_visible(False)
                ax[3,i].xaxis.set_visible(False)
                ax[4,i].xaxis.set_visible(False)

            loop += 1
                
        fig.subplots_adjust(left=0.07, right=0.90, top=0.9)

        step = 0.164
        size = 0.125

        cbar_ax = fig.add_axes([0.91, 0.77, 0.01, size])
        fig.colorbar(pl.cm.ScalarMappable(norm=pl.Normalize(vmin=target_vmin, vmax=target_vmax), cmap=pl.cm.inferno), cax=cbar_ax, orientation='vertical', label='T [kK]')

        cbar_ax = fig.add_axes([0.91, 0.77 - step, 0.01, size])
        fig.colorbar(pl.cm.ScalarMappable(norm=pl.Normalize(vmin=target_vmin, vmax=target_vmax), cmap=pl.cm.inferno), cax=cbar_ax, orientation='vertical', label='T [kK]')

        cbar_ax = fig.add_axes([0.91, 0.77 - 2*step, 0.01, size])
        fig.colorbar(pl.cm.ScalarMappable(norm=pl.Normalize(vmin=idr_vmin, vmax=idr_vmax), cmap=pl.cm.inferno), cax=cbar_ax, orientation='vertical', label='T [kK]')

        cbar_ax = fig.add_axes([0.91, 0.77 - 3*step, 0.01, size])
        fig.colorbar(pl.cm.ScalarMappable(norm=pl.Normalize(vmin=target_vmin, vmax=target_vmax), cmap=pl.cm.inferno), cax=cbar_ax, orientation='vertical', label='T [kK]')

        cbar_ax = fig.add_axes([0.91, 0.77 - 4*step, 0.01, size])
        fig.colorbar(pl.cm.ScalarMappable(norm=pl.Normalize(vmin=target_vmin, vmax=target_vmax), cmap=pl.cm.inferno), cax=cbar_ax, orientation='vertical', label='T [kK]')

        # if (j == 1):
            # fig.supylabel('Latitude [deg]')

        ax[2,0].text(-0.4, 0.18, 'Latitude [deg]', transform=ax[2,0].transAxes, rotation='vertical')

        if (j == 2):
            fig.supxlabel('Longitude [deg]')
        if (savefig):
            pl.savefig(f'figs/validation_cases_new_{j}.png')
            pl.savefig(f'figs/validation_cases_new_{j}.pdf', bbox_inches='tight')

    pl.show()    

if (which == 2):
    loop = 100

    target_vmin = np.min(target[0:6, :])
    target_vmax = np.max(target[0:6, :])

    fig, ax = pl.subplots(nrows=2, ncols=6, figsize=(17,5), sharex=True, sharey=True)        
        
    for i in range(6):
        tmp = projector.projmap(target[loop, :], proj_fun) 
        ax[0,i].imshow(tmp, cmap=pl.cm.inferno, interpolation='none', origin='lower', 
            extent=[-180, 180, -90, 90], vmin=target_vmin, vmax=target_vmax)
        ax[0,i].set_yticks([-90,0,90])
        ax[0,i].set_xticks([-180,-90,0,90,180])
        ax[0,i].set_title(rf'i={np.arcsin(sini[loop])*180/np.pi:5.2f}')
        if (i == 0):
            ax[0,i].set_ylabel(f'Target')

        tmp = projector.projmap(median[loop, :], proj_fun) 
        tmp_alpha = projector.projmap(idr[loop, :], proj_fun) 
        alpha = pl.Normalize(vmin=np.min(idr[loop, :]), vmax=np.max(idr[loop, :]))
        
        ax[1,i].imshow(tmp, cmap=pl.cm.inferno, interpolation='none', origin='lower', 
            extent=[-180, 180, -90, 90], vmin=target_vmin, vmax=target_vmax, alpha=1.0-alpha(tmp_alpha))
        ax[1,i].set_yticks([-90,0,90])
        ax[1,i].set_xticks([-180,-90,0,90,180])
        ax[1,i].set_title(f'N$_p$={nangles[loop]}')
        if (i == 0):
            ax[1,i].set_ylabel(f'Median')
                
        graticule(ax[0,i])
        graticule(ax[1,i])
        
        ax[0,i].xaxis.set_visible(False)
        ax[1,i].xaxis.set_visible(False)
        
        loop += 1
            
    # fig.subplots_adjust(left=0.07, right=0.90, top=0.9)
    # cbar_ax = fig.add_axes([0.91, 0.68, 0.01, 0.21])
    # fig.colorbar(pl.cm.ScalarMappable(norm=pl.Normalize(vmin=target_vmin, vmax=target_vmax), cmap=pl.cm.inferno), cax=cbar_ax, orientation='vertical', label='T [K]')

    # cbar_ax = fig.add_axes([0.91, 0.40, 0.01, 0.21])
    # fig.colorbar(pl.cm.ScalarMappable(norm=pl.Normalize(vmin=target_vmin, vmax=target_vmax), cmap=pl.cm.inferno), cax=cbar_ax, orientation='vertical', label='T [K]')

    # cbar_ax = fig.add_axes([0.91, 0.12, 0.01, 0.21])
    # fig.colorbar(pl.cm.ScalarMappable(norm=pl.Normalize(vmin=idr_vmin, vmax=idr_vmax), cmap=pl.cm.inferno), cax=cbar_ax, orientation='vertical', label='T [K]')

    # if (j == 1):
    #     fig.supylabel('Latitude [deg]')
    # if (j == 2):
    #     fig.supxlabel('Longitude [deg]')
    # if (savefig):
        # pl.savefig(f'figs/validation_cases_{j}.png')
        # pl.savefig(f'figs/validation_cases_{j}.pdf')

    pl.show()    

if (which == 3):
    origin = [5980, 6000, 6020]
    labels = ['5987.1', '6003.0', '6024.1']    

    wl = np.array([5986.54731, 5986.58131, 5986.61631, 5986.65031, 5986.68531,
        5986.71931, 5986.75431, 5986.78831, 5986.82431, 5986.85731,
        5986.89331, 5986.92631, 5986.96231, 5986.99631, 5987.02931,
        5987.06531, 5987.09831, 5987.13431, 5987.16731, 5987.20331,
        5987.23531, 5987.27231, 5987.30531, 5987.34131, 5987.37431,
        5987.41031, 5987.44431, 5987.47931, 5987.51331, 5987.54831,
        5987.58231, 6001.98222, 6002.01522, 6002.04822, 6002.08422,
        6002.11722, 6002.15122, 6002.18422, 6002.21722, 6002.25222,
        6002.28622, 6002.31822, 6002.35222, 6002.38622, 6002.41922,
        6002.45522, 6002.48822, 6002.52122, 6002.55422, 6002.59022,
        6002.62322, 6002.65622, 6002.69022, 6002.72322, 6002.75822,
        6002.79222, 6002.82522, 6002.85922, 6002.89222, 6002.92522,
        6002.96022, 6002.99422, 6003.02622, 6003.06022, 6003.09422,
        6003.12922, 6003.16222, 6003.19622, 6003.22922, 6003.26222,
        6003.29722, 6003.33122, 6003.36422, 6003.39822, 6003.43122,
        6003.46422, 6003.49922, 6003.53322, 6003.56722, 6003.59922,
        6023.48409, 6023.51409, 6023.54709, 6023.58009, 6023.61309,
        6023.64409, 6023.67709, 6023.71009, 6023.74309, 6023.77409,
        6023.80709, 6023.84009, 6023.87309, 6023.90609, 6023.93709,
        6023.97009, 6024.00309, 6024.03609, 6024.06909, 6024.10009,
        6024.13309, 6024.16609, 6024.19909, 6024.23209, 6024.26309,
        6024.29609, 6024.33009, 6024.36209, 6024.39309, 6024.42609,
        6024.46009, 6024.49209, 6024.52509, 6024.55609, 6024.59009,
        6024.62309, 6024.65509])

    left_all = [0, 31, 80]
    right_all = [31, 80, len(wl)]

    residuals = [None] * 3

    #-------------- 12 obs
    # fig, ax = pl.subplots(nrows=1, ncols=3, figsize=(10,18))
    # for region in range(3):
    #     left = left_all[region]
    #     right = right_all[region]

    #     residual = np.zeros((9, 100, len(wl[left:right])))

    #     for i in range(9):            
    #         ax[region].errorbar(wl[left:right] - origin[region], obs[i, left:right] + i*0.13, yerr=1e-3, color='C0')            
            
    #         tmp = []
    #         for j in range(100):
    #             ax[region].plot(wl[left:right] - origin[region], stokesi[j][region][i, :] + i*0.13, color='C1', alpha=0.05)
    #             tmp.append(stokesi[j][region][i, :])
    #             residual[i, j, :] = stokesi[j][region][i, :] - obs[i, left:right]
            
    #         tmp = np.vstack(tmp)



    #         # ax[region].plot(wl[left:right] - origin[region], stokes_median[region][i, :] + i*0.12, color='C2', alpha=1.0)
    #         # ax[region].plot(wl[left:right] - origin[region], stokes_ae[region][i, :] + i*0.12, color='C4', alpha=1.0)
    #         # ax[region].plot(wl[left:right] - origin[region], np.median(tmp, axis=0) + i*0.12, color='C3', alpha=1.0)
    #         if (region == 0):
    #             ax[region].text(6.52, 1.0+i*0.12, f'{phase[i]/(2.0*np.pi):5.3f}')
    #     residuals[region] = residual

    # for region in range(3):
    #     ax[region].set_xlabel(fr'$\lambda+${origin[region]} [$\AA$]')
    #     ax[region].set_title(fr'Fe I {labels[region]} $\AA$')


    fig, ax = pl.subplots(nrows=1, ncols=3, figsize=(10,18))
    for region in range(3):
        left = left_all[region]
        right = right_all[region]

        residual = np.zeros((9, 100, len(wl[left:right])))

        for i in range(9):            

            dy = i * 0.08
            
            for j in range(100):
                ax[region].plot(wl[left:right] - origin[region], stokesi[j][region][i, :] - obs[i, left:right]+ dy, color='C1', alpha=0.02)
                residual[i, j, :] = stokesi[j][region][i, :] - obs[i, left:right]
                # tmp.append(stokesi[j][region][i, :])

            ax[region].errorbar(wl[right-1] - origin[region], dy, yerr=2e-2, color='C0', capsize=3)
            ax[region].errorbar(wl[right-1] - origin[region], dy, yerr=1e-2, color='C2', capsize=3)
            ax[region].errorbar(wl[right-1] - origin[region], dy, yerr=5e-3, color='C3', capsize=3)
            

            pct = np.percentile(residual[i, :, :], [50.0 - 68./2.0, 50.0+68/2.0], axis=0)
            ax[region].plot(wl[left:right] - origin[region], pct[0] + dy, alpha=0.3, color='C4')
            ax[region].plot(wl[left:right] - origin[region], pct[1] + dy, alpha=0.3, color='C4')
            
            # tmp = np.vstack(tmp)

            # ax[region].plot(wl[left:right] - origin[region], stokes_median[region][i, :] + i*0.12, color='C2', alpha=1.0)
            # ax[region].plot(wl[left:right] - origin[region], np.median(tmp, axis=0) + i*0.12, color='C3', alpha=1.0)
            if (region == 0):
                ax[region].text(6.52, 0.02+dy, f'{phase[i]/(2.0*np.pi):5.3f}')

    for region in range(3):
        ax[region].set_xlabel(fr'$\lambda+${origin[region]} [$\AA$]')
        ax[region].set_title(fr'Fe I {labels[region]} $\AA$')

    pl.savefig('figs/spectra_validation.pdf', bbox_inches='tight')