import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.collections import LineCollection
import matplotlib.gridspec as gridspec

class Plotter:

    def __init__(self,
                 ssps=None,
                 data=None,
                 beta_smp=None,
                 df=None,
                 credible_interval_type='even_tailed',
                 p_in_credible_intervals=np.array([0.5, 0.95]),
                 imshow_aspect='square_axes',
                 kw_mod_plot={},
                 kw_data_plot={},
                 kw_true_plot={},
                 kw_post1d_plot={},
                 kw_post2d_plot={}):
        # set input values
        self.ssps = ssps
        self.data = data
        self.beta_smp = beta_smp
        self.df = df
        # set bayesian credible interval parameters
        if credible_interval_type=='even_tailed':
            self.get_bcis = 'even_tailed_bcis'
        elif credible_interval_type=='highest_density':
            self.get_bcis = 'highest_density_bcis'
        else:
            raise ValueError('Unknown credible_interval_type')
        self.set_credible_interval_levels(p_in_credible_intervals)
        # set/update plot formatting keywords
        self.kw_mod_plot = {'cmap':plt.cm.viridis}
        self.kw_mod_plot.update(kw_mod_plot)
        self.kw_data_plot = {'color':'C0', 'marker':'.', 'ls':'None', 'ms':3}
        self.kw_data_plot.update(kw_data_plot)
        self.kw_true_plot = {'color':'k', 'ls':'-'}
        self.kw_true_plot.update(kw_true_plot)
        self.kw_post1d_plot = {'color':'C1', 'alpha':0.3}
        self.kw_post1d_plot.update(kw_post1d_plot)
        self.kw_post2d_plot={'cmap':plt.cm.magma}
        self.kw_post2d_plot.update(kw_post2d_plot)
        # set aspect ratio for calls to imshow
        self.kw_resid_plot = self.kw_post2d_plot.copy()
        self.kw_resid_plot.update({'cmap':plt.cm.Greys_r,
                                   'norm':colors.LogNorm()})
        self.configure_imshow(imshow_aspect=imshow_aspect)

    def set_credible_interval_levels(self, p_in_credible_intervals):
        self.p_in = np.array(p_in_credible_intervals)
        self.n_bcis = len(p_in_credible_intervals)

    def configure_imshow(self,
                         imshow_aspect='square_axes',
                         interpolation='none'):
        par_dims = self.ssps.par_dims
        if imshow_aspect=='square_pixels':
            # aspect ratio for square pixels in imshow calls
            if self.ssps is not None:
                if self.ssps.npars==2:
                    plim = self.ssps.par_lims
                    plim_rng = np.array(plim)[:,1] - np.array(plim)[:,0]
                    plim_ratio = 1.*plim_rng[1]/plim_rng[0]
                    pdim_ratio = 1.*par_dims[1]/par_dims[0]
                    aspect = plim_ratio/pdim_ratio
                elif self.ssps.npars==1:
                    plim = self.ssps.par_lims
                    plim_rng = plim[0][1] - plim[0][0]
                    plim_ratio = 1./plim_rng
                    pdim_ratio = 1./par_dims[0]
                    aspect = pdim_ratio/plim_ratio
                else:
                    pass
        elif imshow_aspect=='square_axes':
            # aspect ratio for square pixels in imshow calls
            if self.ssps is not None:
                if self.ssps.npars==2:
                    plim = self.ssps.par_lims
                    plim_rng = np.array(plim)[:,1] - np.array(plim)[:,0]
                    plim_ratio = 1.*plim_rng[1]/plim_rng[0]
                    aspect = plim_ratio
                elif self.ssps.npars==1:
                    plim = self.ssps.par_lims
                    plim_rng = plim[0][1] - plim[0][0]
                    plim_ratio = 1./plim_rng
                    aspect = plim_ratio
                else:
                    pass
        elif imshow_aspect=='auto':
            # auto: fill the axes
            aspect = 'auto'
        elif isinstance(imshow_aspect, (int, float)):
            aspect = float(imshow_aspect)
        else:
            raise ValueError('unknown imshow_aspect')
        self.imshow_aspect = aspect
        self.kw_post2d_plot.update({'aspect':aspect})
        self.kw_mod_plot.update({'aspect':aspect})
        self.kw_resid_plot.update({'aspect':aspect})
        # set imshow extent
        if self.ssps.npars==1:
            plim = self.ssps.par_lims
            extent = np.concatenate((plim[0], [0,1]))
        elif self.ssps.npars==2:
            plim = self.ssps.par_lims
            extent = np.concatenate(plim[::-1])
        else:
            pass
        self.kw_post2d_plot.update({'extent':extent})
        self.kw_mod_plot.update({'extent':extent})
        self.kw_resid_plot.update({'extent':extent})
        # set interpolation
        self.kw_post2d_plot.update({'interpolation':interpolation})
        self.kw_mod_plot.update({'interpolation':interpolation})
        self.kw_resid_plot.update({'interpolation':interpolation})
        # set origin
        self.kw_post2d_plot.update({'origin':'lower'})
        self.kw_mod_plot.update({'origin':'lower'})
        self.kw_resid_plot.update({'origin':'lower'})

    def plot_ssps(self, y_offset=0, lmd_rng=None):
        # get colors
        models = self.ssps.X.T
        v = [np.array([self.ssps.lmd, models[i]+i*y_offset]).T for i in range(self.ssps._p)]
        cols = np.linspace(0, 1, self.ssps._p)
        if self.ssps.npars==1:
            mod_key_arr = np.array([cols])
        elif self.ssps.npars==2:
            mod_key_arr = self.ssps.reshape_beta(cols)
        cmap = self.kw_mod_plot.pop('cmap')
        lc = LineCollection(np.array(v),
                            colors=cmap(cols))
        self.kw_mod_plot.update({'cmap':cmap})

        # fig, ax = plt.subplots(1, 2, figsize=(7, 3.5))
        fig = plt.figure(figsize=(7, 3.5))
        spec = gridspec.GridSpec(ncols=3, nrows=1, figure=fig)
        ax0 = fig.add_subplot(spec[:2])
        ax1 = fig.add_subplot(spec[2])
        ax = [ax0, ax1]

        # plot models
        ax[0].add_collection(lc)
        ax[1].imshow(mod_key_arr,
                     **self.kw_mod_plot)
        # format ax0
        minx, maxx = np.min(self.ssps.lmd), np.max(self.ssps.lmd)
        miny, maxy = np.min(self.ssps.X), np.max(self.ssps.X)
        ax[0].set_xlim(minx, maxx)
        ax[0].set_ylim(miny, maxy)
        ax[0].set_ylabel('Flux')
        ax[0].set_xlabel('$\lambda$ [angstrom]')
        # format ax1
        if self.ssps.npars==2:
            ax[1].set_xlabel(self.ssps.par_pltsym[1])
            ax[1].set_ylabel(self.ssps.par_pltsym[0])
        else:
            ax[1].set_xlabel(self.ssps.par_pltsym[0])
            ax[1].get_yaxis().set_visible(False)

        ax[1].set_xticks(self.ssps.img_t_ticks)
        ax[1].set_xticklabels(self.ssps.t_ticks)
        ax[1].set_yticks(self.ssps.img_z_ticks)
        ax[1].set_yticklabels(self.ssps.z_ticks)
        if lmd_rng is not None:
            ax[0].set_xlim(lmd_rng)
        fig.tight_layout()
        return

    def plot_data(self, plot_true_ybar=True):
        fig, ax = plt.subplots(1, 1, figsize=(7,3.5))
        # plot data
        ax.plot(self.data.lmd, self.data._y, label='$_y$', **self.kw_data_plot)
        if plot_true_ybar:
            if hasattr(self.data, 'ybar'):
                ax.plot(self.data.lmd,
                        self.data.ybar,
                        label='true $\\bar{_y}$',
                        **self.kw_true_plot)
        # format
        ax.legend()
        ax.set_ylabel('Data')
        ax.set_xlabel('$\lambda$ [angstrom]')
        fig.tight_layout()
        fig.subplots_adjust(hspace=0)
        return

    def plot_df(self,
                ax=None,
                view='median',    # median / mode / percentile / map / mad / true
                percentile=None,
                clim=None,
                neg_fill_value='min',
                label_axes=True,
                colorbar=False,
                lognorm=True):
        if self.ssps.npars!=2:
            raise ValueError('Model not 2D')
        if ax is None:
            ax = plt.gca()

        if view=='median':
            img = self.beta_smp.get_median()
        elif view=='mode':
            img = self.beta_smp.get_halfsample_mode()
        elif view=='percentile':
            if percentile is None:
                raise ValueError('Set percentile value 0 < ... < 100')
            img = self.beta_smp.get_percentile(percentile)
        elif view=='map':
            img = self.beta_smp.get_MAP()
        elif view=='mad':
            img = self.beta_smp.get_mad_from_mode()
        elif view=='true':
            img = self.df.beta
        else:
            raise ValueError('Unknown view')

        # fix negatives
        if neg_fill_value=='min':
            neg_fill_value = np.min(img[img>0.])
        img[img<0] = neg_fill_value

        # color limits
        if clim is None:
            vmin, vmax = np.min(img), np.max(img)
        else:
            vmin, vmax = clim

        # plot
        img = self.ssps.reshape_beta(img)
        if lognorm:
            img = ax.imshow(img,
                            **self.kw_post2d_plot,
                            norm=colors.LogNorm(vmin=vmin, vmax=vmax))
        else:
            img = ax.imshow(img,
                            **self.kw_post2d_plot,
                            vmin=vmin,
                            vmax=vmax)

        if label_axes:
            ax.set_xlabel(self.ssps.par_pltsym[1])
            ax.set_ylabel(self.ssps.par_pltsym[0])
            ax.set_xticks(self.ssps.img_t_ticks)
            ax.set_xticklabels(self.ssps.t_ticks)
            ax.set_yticks(self.ssps.img_z_ticks)
            ax.set_yticklabels(self.ssps.z_ticks)
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        if colorbar:
            plt.colorbar(img)

        return img