# #############################
#
#
# Cheng Song   (songcheng@snu.ac.kr)
# Modified from Jennifer Dreiling   (dreiling@gfz-potsdam.de)
#
# #############################

import zmq
import time
import glob
import logging
import numpy as np
import os.path as op

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams["axes.axisbelow"] = False
from matplotlib.widgets import Button
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from seabinv.utils import SerializingContext
from seabinv import Model
from seabinv import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('BayWatch')
plt.ion()

class BayWatcher(object):

    def __init__(self, configfile, capacity=100, address='127.0.0.1',
                 port=5556, save_plots=None):
        # set up socket
        sock_addr = 'tcp://%s:%d' % (address, port)
        logger.info('Connecting to %s' % sock_addr)
        context = SerializingContext()
        self.socket = context.socket(zmq.SUB)
        self.socket.connect(sock_addr)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, u'')
        if save_plots and not op.exists(op.dirname(save_plots)):
            raise OSError('%s does not exist' % op.dirname(save_plots))
        self.save_plots = save_plots

        data_pars = utils.read_config(configfile)
        defaults = utils.get_path('defaults.ini')
        self.priors, self.initparams = utils.load_params(defaults)
        self.priors.update(data_pars['priors'])
        self.initparams.update(data_pars['initparams'])
        self.nchains = self.initparams['nchains']
        self.refmodel = data_pars.get('refmodel', dict())

        self.capacity = capacity
        self.modellength = int(self.priors['segments'][1]) * 2
        self.rsl_step = np.ones((capacity, self.modellength)) * np.nan
        self.age_step = np.ones((capacity, self.modellength)) * np.nan
        self.likes = np.ones((capacity)) * np.nan
        self.modelmatrix = np.ones((capacity, self. modellength)) * np.nan

        # colors only for inversion targets - modeled data
        # self.colors = ['purple', 'green', 'orange' 'red', 'brown', 'blue']
        self.colors = ['teal', 'lightcoral', 'saddlebrown', 'magenta', 'royalblue']

        self.targets = data_pars['targets']
        self.targetrefs = [target.ref for target in self.targets]
        self.ntargets = len(self.targets)

        self.noises = np.ones((self.capacity, self.ntargets*2)) * np.nan
        self.init_style_dicts()
        self.init_plot()
        self.init_arrays()
        self.breakloop = np.zeros(self.nchains)

    def init_style_dicts(self):
        obsrsl = {'color': 'k', 'alpha': 0.5, 'lw': 1.2,
                  'marker': '*', 'ms': 3, 'elinewidth': 1}
        noise = {'lw': 1.2, 'marker': 'o', 'ms': 0.2, 'ls': '-'}
        self.mod = {'lw': 1.2, 'ls': '-'}

        self.axdict = {'rsl': {'ax': 3, 'style': obsrsl},
                       'noise': {'ax': 4, 'style': noise}}

    def init_plot(self):
        self.fig, self.axes = plt.subplots(figsize=(8, 7))
        self.fig.subplots_adjust(hspace=0.9, wspace=0.1)
        self.fig.canvas.manager.set_window_title('BayWatch. Inversion live-stream.')
        ax1 = plt.subplot2grid((10, 8), (0, 0), rowspan=10, colspan=3)  # vel-dep
        ax2 = plt.subplot2grid((10, 8), (0, 4), rowspan=2, colspan=4)  # like
        ax3 = plt.subplot2grid((10, 8), (4, 4), rowspan=2, colspan=4)  # rf
        ax4 = plt.subplot2grid((10, 8), (6, 4), rowspan=2, colspan=4)  # disp
        ax5 = plt.subplot2grid((10, 8), (8, 4), rowspan=2, colspan=4)  # rayleigh wave ellipticity
        ax6 = plt.subplot2grid((10, 8), (2, 4), rowspan=2, colspan=4)  # noise

        self.axes = [ax1, ax3, ax4, ax5, ax2, ax6]
        # -----------------------------------
        # plot 1: age-rsl model
        self.modelline, = self.axes[0].plot(np.nan, np.nan, color='k', lw=0.7)
        colors = np.arange(self.capacity)
        segments = [np.column_stack([x, y])
                    for x, y in zip(self.age_step, self.rsl_step)]

        lc = LineCollection(segments, cmap='plasma_r')
        lc.set_array(np.asarray(colors))

        self.modelcollection = self.axes[0].add_collection(lc)
        self.modelcollection.set_linewidths(0.7)

        self.axes[0].set_xlim(self.priors['age'])
        self.axes[0].set_ylim(self.priors['rsl'])
        self.axes[0].invert_yaxis()
        self.axes[0].set_xlabel('Age (year)')
        self.axes[0].set_ylabel('RSL (m)')
        self.axes[0].grid(color='gray', ls=':')

        # # plot 1, vpvs
        # self.vpvsline = self.axes[6].axvline(np.nan, color='k', lw=1.2)

        # a = np.repeat(([0, 1], ), self.capacity, axis=0)
        # b = np.array([[v]*2 for v in self.vpvss])
        # segments = [np.column_stack([x, y]) for x, y in zip(b, a)]
        # lc = LineCollection(segments, cmap='plasma_r')
        # lc.set_array(np.asarray(colors))

        # self.vpvscollection = self.axes[6].add_collection(lc)
        # self.vpvscollection.set_linewidths(0.7)

        # if type(self.priors['vpvs']) in [tuple, list, np.array]:
        #     self.axes[6].set_xlim(self.priors['vpvs'])
        #     self.axes[6].set_title('Vp/Vs', fontsize=10)
        #     self.axes[6].tick_params(axis="x", direction="in", pad=-15)

        # elif type(self.priors['vpvs']) == float:
        #     self.axes[6].set_xlim(self.priors['vpvs']-0.2, self.priors['vpvs']-0.1)
        #     self.axes[6].text(0.5, 0.5, 'Vp/Vs: %.2f' % self.priors['vpvs'],
        #                       horizontalalignment='center',
        #                       verticalalignment='center',
        #                       transform=self.axes[6].transAxes,
        #                       fontsize=11)
        #     self.axes[6].set_xticks([])

        # self.axes[6].set_ylim([0, 1])
        # self.axes[6].set_yticks([])

        # -----------------------------------
        # plot2, plot3, plot4: modeled and observed data
        # self.axes[1].yaxis.tick_right()
        # self.axes[2].yaxis.tick_right()
        self.axes[3].set_ylabel('RSL (m)')
        self.axes[3].set_xlabel('Age (year)')

        self.axes[3].spines['right'].set_visible(False)
        self.axes[3].spines['top'].set_visible(False)
        self.axes[3].set_xticklabels([])
        self.axes[3].set_yticklabels([])

        # plot obsdata
        for i, target in enumerate(self.targets):
            x, y, yerr = target.obsdata.x, target.obsdata.y, target.obsdata.yerr
            ref = target.noiseref

            if ref not in ['rsl']:
                # 'By default, user targets are plotted in (upper) swd plot.'
                ref = 'swd'
            idx = self.axdict[ref]['ax']
            style = self.axdict[ref]['style']

            self.axes[idx].errorbar(x, y, yerr=yerr, **style)

        # initiate mod data
        self.targetlines = []
        for i, target in enumerate(self.targets):
            x = target.obsdata.x
            color = self.colors[i]
            ref = target.noiseref
            label = target.ref
            idx = self.axdict[ref]['ax']

            line, = self.axes[idx].plot(
                x, np.ones(x.size)*np.nan, color=color, label=label, **self.mod)
            self.targetlines.append(line)

        hand1, lab1 = self.axes[1].get_legend_handles_labels()
        hand2, lab2 = self.axes[2].get_legend_handles_labels()
        hand3, lab3 = self.axes[3].get_legend_handles_labels()
        handles = hand1 + hand2 + hand3
        labels = lab1 + lab2 + lab3
        self.axes[3].legend(handles, labels, loc='upper left', fancybox=True,
                            bbox_to_anchor=(0, -0.3), frameon=True, ncol=3)

        # grid lines for receiver functions
        self.axes[2].axhline(0, color='k', ls='--', lw=0.5)
        self.axes[2].axvline(0, color='k', ls='--', lw=0.5)

        # plot 4: likelihood
        self.axes[4].set_ylabel('Likelihood')
        self.axes[4].yaxis.set_label_position("right")
        self.likeline, = self.axes[4].plot(
            np.arange(self.capacity), np.ones(self.capacity)*np.nan,
            color='darkblue', lw=1.2, marker='o', ms=0.2, ls='-')

        self.axes[4].set_xlim([0, self.capacity])
        self.axes[4].grid(color='gray', ls=':')
        self.axes[4].set_xticklabels([], visible=False)

        # plot 5: noise
        self.axes[5].set_ylabel('Sigma')
        self.axes[5].yaxis.set_label_position("right")
        style = self.axdict['noise']['style']

        self.sigmalines = []
        for i, ref in enumerate(self.targetrefs):
            color = self.colors[i]
            noiseline, = self.axes[5].plot(
                np.arange(self.capacity),
                np.ones(self.capacity)*np.nan, color=color, **style)
            self.sigmalines.append(noiseline)

        self.axes[5].set_xlim([0, self.capacity])
        self.axes[5].grid(color='gray', ls=':')
        self.axes[5].set_xticklabels([], visible=False)

        self.axes[4].spines['top'].set_visible(False)
        self.axes[4].spines['right'].set_visible(False)
        self.axes[5].spines['top'].set_visible(False)
        self.axes[5].spines['right'].set_visible(False)

        # reference model
        x_age, y_rsl = self.refmodel.get('model', ([np.nan], [np.nan]))
        noise = self.refmodel.get('noise', [np.nan, np.nan])[1::2]
        explike = self.refmodel.get('explike', np.nan)

        self.axes[0].plot(x_age, y_rsl, color='k', ls=':')
        self.axes[4].axhline(explike, color='darkblue', ls=':')

        for i, sigma in enumerate(noise):
            self.axes[5].axhline(noise[i], color=self.colors[i], ls=':')

        # buttons
        axprev = plt.axes([0.69, 0.91, 0.08, 0.060])
        axnext = plt.axes([0.80, 0.91, 0.08, 0.060])

        self.bnext = Button(axnext, '>', color='white', hovercolor='lightgray')
        self.bprev = Button(axprev, '<', color='white', hovercolor='lightgray')

    def _same_event(self, event, eventtime):
        if ((self.event.x - event.x) == 0 and
                (self.event.y - event.y) == 0):
            if abs(self.eventtime - eventtime) < 0.5:
                # same event time
                return True
            else:
                return False
        else:
            return False

    def next(self, event):
        """Display next correlogram from file list."""
        eventtime = time.time()

        if self.eventnumber == 0:
            self.eventtime = eventtime
            self.event = event
            self.eventnumber += 1
            logger.debug('next - first event')
            if (self.chainidx + 1) > (self.nchains - 1):
                self.chainidx = 0
            else:
                self.chainidx += 1

            self.update_chain()

        elif not self._same_event(event, eventtime):
            self.eventtime = eventtime
            self.eventnumber += 1
            self.event = event
            logger.debug('next')
            if (self.chainidx + 1) > (self.nchains - 1):
                self.chainidx = 0
            else:
                self.chainidx += 1
            self.update_chain()

    def prev(self, event):
        """Display previous correlogram from file list."""
        eventtime = time.time()

        if self.eventnumber == 0:
            self.eventtime = time.time()
            self.event = event
            self.eventnumber += 1
            logger.debug('previous - first event')

            if (self.chainidx - 1) < 0:
                self.chainidx = (self.nchains - 1)
            else:
                self.chainidx -= 1

            self.chainidx -= 1
            self.update_chain()

        elif not self._same_event(event, eventtime):
            self.eventtime = eventtime
            self.eventnumber += 1
            self.event = event
            logger.debug('previous')
            if (self.chainidx - 1) < 0:
                self.chainidx = (self.nchains - 1)
            else:
                self.chainidx -= 1
            self.update_chain()

    def update_chain(self):
        print('New chain index:', self.chainidx)

        # if new chain is chosen
        self.modelmatrix, self.likes, self.noises = self.chainarrays[self.chainidx]
        nantmp = np.ones(self.modellength) * np.nan

        # reset age and rsl matrix to nan
        self.age_step = np.ones((self.capacity, self.modellength)) * np.nan
        self.rsl_step = np.ones((self.capacity, self.modellength)) * np.nan

        for i, model in enumerate(self.modelmatrix):
            # if nan model, the first element is also nan !
            if ~np.isnan(model[0]):
                x_age, y_rsl = Model.get_stepmodel(model)
                x_age[-1] = self.priors['age'][-1] * 1.5

                self.rsl_step = np.roll(self.rsl_step, -1, axis=0)  # rolling up models
                y_rsl = nantmp[:y_rsl.size] = y_rsl
                self.rsl_step[-1][:y_rsl.size] = y_rsl

                self.age_step = np.roll(self.age_step, -1, axis=0)  # rolling up models
                x_age = nantmp[:x_age.size] = x_age
                self.age_step[-1][:x_age.size] = x_age

                # update title
                self.axes[0].set_title('Chain %d' % self.chainidx)

                lastmodel = model

        # immediately update data fit lines
        x_age, y_rsl = Model.get_age_rsl(model)
        ymod = self.compute_synth(x_age, y_rsl)

        for i, tline in enumerate(self.targetlines):
            if tline is not None:
                tline.set_ydata(ymod[i])

        # # create LineCollection and update velocity models
        segments = [np.column_stack([x, y])
                    for x, y in zip(self.age_step, self.rsl_step)]
        self.modelcollection.set_segments(segments)
        self.likeline.set_ydata(self.likes)
        self.axes[4].set_ylim([np.nanmin(self.likes)*0.999,
                               np.nanmax(self.likes)*1.001])

        for i, sline in enumerate(self.sigmalines):
            if sline is not None:
                ref = self.targetrefs[i]
                idx = self.targetrefs.index(ref)
                sline.set_ydata(self.noises.T[1::2][idx])

        self.axes[5].set_ylim([np.nanmin(self.noises.T[1::2])*0.98,
                               np.nanmax(self.noises.T[1::2])*1.02])
        self.fig.canvas.draw_idle()

    def compute_synth(self, x_age, y_rsl):
        moddata = []
        # compute the synthetic data
        for target in self.targets:
            if target is None:
                moddata.append(np.nan)
                continue
            else:
                _, ymod = target.moddata.plugin.run_model(
                    x_age, y_rsl)
                moddata.append(ymod)

        return moddata

    def init_arrays(self):
        models = np.ones((self.capacity, self.modellength)) * np.nan
        likes = np.ones((self.capacity)) * np.nan
        noises = np.ones((self.capacity, self.ntargets*2)) * np.nan

        self.chainarrays = []
        for chain in np.arange(self.nchains):
            self.chainarrays.append((models, likes, noises))
        self.arrays = False

    def store_data(self, arrmodels=None, arrlikes=None, arrnoise=None):
        """Take input array and append to list data"""

        for idx, chain in enumerate(self.chainarrays):
            # print(idx)
            models, likes, noises = chain

            # if all the incoming values are identical, BayWatch stops updating
            # the specific chain. If no chain get delivered an update,
            # then the inversion is done and BayWatch stops waiting for
            # incoming arrays.
            if (np.nansum(models[-1] - models[-2]) == 0 and
                np.nansum(likes[-1] - likes[-2]) == 0 and
                    np.nansum(noises[-1] - noises[-2]) == 0):

                # ignore initiation phase when nan placeholds of arrays.
                if (~np.isnan(models[-2][0]) and
                    ~np.isnan(noises[-2][0]) and
                    ~np.isnan(likes[-2])):

                    self.breakloop[idx] = 1
                    continue

            if arrmodels is not None:

                ###
                model = arrmodels[idx]

                models = np.roll(models, -1, axis=0)  # rolling up models
                models[-1][:model.size] = model

                if idx == self.chainidx:
                    self.update_models(model)  # plot

            if arrlikes is not None:
                like = float(arrlikes[idx])

                likes = np.roll(likes, -1)
                likes[-1] = like

                if idx == self.chainidx:
                    self.update_likes(like)  # plot

            if arrnoise is not None:
                noise = arrnoise[idx]

                noises = np.roll(noises, -1, axis=0)
                noises[-1] = noise

                if idx == self.chainidx:
                    self.update_noises(noise)  # plot

            self.chainarrays[idx] = (models, likes, noises)

    def update_models(self, model):
        logger.debug('### Found new chain model')
        x_age, y_rsl = Model.get_stepmodel(model)

        self.rsl_step = np.roll(self.rsl_step, -1, axis=0)  # rolling up models
        nantmp = np.ones(self.modellength) * np.nan
        nantmp[:y_rsl.size] = y_rsl
        self.rsl_step[-1] = nantmp

        self.age_step = np.roll(self.age_step, -1, axis=0)  # rolling up models
        nantmp = np.ones(self.modellength) * np.nan
        nantmp[:x_age.size] = x_age
        self.x_age_step[-1] = nantmp

        x_age, y_rsl = Model.get_age_rsl(model)
        ymod = self.compute_synth(x_age, y_rsl)

        for i, tline in enumerate(self.targetlines):
            if tline is not None:
                tline.set_ydata(ymod[i])

        # # create LineCollection
        segments = [np.column_stack([x, y])
                    for x, y in zip(self.age_step, self.rsl_step)]
        self.modelcollection.set_segments(segments)

    def update_likes(self, like):
        self.likes = np.roll(self.likes, -1)
        self.likes[-1] = like
        self.likeline.set_ydata(self.likes)
        self.axes[4].set_ylim([np.nanmin(self.likes)*0.999,
                               np.nanmax(self.likes)*1.001])    

    def update_noises(self, noise):
        self.noises = np.roll(self.noises, -1, axis=0)
        self.noises[-1] = noise

        for i, sline in enumerate(self.sigmalines):
            if sline is not None:
                ref = self.targetrefs[i]
                idx = self.targetrefs.index(ref)
                sline.set_ydata(self.noises.T[1::2][idx])

        self.axes[5].set_ylim([np.nanmin(self.noises.T[1::2])*0.98,
                               np.nanmax(self.noises.T[1::2])*1.02])

    def watch(self):
        self.chainidx = 0
        self.axes[0].set_title('Chain %d' % self.chainidx)
        self.arrays = True
        self.eventnumber = 0
        self.lastincome = time.time()

        m = 0
        n = 0

        while True:
            arr = self.socket.recv_array()

            if self.arrays:
                self.nchains = len(arr)
                self.init_arrays()
                self.breakloop = np.zeros(self.nchains)

            if arr.shape[1] == 1:
                logger.debug('Received likelihood array')
                self.store_data(arrlikes=arr)

            elif (arr.shape[1] - 1) == self.modellength:
                logger.debug('Received model array: shape, %s' % str(arr.shape))
                self.store_data(arrmodels=arr[:, 1:])

            elif arr.shape[1] % 2 == 0:
                logger.debug('Received noise array: shape, %s' % str(arr.shape))
                self.store_data(arrnoise=arr)

            self.bnext.on_clicked(self.next)
            self.bprev.on_clicked(self.prev)

            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

            if np.all(self.breakloop):
                logger.info('BayHunter has finished the inversion.')
                self.socket.close()
                break

            if self.save_plots:
                if n % 10 == 0:
                    self.fig.savefig(self.save_plots.format(count=m))
                    m += 1
            n += 1

        while True:
            self.bnext.on_clicked(self.next)
            self.bprev.on_clicked(self.prev)

            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

            # keyboard interrupt not working...


def main():
    import sys
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description='Watch your BayHunter.')

    parser.add_argument(
        'folder', type=str,
        help='Path to BayWatch configfile, default %(default)s')
    parser.add_argument(
        '--address', default='127.0.0.1', type=str,
        help='Address where BayHunter is running, default %(default)s')
    parser.add_argument(
        '--port', default=5556, type=int,
        help='Port to connect to, default %(default)s')
    parser.add_argument(
        '--capacity', default=200, type=int,
        help='Number of displayed models, default %(default)s')
    parser.add_argument(
        '--save-plots', default=None, type=str,
        help='Path to save plots, format: /path/to/plots/fig{count:04d}.png')


    args = parser.parse_args()

    # use configfile saved directly before inversion start.
    # Make sure to start inversion before baywatch, otherwise you might get
    # displayed the reference data from the last, but not the current inversion.

    configfile = op.join(args.folder, 'baywatch.pkl')
    if not op.exists(configfile):
        print('Configfile %s not found!' % configfile)
        sys.exit(1)

    pro = BayWatcher(configfile=configfile, capacity=args.capacity,
                     address=args.address, port=args.port, save_plots=args.save_plots)

    # start baywatcher
    pro.watch()


if __name__ == '__main__':
    main()
