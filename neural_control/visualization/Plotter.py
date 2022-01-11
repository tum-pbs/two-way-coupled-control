import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import animation, use
from cycler import cycler
from itertools import cycle
import seaborn as sns
import mimic_alpha as ma
import matplotlib


class Plotter():

    def __init__(self, figsize: tuple = (7, 7), imshow_kwargs: dict = {}, plot_kwargs: dict = {}, arrows_kwargs: dict = {}, scatter_kwargs: dict = {}):
        """
        Initialize parameters

        Params:
            figsize: size of figure window
            imshow_kwargs: kwargs for imshow plots
            plot_kwargs: kwargs for plot
            arrow_kwargs: kwargs for quiver
            scatter_kwargs: kwargs for scatter plot

        """
        self.data = {}
        self.figs = {}
        self.imshow_kwargs = imshow_kwargs
        self.plot_kwargs = plot_kwargs
        self.arrows_kwargs = arrows_kwargs
        self.scatter_kwargs = scatter_kwargs
        self.should_export = []
        self.export_path = './'
        self.figsize = figsize
        plt.rc('axes', prop_cycle=(cycler('color', [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf'])))
        self.colors = cycle([u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf'])
        use('qt5agg')
        # Make plot iteractive
        # plt.ion()

    def set_export_path(self, path: str):
        """
        Set path to where figures will be exported to

        Params:
            path: path to folder where figures will be saved

        """
        self.should_export = True
        self.export_path = path
        if not os.path.isdir(self.export_path):
            os.mkdir(self.export_path)

    def clear(self):
        """
        Clear all plots

        """
        plt.close('all')
        self.data = {}
        self.figs = {}

    def add_data(self, dataset: list, ids: list):
        """
        Add data that will be used for plots later

        Params:
            dataset: list containing arrays for plots
            ids: list of same size of dataset containing strings for identifying dataset

        """
        assert isinstance(dataset, list)
        assert isinstance(ids, list)
        for data_values, id in zip(dataset, ids):
            self.data[id] = {}
            self.data[id]['values'] = data_values
            if np.ndim(data_values) == 1:
                self.data[id]['dim'] = '1D'
            elif np.ndim(data_values) == 2:
                self.data[id]['dim'] = '2D'
            elif np.ndim(data_values) == 3:
                self.data[id]['dim'] = '3D'
            else:
                raise ValueError('Data with incompatible dimensions')

    def check_plot_id(self, id: str, constrained_layout: bool = True):
        """
        Check if plot id already exists, otherwise create one

        Params:
            id: plot id
            contrained_layout: if true then figure will be created with contrained layout

        Return:
            fig: figure object
            ax: axes object
        """
        if id not in self.figs:
            fig, ax = plt.subplots(1, 1, figsize=self.figsize, constrained_layout=constrained_layout)
            # plt.tight_layout()
            # ax.set_rasterized(True)
            self.figs[id] = {'fig': fig, 'ax': ax}
        else:
            fig = self.figs[id]['fig']
            ax = self.figs[id]['ax']
        return fig, ax

    def imshow(self,
               data_ids: list = [None],
               plot_id: str = None,
               export_filename: str = None,
               transpose: bool = False,
               create_cbar: bool = True,
               create_title: bool = True,
               **kwargs):
        """
        Create imshow plot of data with ids in data_ids. If data_ids is None, then all data that admits imshow will be used.

        Params:
            data_ids: ids of data that will be used for imshows
            plot_id: id of axes in which imshow should be performed on
            export_filename: name of exported file
            transpose: if true then data will be plotted transposed
            kwargs: kwargs for imshow

        Return:
            image: image object
            ax: axes object
            fig: fig object

        """
        assert isinstance(data_ids, list)
        merged_kwargs = self.merge_kwargs(self.imshow_kwargs, kwargs)
        for id, data in self.data.items():
            if data_ids[0] is None:
                if data['dim'] is not '2D':
                    continue
            else:
                if id not in data_ids:
                    continue
            plot_id_local = id if plot_id is None else plot_id
            fig, ax = self.check_plot_id(plot_id_local)
            if transpose:
                image = ax.imshow(np.transpose(data['values']), **merged_kwargs)
            else:
                image = ax.imshow(data['values'], **merged_kwargs)
            if create_title: ax.set_title(id)
            cbar = fig.colorbar(image, ax=ax) if create_cbar else None
            export_filename_ = id if export_filename is None else export_filename
            if self.should_export:
                self.export(fig, export_filename_)
            print('Created imshow figure with data from %s ' % (id))
        return image, ax, fig, cbar

    def export(self, fig: "plt.figure", name: str):
        """
        Export figure

        Params:
            fig: figure that will be exported
            name: name of file

        """
        filePath = '%s/%s.pdf' % (self.export_path, name)
        if isinstance(fig, plt.Figure):
            fig.savefig(filePath, dpi=100)
        elif isinstance(fig, str):
            self.figs[fig]['fig'].savefig(filePath, dpi=100)

    def merge_kwargs(self, kwargs1: dict, kwargs2: dict):
        """
        Merge kwargs1 and kwargs2

        Params:
            kwargs1: dictionary containing kwargs for plot
            kwargs2: dictionary containing kwargs for plot

        Returns:
            merged_kwargs: dictionary created by merging kwargs1 and kwargs2
        """
        merged_kwargs = dict(kwargs1)
        for key, value in kwargs2.items():
            merged_kwargs[key] = value
        return merged_kwargs

    def plot(self, data_ids=[None],
             plot_id=None,
             export_filename=None,
             x_limits=None,
             y_limits=None,
             create_legend: bool = True,
             labels: dict = None,
             colors: dict = None,
             create_grid: bool = False,
             new_axis: bool = False,
             ylog: bool = False,
             fill_between: dict = None,
             ** kwargs):
        """
        Create plot of data in data_ids

        Params:
            data_ids: ids of data that will be used for imshows
            plot_id: id of axes in which plot should be performed on
            export_filename: name of exported file
            x_limits: x limits of plot
            y_limits: y limits of plot
            create_legend: if true then a legend will be created
            labels: dictionary containing labels for each data
            colors: dictionary containing colors for each data
            create_grid: if true then a grid will be created
            new_axis: if true then another vertical axes will be created
            ylog: if true then y axis will be logarithmic
            fill_between: dictionary containing fill_between parameters
            kwargs: kwargs for plot

        """
        assert isinstance(data_ids, list)
        merged_kwargs = self.merge_kwargs(self.plot_kwargs, kwargs)
        for id, data in self.data.items():
            if data_ids[0] is not None and id not in data_ids:
                continue
            plot_id_local = id if plot_id is None else plot_id
            fig, ax = self.check_plot_id(plot_id_local)
            if new_axis:
                if len(fig.axes) == 1:
                    ax = ax.twinx()
                else:
                    ax = fig.axes[1]
            # Labels
            if not labels: labels = {id: id}
            # Colors
            if colors:
                if id in colors:
                    merged_kwargs['color'] = colors[id]
            # Plot
            if data['dim'] == '1D':
                line = ax.plot(data['values'], label=labels[id] if create_legend else None, **merged_kwargs)
            if data['dim'] == '2D':
                line = ax.plot(*data['values'], label=labels[id] if create_legend else None, **merged_kwargs)
            if data['dim'] == '3D':
                line = []
                for xy in data['values']:
                    line.append(ax.plot(*xy, label=labels[id] if create_legend else None, **merged_kwargs))
            # Legend
            if create_legend: ax.legend()
            # Grid
            if create_grid: ax.grid(True, linestyle='--')
            # Limits
            if x_limits is not None: ax.set_xlim(x_limits)
            if y_limits is not None: ax.set_ylim(y_limits)
            # Y scale
            if ylog: ax.set_yscale('log')
            # Fill between (working only with 1D)
            if fill_between is not None and data['dim'] in ['1D', '2D']:
                offset = fill_between[id]['offset']
                color = ma.colorAlpha_to_rgb([line[0].get_color()], fill_between[id]['kwargs'].pop('alpha', 1))[0]  # Hack to simulate transparency
                values = data['values'] if data['dim'] == '1D' else data['values'][1]
                ax.fill_between(line[0].get_xdata(), values - offset, values + offset, color=color, **fill_between[id]['kwargs'])
            export_filename_ = id if export_filename is None else export_filename
            if self.should_export:
                self.export(fig, export_filename_)
            print('Created plot figure with data from %s ' % (id))
        return line, fig, ax

    def distribution(self, data_ids=[None], plot_id=None, export_filename=None, x_limits=None, y_limits=None, create_legend: bool = True, **kwargs):
        """
        Create distribution plot of data in data_ids

        Params:
            data_ids: ids of data that will be used for imshows
            plot_id: id of axes in which kdeplot should be performed on
            export_filename: name of exported file
            x_limits: x limits of plot
            y_limits: y limits of plot
            create_legend: if true then a legend will be created
            kwargs: kwargs for distribution (currently not implemented)

        """
        assert isinstance(data_ids, list)
        merged_kwargs = self.merge_kwargs(self.plot_kwargs, kwargs)
        for id, data in self.data.items():
            if data_ids[0] is None:
                if data['dim'] not in ['1D']:
                    continue
            else:
                if id not in data_ids:
                    continue
            plot_id_local = id if plot_id is None else plot_id
            fig, ax = self.check_plot_id(plot_id_local)
            sns.kdeplot(data['values'], label=id, ax=ax, **merged_kwargs)
            # ax.hist(data['values'], density=True, stacked=True, **hist_kwargs)
            mean = np.mean(data['values'])
            ax.autoscale(False)
            mean_color = ax.lines[-1].get_color()
            ax.plot([mean, mean], [0, 10], linestyle='--', color=mean_color, label=None)
            if create_legend:
                ax.legend()
            if x_limits is not None:
                ax.set_xlim(x_limits)
            if y_limits is not None:
                ax.set_ylim(y_limits)
            export_filename_ = id if export_filename is None else export_filename
            if self.should_export:
                self.export(fig, export_filename_)
            print('Created distribution figure with data from %s ' % (id))
        return fig, ax

    def animation(self, data_ids=[None], plot_id=None, should_export_frames=False, export_filename=None, **kwargs):
        """
        Create animation from various imshow plots

        Params:
            data_ids: ids of data that will be used for imshows
            plot_id: id of axes in which imshow should be performed on
            should_export_frames: if true frames will be exported
            export_filename: name of exported file
            kwargs: kwargs for animation

        """
        assert isinstance(data_ids, list)

        def animate(i, data, image):
            image.set_data(data[i])
            return image

        def export_animation(self, animation_handle, name):
            writer = animation.FFMpegWriter(fps=20)
            animation_handle.save('%s/%s.mp4' % (self.export_path, name), writer=writer)

        def export_frames(dataset, fig, name, image):
            if not os.path.isdir(self.export_path + '/frames/' + name):
                try:
                    os.mkdir(self.export_path + '/frames/' + name)
                except FileNotFoundError:
                    os.mkdir(self.export_path + '/frames/')
                    os.mkdir(self.export_path + '/frames/' + name)
            for i in range(len(dataset)):
                image = animate(i, dataset, image)
                self.export(fig, '/frames/' + name + '/' + name + str(i))
        merged_kwargs = self.merge_kwargs(self.imshow_kwargs, kwargs)
        for id, data in self.data.items():
            if data_ids[0] is None:
                if data['dim'] is not '3D':
                    continue
            else:
                if id not in data_ids:
                    continue
            plot_id_local = id if plot_id is None else plot_id
            fig, ax = self.check_plot_id(plot_id_local)
            image = ax.imshow(data['values'][0], **merged_kwargs)
            self.data[id]['image'] = image
            ax.set_title(id)
            animation_handle = animation.FuncAnimation(fig, animate, len(data['values']), fargs=(data['values'], image), interval=50)
            self.data[id]['animation'] = animation_handle
            export_filename_ = id if export_filename is None else export_filename
            if should_export_frames:
                export_frames(data['values'], fig, export_filename_, image)
            if self.should_export:
                export_animation(animation_handle, export_filename_)
            print('Created animation with data from %s ' % (id))

    def arrows_field(self, data_ids=[None, None], plot_id=None, draw_dots=True, export_filename=None, offset=[[-0.5, 0], [0, -0.5]], **kwargs):
        """
        Create quiver field from data U,V in data_ids

        Params:
            data_ids: ids of data that will be used for quiver
            plot_id: id of axes in which quiver should be performed on
            draw_dots: if true then dots at base of quivers will be drawn
            export_filename: name of exported file
            offset: offset quivers location
            kwargs: kwargs for quiver

        """
        merged_kwargs = self.merge_kwargs(self.arrows_kwargs, kwargs)
        min_UV = 0.0
        assert len(data_ids) == 2
        [u_name, v_name] = data_ids
        for id, data in self.data.items():
            if id not in [u_name, v_name]:
                continue
            nx = np.shape(data['values'])[1]
            ny = np.shape(data['values'])[0]
            x = np.arange(nx, dtype=np.float)
            y = np.arange(ny, dtype=np.float)
            if id == u_name:
                x += offset[0][0]
                y += offset[0][1]
            else:
                y += offset[1][1]
                x += offset[1][0]
            min_local = np.min(data['values'])
            if np.abs(min_local) < 1e-2:
                min_local = 1e-2
            min_UV = np.min([min_local, min_UV]) - 0.1
            if id == u_name:
                U = data['values']
                V = U * 0.0
            else:
                V = data['values']
                U = V * 0.0
            plot_id_local = id if plot_id is None else plot_id
            fig, ax = self.check_plot_id(plot_id_local)
            color = [np.sqrt(u**2) + np.sqrt(v**2) for u, v in zip(U, V)]
            color = (color - np.min(color)) / (np.max(color) - np.min(color))
            ax.quiver(x, y, U, V, **merged_kwargs)
            if draw_dots:
                [X, Y] = np.meshgrid(x, y)
                ax.scatter(np.ndarray.flatten(X), np.ndarray.flatten(Y), c=color, cmap='Purples')
        # Export
        export_filename_ = id if export_filename is None else export_filename
        if self.should_export:
            self.export(fig, export_filename_)
        print('Created quiver figure with data from %s ' % (id))

    def scatter(self, data_ids=[None], plot_id=None, offset=[0.0, 0.0], should_normalize=False, export_filename=None, **kwargs):
        """
        Create scatter plot from data in data_ids

        Params:
            data_ids: ids of data that will be used for scatter
            plot_id: id of axes in which scatter should be performed on
            offset: offset scatter location
            should_normalize: normalize values for colors
            export_filename: name of exported file
            kwargs: kwargs for scatter
        """
        assert isinstance(data_ids, list)
        merged_kwargs = self.merge_kwargs(self.scatter_kwargs, kwargs)
        for id, data in self.data.items():
            if data_ids[0] is None:
                if data['dim'] != '2D':
                    continue
            else:
                if id not in data_ids:
                    continue
            plot_id_local = id if plot_id is None else plot_id
            fig, ax = self.check_plot_id(plot_id_local)
            x = np.arange(data['values'].shape[1]) + offset[0]
            y = np.arange(data['values'].shape[0]) + offset[1]
            [X, Y] = np.meshgrid(x, y)
            colors = np.array(data['values'])
            if should_normalize:
                colors = ((colors - colors.flatten().min()) / (colors.flatten().max() - colors.flatten().min()) - 0.5) * 2.0
            ax.scatter(X, Y, c=data['values'], **merged_kwargs)
            # Export
            export_filename_ = id if export_filename is None else export_filename
            if self.should_export:
                self.export(fig, export_filename_)
            print('Created scatter figure with data from %s ' % (id))

    # def fig2rgb_array(self, fig: "matplotlib figure" = None, expand: bool = True) -> np.array:
    #     """
    #     Return the flattened rgb array of fig

    #     param: fig: figure which will be used to create an rgb array
    #     param: expand: determine if the shape of array should be of rank 3 or 4
    #     return: np.array containing rgb of figure
    #     """
    #     if fig == None:
    #         fig = plt.figure(figsize=self.figsize)
    #     fig.canvas.draw()
    #     buf = fig.canvas.tostring_rgb()
    #     ncols, nrows = fig.canvas.get_width_height()
    #     shape = (nrows, ncols, 3) if not expand else (1, nrows, ncols, 3)
    #     plt.close(fig)
    #     return np.frombuffer(buf, dtype=np.uint8).reshape(shape)

    def show(self, block=True):
        plt.show(block=block)

    def remove_repeated_labels(self, plot_id: str):
        """
        Remove repeated labels from legend

        Params:
            plot_id: id of plot which will be modified

        """
        _, ax = self.check_plot_id(plot_id)
        handles, labels = ax.get_legend_handles_labels()
        legend_dict = dict(zip(labels, handles))
        ax.legend(legend_dict.values(), legend_dict.keys())


if __name__ == '__main__':
    p = Plotter()

    x, y = np.arange(-5, 5), np.arange(-5, 5)
    z = 2 * x**2
    zz = 3 * y
    [X, Y] = np.meshgrid(x, y)
    Z = X**2 + Y**2
    ZZ = X * 0
    ZZ[4:8, 4:8] = 1.0
    p.add_data([Z], ['arrows'])
    p.add_data([-Z], ['arrows_negative'])
    p.add_data([ZZ], ['obs'])
    p.add_data([z, zz], ['z', 'zz'])
    # TODO update tests
    # ------------------------------------------------------
    # Test imshow
    # p.imshow(['arrows'], 'plot', block=False, alpha = 0.4)
    # p.imshow(['arrows','obs'], 'plot', block=True, alpha=0.4)
    # p.imshow(block=True)

    # ------------------------------------------------------
    # Test plots
    # p.plot(block=True)

    # p.plot(['z'], plot_id='plot',block=False)
    # p.plot(['zz'], plot_id='plot',block=True)
    # p.plot(['zz','z'], plot_id='plot',block=True)

    # ------------------------------------------------------

    # Test quivers
    # p.arrows_field('arrows', 'arrows_negative', plot_id='plot', block=True)
    # p.imshow(['obs'], 'plot', block=True, alpha=0.5)

    # ------------------------------------------------------
    # Test scatter
    # p.scatter(['arrows'], should_normalize=True)
    # p.show()
    # input('debug')
