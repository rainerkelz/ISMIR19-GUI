import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PyQt5.QtCore import pyqtSignal, QObject


DIVERGING_CMAP = 'RdBu_r'
PAD = 0.3
FONT_SIZE = 8
LABEL_SIZE = 10
ROTATION = 0


# exactly as in mumoma
def curve(start, end):
    duration = end - start
    xs = np.arange(0, duration)
    return (0.99 ** xs) * 5


def create_figure():
    return plt.figure(figsize=(16, 9))


def nround(a):
    return int(round(a))


class AllPlots(object):
    def __init__(self, fig, x_true, y_pred, z_pred, yz_pad_pred, x_inv):
        figsize = (32, 18)
        labelsize = 10
        pad = 0.03
        height_ratios = [9, 88, 88, 9, 62]
        rotation = 0
        font_size = 8

        phase_start = 0
        phase_end = phase_start + 88

        vel_start = phase_end
        vel_end = vel_start + 88

        inst_start = vel_end
        inst_end = inst_start + 9

        self.fig = fig
        self.gs = self.fig.add_gridspec(
            nrows=5,
            ncols=4,
            wspace=0.3,
            hspace=0.1,
            height_ratios=height_ratios
        )

        self.x_true = x_true
        self.x_inv = x_inv
        # this needs to be refreshed!
        self.x_diff = x_true - x_inv

        ########################################################################################
        # plotting all the images
        self.ipo_x = ImagePlot(
            fig=self.fig,
            gs=self.gs[:, 0],
            title='x',
            img=x_true,
            cmap='viridis'
        )

        self.ipo_y_instrument = SelectableImagePlot(
            fig=self.fig,
            gs=self.gs[0, 1],
            title='inst',
            img=y_pred[:, inst_start:inst_end],
            cmap='Purples',
            sharex=self.ipo_x.ax,
            editor_range=(-0.1, 0, 1, 1.1)
        )

        self.ipo_y_velocity = SelectableImagePlot(
            fig=self.fig,
            gs=self.gs[1, 1],
            title='vel',
            img=y_pred[:, vel_start:vel_end],
            cmap='Oranges',
            sharex=self.ipo_x.ax,
            editor_range=(-0.1, 0, 1, 1.1)
        )

        self.ipo_y_phase = SelectableImagePlot(
            fig=self.fig,
            gs=self.gs[2, 1],
            title='phase',
            img=y_pred[:, phase_start:phase_end],
            cmap='gray_r',
            sharex=self.ipo_x.ax,
            editor_range=(-0.5, 0, 5, 5.5)
        )

        self.ipo_z = SelectableImagePlot(
            fig=self.fig,
            gs=self.gs[3, 1],
            title='z',
            img=z_pred,
            cmap=DIVERGING_CMAP,
            sharex=self.ipo_x.ax,
            editor_range=(-3.5, -3, 3, 3.5)
        )

        self.ipo_yz_pad_pred = SelectableImagePlot(
            fig=self.fig,
            gs=self.gs[4, 1],
            title='yz pad',
            img=yz_pad_pred,
            cmap='Greens',
            sharex=self.ipo_x.ax,
            editor_range=(-0.1, -0.01, 0.01, 0.1)
        )

        self.ipo_x_inv = ImagePlot(
            fig=self.fig,
            gs=self.gs[:, 2],
            title='x inv',
            img=x_inv,
            cmap='viridis',
            sharex=self.ipo_x.ax,
            sharev=x_true
        )

        self.ipo_diff = ImagePlot(
            fig=self.fig,
            gs=self.gs[:, 3],
            title='x diff',
            img=self.x_diff,
            cmap=DIVERGING_CMAP,
            sharex=self.ipo_x.ax
        )
        self.fig.subplots_adjust(
            left=0.02,
            right=0.95,
            top=0.98,
            bottom=0.02
        )

    def redraw(self):
        # draw plots according to state
        self.ipo_x.redraw()
        self.ipo_x_inv.redraw()
        self.ipo_y_instrument.redraw()
        self.ipo_y_phase.redraw()
        self.ipo_y_velocity.redraw()
        self.ipo_yz_pad_pred.redraw()
        self.ipo_z.redraw()

        self.x_diff[:] = self.x_true - self.x_inv
        self.ipo_diff.redraw()

        # draw canvas
        self.fig.canvas.draw()


class ImagePlot(object):
    def __init__(self,
                 fig=None,
                 gs=None,
                 title=None,
                 img=None,
                 cmap=None,
                 sharex=None,
                 sharey=None,
                 text=None,
                 pad=PAD,
                 label_size=LABEL_SIZE,
                 font_size=FONT_SIZE,
                 rotation=ROTATION,
                 sharev=None,
                 editor_range=None):
        super().__init__()
        self.fig = fig

        self.ax = self.fig.add_subplot(gs, sharex=sharex, sharey=sharey)
        self.sharev = sharev
        self.editor_range = editor_range

        self.title = title
        self.img = img
        self.cmap = cmap
        self.text = text
        divider = make_axes_locatable(self.ax)
        self.cax = divider.append_axes('right', size='5%', pad=pad)
        self.label_size = label_size
        self.font_size = font_size
        self.rotation = rotation
        self.fig.canvas.mpl_connect('button_press_event', self.__on_click)
        self.cbar = None
        self.redraw()

    def color(self):
        return mpl.cm.get_cmap(self.cmap)(200)

    def redraw(self):
        print('redraw img_plot')
        self.ax.clear()
        self.ax.xaxis.set_major_locator(plt.NullLocator())
        self.ax.yaxis.set_major_locator(plt.NullLocator())
        width, length = self.img.shape

        if self.sharev is None:
            min_v = np.min(self.img)
            max_v = np.max(self.img)
        else:
            min_v = min(np.min(self.sharev), np.min(self.img))
            max_v = max(np.max(self.sharev), np.max(self.img))

        self.im = self.ax.imshow(
            self.img.T,
            origin='lower',
            cmap=self.cmap,
            vmin=min_v,
            vmax=max_v,
            aspect='auto'
        )

        self.ax.set_ylabel(self.title)
        if self.text is not None:
            self.ax.text(length * 0.3, 2, self.text, fontdict=dict(size=self.font_size))

        if self.cbar is None:
            self.cbar = self.fig.colorbar(
                self.im, cax=self.cax, ticks=[min_v, max_v], orientation='vertical'
            )
        self.cbar.set_clim([min_v, max_v])
        self.cbar.ax.set_yticklabels([
            '{:4.2g}'.format(min_v),
            '{:4.2g}'.format(max_v)
        ], rotation=self.rotation)
        self.cbar.ax.tick_params(labelsize=self.label_size)

    def get_image(self):
        return self.im

    def __on_click(self, event):
        if event.inaxes == self.ax:
            if hasattr(self, 'on_click'):
                self.on_click(event)


class SelectableImagePlot(QObject, ImagePlot):
    do_select = pyqtSignal([ImagePlot, int])

    def __init__(self, *args, **kwargs):
        self.selection = None
        super().__init__(**kwargs)

    def on_click(self, event):
        self.do_select.emit(self, nround(event.ydata))

    def select(self, y):
        print('selected', self.title, y)
        print('self.img.shape[1]', self.img.shape[1])
        self.selection = min(max(0, y), self.img.shape[1] - 1)
        return self.selection

    def unselect(self):
        print('unselected', self.title)
        self.selection = None

    def redraw(self):
        print('redraw selectable img_plot')
        super().redraw()
        if self.selection is not None:
            width, length = self.img.shape
            rect = mpl.patches.Rectangle(
                xy=(-0.5, self.selection - 0.5),
                width=width,
                height=1,
                linewidth=1,
                edgecolor='cyan',
                facecolor=mpl.colors.to_rgba('cyan', 0.3)
            )
            self.ax.add_patch(rect)


class ActiveXRectangle(object):
    def __init__(self, ax, color='b'):
        self.ax = ax
        self.rect = None
        self.color = color

    def start_tracking(self, start):
        self.start = start

    def track(self, current):
        self.end = current
        self.draw()

    def end_tracking(self, end):
        self.end = end

    def get_start_end(self):
        return sorted([self.start, self.end])

    def draw(self):
        if self.rect is not None:
            self.rect.remove()
            self.rect = None

        start, end = self.get_start_end()
        y_lo, y_hi = sorted(self.ax.get_ylim())
        height = y_hi - y_lo
        self.rect = mpl.patches.Rectangle(
            xy=(start - 0.5, y_lo),
            width=end - start,
            height=height,
            linewidth=1,
            edgecolor=self.color,
            facecolor=mpl.colors.to_rgba(self.color, 0.3)
        )
        self.ax.add_patch(self.rect)


class ActiveYRectangle(ActiveXRectangle):
    def track_height(self, height):
        self.height = height

    def get_height(self):
        return self.height

    def draw(self):
        if self.rect is not None:
            self.rect.remove()
            self.rect = None

        start, end = self.get_start_end()
        self.rect = mpl.patches.Rectangle(
            xy=(start - 0.5, 0),
            width=end - start,
            height=self.height,
            linewidth=1,
            edgecolor=self.color,
            facecolor=mpl.colors.to_rgba(self.color, 0.3)
        )
        self.ax.add_patch(self.rect)


class SingleEditorPlot(object):
    def __init__(self,
                 fig,
                 title,
                 data,
                 color,
                 edit_buttons,
                 sharex=None,
                 sharey=None,
                 position=111,
                 editor_range=None):

        self.fig = fig
        self.ax = fig.add_subplot(position)
        self.title = title
        self.editor_range = editor_range

        self.data = data

        self.color = color

        self.edit_buttons = edit_buttons

        self.state = 'up'

        self.active_rect = ActiveXRectangle(self.ax, self.color)
        self.last_draw_pos = None
        self.last_draw_val = None

        self.fig.canvas.mpl_connect('button_press_event', self.on_down)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)

        self.fig.canvas.mpl_connect('button_release_event', self.on_up)
        self.fig.canvas.mpl_connect('figure_leave_event', self.on_up)
        self.fig.canvas.mpl_connect('axes_leave_event', self.on_up)

        self.fig.subplots_adjust(
            left=0.1,
            right=0.99,
            top=0.99,
            bottom=0.05
        )

        self.redraw_data()

    def redraw_data(self):
        self.ax.clear()
        xs = np.arange(len(self.data))
        y_lo = np.zeros(len(self.data))
        y_hi = self.data
        self.ax.vlines(xs, y_lo, y_hi, color=self.color)
        self.ax.scatter(xs, y_hi, marker='o', edgecolor=self.color, facecolor=self.color)
        lol = None
        hil = None
        if self.editor_range is not None:
            lo, lol, hil, hi = self.editor_range
        else:
            lo = min(0, min(y_hi) * 1.1)
            hi = max(0.1, max(y_hi) * 1.1)
        self.ax.set_ylim([lo, hi])

        options = dict(
            linestyles='dashed',
            linewidth=1,
            alpha=0.4
        )
        if lol is not None:
            self.ax.hlines([lol], [0], [len(self.data)], **options)

        if hil is not None:
            self.ax.hlines([hil], [0], [len(self.data)], **options)

        self.ax.set_ylabel(self.title)
        self.ax.set_xticks([])
        self.fig.canvas.draw()

    def on_down(self, event):
        if event.inaxes != self.ax:
            return

        if self.edit_buttons.state in ['draw', 'zero', 'one', 'gauss', 'uniform']:
            self.state = 'down'
            if self.edit_buttons.state == 'draw':
                x = self.clip_x_to_bounds(nround(event.xdata))
                x = min(x, len(self.data) - 1)
                self.last_draw_pos = x
                self.last_draw_val = event.ydata
            elif self.edit_buttons.state in ['zero', 'one', 'gauss', 'uniform']:
                self.active_rect.start_tracking(nround(event.xdata))

    def clip_x_to_bounds(self, ae):
        ae = max(0, ae)
        return min(ae, len(self.data))

    def on_move(self, event):
        if event.inaxes != self.ax:
            return

        if self.edit_buttons.state in ['draw', 'zero', 'one', 'gauss', 'uniform']:
            if self.state == 'down':
                if self.edit_buttons.state == 'draw':
                    x = self.clip_x_to_bounds(nround(event.xdata))
                    x = min(x, len(self.data) - 1)
                    y = event.ydata
                    start, end = sorted([x, self.last_draw_pos])
                    if start == end:
                        self.data[start] = y
                    else:
                        values = np.linspace(self.last_draw_val, y, end - start)
                        if x < self.last_draw_pos:
                            values = values[::-1]
                        self.data[start:end] = values
                    self.last_draw_pos = x
                    self.last_draw_val = y
                    self.redraw_data()
                elif self.edit_buttons.state in ['zero', 'one', 'gauss', 'uniform']:
                    self.active_rect.track(self.clip_x_to_bounds(nround(event.xdata)))

                self.fig.canvas.draw()

    def on_up(self, event):
        if event.inaxes != self.ax:
            return

        if self.edit_buttons.state in ['draw', 'zero', 'one', 'gauss', 'uniform']:
            if self.state == 'down':
                self.state = 'up'
                if self.edit_buttons.state == 'draw':
                    pass
                elif self.edit_buttons.state in ['zero', 'one', 'gauss', 'uniform']:
                    self.active_rect.end_tracking(self.clip_x_to_bounds(nround(event.xdata)))

                    start, end = self.active_rect.get_start_end()
                    if self.edit_buttons.state == 'zero':
                        self.data[start:end] = 0
                    if self.edit_buttons.state == 'one':
                        self.data[start:end] = 1
                    elif self.edit_buttons.state == 'gauss':
                        self.data[start:end] = np.random.normal(0, 1, end - start)
                    elif self.edit_buttons.state == 'uniform':
                        self.data[start:end] = np.random.uniform(0, 1, end - start)

                self.redraw_data()


class DoubleEditorPlot(object):
    def __init__(self,
                 fig,
                 title,
                 data_velocity,
                 data_phase,
                 color_velocity,
                 color_phase,
                 edit_buttons,
                 editor_range_velocity=None,
                 editor_range_phase=None,
                 sharex=None,
                 sharey=None):

        self.fig = fig
        self.title = title
        self.plot_velocity = SingleEditorPlot(
            self.fig,
            'velocity',
            data_velocity,
            color_velocity,
            edit_buttons,
            sharex=sharex,
            sharey=sharey,
            position=211,
            editor_range=editor_range_velocity
        )
        self.plot_phase = SingleEditorPlot(
            self.fig,
            'phase',
            data_phase,
            color_phase,
            edit_buttons,
            sharex=sharex,
            sharey=sharey,
            position=212,
            editor_range=editor_range_phase
        )

        self.active_y = ActiveYRectangle(self.plot_velocity.ax, color_velocity)
        self.active_x = ActiveXRectangle(self.plot_phase.ax, color_phase)

        self.edit_buttons = edit_buttons

        self.state = 'up'

        self.fig.canvas.mpl_connect('button_press_event', self.on_down)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)

        self.fig.canvas.mpl_connect('button_release_event', self.on_up)
        self.fig.canvas.mpl_connect('figure_leave_event', self.on_up)
        self.fig.canvas.mpl_connect('axes_leave_event', self.on_up)

        self.fig.subplots_adjust(
            left=0.05,
            right=0.99,
            top=0.99,
            bottom=0.05
        )

        self.redraw_data()

    def redraw_data(self):
        self.plot_phase.redraw_data()
        self.plot_velocity.redraw_data()

    def clip_x_to_bounds(self, x):
        x = max(0, x)
        return min(x, len(self.plot_velocity.data))

    def clip_y_to_bounds(self, y):
        y = max(0, y)
        return min(y, max(self.plot_velocity.ax.get_ylim()))

    def on_down(self, event):
        if self.edit_buttons.state == 'note':
            self.state = 'down'
            print('on_down dbl')
            self.active_x.start_tracking(self.clip_x_to_bounds(nround(event.xdata)))
            self.active_y.start_tracking(self.clip_x_to_bounds(nround(event.xdata)))
            self.active_y.track_height(self.clip_y_to_bounds(nround(event.ydata)))

    def on_move(self, event):
        if self.edit_buttons.state == 'note':
            if self.state == 'down':
                print('nround(event.xdata)', nround(event.xdata))
                self.active_x.track(self.clip_x_to_bounds(nround(event.xdata)))

                self.active_y.track(self.clip_x_to_bounds(nround(event.xdata)))
                self.active_y.track_height(self.clip_y_to_bounds(event.ydata))

                self.active_x.draw()
                self.active_y.draw()

                self.fig.canvas.draw()

    def on_up(self, event):
        if self.edit_buttons.state == 'note':
            if self.state == 'down':
                self.state = 'up'
                print('on_up dbl')
                self.active_x.end_tracking(self.clip_x_to_bounds(nround(event.xdata)))

                self.active_y.end_tracking(self.clip_x_to_bounds(nround(event.xdata)))
                self.active_y.track_height(self.clip_y_to_bounds(event.ydata))

                start, end = self.active_x.get_start_end()
                height = self.active_y.get_height()

                self.plot_velocity.data[start:end] = height
                self.plot_phase.data[start:end] = curve(start, end)

                self.redraw_data()
