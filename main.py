from mpl_plot_definitions import AllPlots, create_figure
from mpl_plot_definitions import SingleEditorPlot, DoubleEditorPlot
from invertible_model import InvertibleModel
from audio_midi_dataset import get_xy_from_file
import numpy as np

# from matplotlib.backends.qt_compat import QtCore
from PyQt5 import QtGui
from PyQt5.QtCore import Qt

from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.backends.qt_compat import is_pyqt5

if is_pyqt5():
    from matplotlib.backends.backend_qt5agg import FigureCanvas
else:
    from matplotlib.backends.backend_qt4agg import FigureCanvas

filenames_and_starts = [
    ('./data/maps_piano/data/AkPnBcht/ISOL/NO/MAPS_ISOL_NO_F_S0_M69_AkPnBcht', 0),
    ('./data/maps_piano/data/AkPnBcht/MUS/MAPS_MUS-chp_op31_AkPnBcht', 100),
    ('./data/maps_piano/data/ENSTDkCl/MUS/MAPS_MUS-chpn-p19_ENSTDkCl', 100)
]


class SingleEditButtonsWidget(QtWidgets.QFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.layout = QtWidgets.QVBoxLayout(self)

        self.state = 'draw'
        free_button = QtWidgets.QRadioButton('draw')
        free_button.setChecked(True)
        free_button.state = 'draw'
        free_button.toggled.connect(self.on_click)

        zero_button = QtWidgets.QRadioButton('zero region')
        zero_button.state = 'zero'
        zero_button.toggled.connect(self.on_click)

        one_button = QtWidgets.QRadioButton('ones region')
        one_button.state = 'one'
        one_button.toggled.connect(self.on_click)

        gauss_button = QtWidgets.QRadioButton('gauss region')
        gauss_button.state = 'gauss'
        gauss_button.toggled.connect(self.on_click)

        uniform_button = QtWidgets.QRadioButton('uniform region')
        uniform_button.state = 'uniform'
        uniform_button.toggled.connect(self.on_click)

        self.layout.addWidget(free_button)
        self.layout.addWidget(zero_button)
        self.layout.addWidget(one_button)
        self.layout.addWidget(gauss_button)
        self.layout.addWidget(uniform_button)

    def on_click(self):
        radio_button = self.sender()
        if radio_button.isChecked():
            self.state = radio_button.state


class SingleEditorWidget(QtWidgets.QFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.layout = QtWidgets.QHBoxLayout(self)
        # create a mpl figure
        self.figure = create_figure()

        self.edit_buttons = SingleEditButtonsWidget(self)

        # create the figurecanvas object
        self.canvas = FigureCanvas(self.figure)

        self.layout.addWidget(self.canvas)
        self.layout.addStretch(1)
        self.layout.addWidget(self.edit_buttons)
        self.plot = None

    def set_plot(self, title, data, editor_range, color='cyan'):
        axes = self.canvas.figure.get_axes()
        for ax in axes:
            self.canvas.figure.delaxes(ax)
        del axes
        self.plot = SingleEditorPlot(
            self.figure,
            title,
            data,
            color,
            self.edit_buttons,
            editor_range=editor_range
        )


class DoubleEditButtonsWidget(QtWidgets.QFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.layout = QtWidgets.QVBoxLayout(self)

        self.state = 'note'
        note_button = QtWidgets.QRadioButton('note')
        note_button.setChecked(True)
        note_button.state = 'note'
        note_button.toggled.connect(self.on_click)

        free_button = QtWidgets.QRadioButton('draw')
        free_button.state = 'draw'
        free_button.toggled.connect(self.on_click)

        zero_button = QtWidgets.QRadioButton('zero region')
        zero_button.state = 'zero'
        zero_button.toggled.connect(self.on_click)

        one_button = QtWidgets.QRadioButton('ones region')
        one_button.state = 'one'
        one_button.toggled.connect(self.on_click)

        gauss_button = QtWidgets.QRadioButton('gauss region')
        gauss_button.state = 'gauss'
        gauss_button.toggled.connect(self.on_click)

        uniform_button = QtWidgets.QRadioButton('uniform region')
        uniform_button.state = 'uniform'
        uniform_button.toggled.connect(self.on_click)

        self.layout.addWidget(note_button)
        self.layout.addWidget(free_button)
        self.layout.addWidget(zero_button)
        self.layout.addWidget(one_button)
        self.layout.addWidget(gauss_button)
        self.layout.addWidget(uniform_button)

    def on_click(self):
        radio_button = self.sender()
        if radio_button.isChecked():
            self.state = radio_button.state


class DoubleEditorWidget(QtWidgets.QFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.layout = QtWidgets.QHBoxLayout(self)
        # create a mpl figure
        self.figure = create_figure()

        self.edit_buttons = DoubleEditButtonsWidget(self)

        # create the figurecanvas object
        self.canvas = FigureCanvas(self.figure)

        self.layout.addWidget(self.canvas)
        self.layout.addStretch(1)
        self.layout.addWidget(self.edit_buttons)
        self.plot = None

    def set_plot(self, data_velocity, data_phase, editor_range_velocity, editor_range_phase):
        axes = self.canvas.figure.get_axes()
        for ax in axes:
            self.canvas.figure.delaxes(ax)
        del axes
        self.plot = DoubleEditorPlot(
            self.figure,
            'double editor',
            data_velocity,
            data_phase,
            'orange',
            'gray',
            self.edit_buttons,
            editor_range_velocity=editor_range_velocity,
            editor_range_phase=editor_range_phase
        )


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.T = 60
        self.invertible_model = InvertibleModel(self.T)

        self._main = QtWidgets.QFrame(self)
        self.vertical_layout = QtWidgets.QVBoxLayout(self._main)
        self._main.setLayout(self.vertical_layout)
        self.setCentralWidget(self._main)

        self.toolbar = self.addToolBar('Toolbar')

        combo = QtWidgets.QComboBox(self)
        combo.addItem('Simple Example')
        combo.addItem('Complex Example 1')
        combo.addItem('Complex Example 2')
        combo.setCurrentIndex(0)
        combo.activated.connect(self.combo_activated)
        self.toolbar.addWidget(combo)

        compute_encode_action = QtWidgets.QAction('encode f(x) -> y,z', self)
        compute_encode_action.triggered.connect(self.compute_encode)
        self.toolbar.addAction(compute_encode_action)

        compute_decode_action = QtWidgets.QAction('decode f_inv(y,z) -> x', self)
        compute_decode_action.triggered.connect(self.compute_decode)
        self.toolbar.addAction(compute_decode_action)

        denoise_action = QtWidgets.QAction('denoise [y, yz_pad, z]', self)
        denoise_action.triggered.connect(self.denoise)
        self.toolbar.addAction(denoise_action)

        # zero_yz_pad_action = QtWidgets.QAction('yz padding -> 0', self)
        # zero_yz_pad_action.triggered.connect(self.zero_yz_pad)
        # self.toolbar.addAction(zero_yz_pad_action)

        # confusing, leave out
        # label = 'yz padding -> N(0, {:>4.2g})'.format(self.invertible_model.model.zeros_noise_scale)
        # noise_yz_pad_action = QtWidgets.QAction(label, self)
        # noise_yz_pad_action.triggered.connect(self.noise_yz_pad)
        # self.toolbar.addAction(noise_yz_pad_action)

        gauss_z_action = QtWidgets.QAction('z -> N(0, 1)', self)
        gauss_z_action.triggered.connect(self.gauss_z)
        self.toolbar.addAction(gauss_z_action)

        self.ssim_label = QtWidgets.QLabel('structural similarity(x, x_hat) = {:4.2g}'.format(0.9999), self)

        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.toolbar.addWidget(spacer)
        self.toolbar.addWidget(self.ssim_label)

        # self.threshold_slider = QtWidgets.QSlider(Qt.Horizontal, self)
        # self.threshold_slider.setMinimum(0)
        # self.threshold_slider.setMaximum(500)
        # self.threshold_slider.setValue(50)
        # self.threshold_slider.valueChanged.connect(self.changeThreshold)
        # self.toolbar.addWidget(self.threshold_slider)

        # threshold_label = 'set all values < {:4.2g} to 0'.format(
        #     self.threshold_slider.value() / 100.
        # )
        # self.threshold_action = QtWidgets.QAction(threshold_label, self)
        # self.threshold_action.triggered.connect(self.threshold)
        # self.toolbar.addAction(self.threshold_action)

        # hide_editors_action = QtWidgets.QAction('Hide Editors', self)
        # hide_editors_action.triggered.connect(self.hide_editors)
        # self.toolbar.addAction(hide_editors_action)

        self.upper = QtWidgets.QFrame(self._main)
        self.lower_single = SingleEditorWidget(self._main)
        self.lower_single.setFixedHeight(170)
        self.lower_double = DoubleEditorWidget(self._main)
        self.lower_double.setFixedHeight(210)

        self.vertical_layout.addWidget(self.upper)
        self.vertical_layout.addWidget(self.lower_single)
        self.vertical_layout.addWidget(self.lower_double)

        upper_layout = QtWidgets.QHBoxLayout(self.upper)

        #####################################################
        # these calls need to stay in this order! otherwise
        # the event handlers registered in the figure
        # will be overwritten!

        # create a mpl figure
        self.figure = create_figure()

        # create the figurecanvas object
        self.canvas = FigureCanvas(self.figure)

        # plot into the figure
        # self.combo_activated(combo.currentIndex())
        filename, start = filenames_and_starts[combo.currentIndex()]

        x, x_frames, x_velocity = get_xy_from_file(
            filename + '.flac',
            filename + '.mid', self.invertible_model.audio_options
        )
        print('x.shape', x.shape)
        print('x_frames.shape', x_frames.shape)
        print('x_velocity.shape', x_velocity.shape)

        self.x_true = np.array(x[start: start + self.T])

        self.z_pred, self.yz_pad_pred, self.y_pred = self.invertible_model.encode(self.x_true)

        self.x_inv, self.x_inv_padding = self.invertible_model.decode(
            self.z_pred,
            self.yz_pad_pred,
            self.y_pred
        )

        #################################################################
        # test only

        # self.x_true = np.random.uniform(0, 1, (self.T, 256))

        # self.y_pred = np.random.uniform(0, 1, (self.T, 185))
        # self.z_pred = np.random.uniform(0, 1, (self.T, 9))
        # self.yz_pad_pred = np.random.uniform(0, 1, (self.T, 62))

        # self.x_inv = np.random.uniform(0, 1, (self.T, 256))
        #################################################################
        print('self.x_true.shape', self.x_true.shape)
        print('self.y_pred.shape', self.y_pred.shape)
        print('self.yz_pad_pred.shape', self.yz_pad_pred.shape)
        print('self.z_pred.shape', self.z_pred.shape)
        print('self.x_inv.shape', self.x_inv.shape)
        print('self.x_inv_padding.shape', self.x_inv_padding.shape)

        self.all_plots = AllPlots(
            fig=self.figure,
            x_true=self.x_true.view(),

            y_pred=self.y_pred.view(),
            z_pred=self.z_pred.view(),
            yz_pad_pred=self.yz_pad_pred.view(),

            x_inv=self.x_inv.view()
        )

        upper_layout.addWidget(self.canvas)

        # connect the signals to slots
        self.all_plots.ipo_y_instrument.do_select.connect(self.select)
        self.all_plots.ipo_y_phase.do_select.connect(self.select_pv)
        self.all_plots.ipo_y_velocity.do_select.connect(self.select_pv)
        self.all_plots.ipo_z.do_select.connect(self.select)
        self.all_plots.ipo_yz_pad_pred.do_select.connect(self.select)
        ######################################################

        self.selected_y = 8
        self.selected_ipo = self.all_plots.ipo_y_instrument
        self.select(self.selected_ipo, self.selected_y)
        self.lower_single.show()
        self.lower_double.hide()
        self.upper.setFocus()

    def compute_encode(self):
        self.z_pred[:], self.yz_pad_pred[:], self.y_pred[:] = self.invertible_model.encode(
            self.x_true
        )
        self.compute_decode()
        self.all_plots.redraw()

    def compute_decode(self):
        self.x_inv[:], self.x_inv_padding[:] = self.invertible_model.decode(
            self.z_pred,
            self.yz_pad_pred,
            self.y_pred
        )

        from skimage.measure import compare_ssim
        data_range = self.x_inv.max() - self.x_inv.min()
        similarity = compare_ssim(self.x_true, self.x_inv, data_range=data_range)

        self.ssim_label.setText('structural similarity: {:4.2g}'.format(similarity))

        self.all_plots.redraw()

    def load_file(self, filename, start):
        x, x_frames, x_velocity = get_xy_from_file(
            filename + '.flac',
            filename + '.mid', self.invertible_model.audio_options
        )
        print('x.shape', x.shape)
        print('x_frames.shape', x_frames.shape)
        print('x_velocity.shape', x_velocity.shape)

        self.x_true[:] = np.array(x[start: start + self.T])

        self.z_pred[:], self.yz_pad_pred[:], self.y_pred[:] = self.invertible_model.encode(self.x_true)

        self.x_inv[:], self.x_inv_padding[:] = self.invertible_model.decode(
            self.z_pred,
            self.yz_pad_pred,
            self.y_pred
        )

    def combo_activated(self, index):
        print('***********************************')
        filename, start = filenames_and_starts[index]
        self.load_file(filename, start)
        self.compute_encode()

    # def zero_yz_pad(self):
    #     self.yz_pad_pred[:] = np.zeros_like(self.yz_pad_pred)
    #     self.compute_decode()

    # def noise_yz_pad(self):
    #     noise = self.invertible_model.model.zeros_noise_scale
    #     self.yz_pad_pred[:] = np.random.uniform(0, noise, self.yz_pad_pred.shape)
    #     self.compute_decode()

    def denoise(self):
        self.yz_pad_pred[:] = np.zeros_like(self.yz_pad_pred)
        self.z_pred[:] = np.random.normal(0, 1, self.z_pred.shape)

        ########################################################
        # heuristics, heuristics everywhere!

        phase_start = 0
        phase_end = phase_start + 88

        vel_start = phase_end
        vel_end = vel_start + 88

        inst_start = vel_end
        inst_end = inst_start + 9

        inst = self.y_pred[:, inst_start:inst_end]
        vel = self.y_pred[:, vel_start:vel_end]
        phase = self.y_pred[:, phase_start:phase_end]

        inst[inst < 0.5] = 0

        pn = np.sum(phase > 2, axis=0)
        pn = np.nonzero(pn)[0]

        zero_out_idx = list(set(range(88)) - set(pn))
        vel[:, zero_out_idx] = 0
        phase[:, zero_out_idx] = 0

        self.compute_decode()

    def gauss_z(self):
        self.z_pred[:] = np.random.normal(0, 1, self.z_pred.shape)
        self.compute_decode()

    # def changeThreshold(self, value):
    #     fvalue = float(value) / 100.
    #     # self.threshold_label.setText('{:4.2g}'.format(fvalue))
    #     threshold_label = 'set all values < {:4.2g} to 0'.format(fvalue)
    #     self.threshold_action.setText(threshold_label)

    # def threshold(self):
    #     threshold = float(self.threshold_slider.value()) / 100.
    #     # self.yz_pad_pred[self.yz_pad_pred < threshold] = 0
    #     # self.z_pred[self.z_pred < threshold] = 0
    #     self.y_pred[self.y_pred < threshold] = 0
    #     self.compute_decode()

    def unselect_all(self):
        self.all_plots.ipo_y_instrument.unselect()
        self.all_plots.ipo_y_phase.unselect()
        self.all_plots.ipo_y_velocity.unselect()
        self.all_plots.ipo_z.unselect()
        self.all_plots.ipo_yz_pad_pred.unselect()

    def select(self, ipo, y):
        self.unselect_all()
        self.selected_y = ipo.select(y)
        self.selected_ipo = ipo
        self.all_plots.redraw()

        self.lower_double.hide()
        self.lower_single.show()
        self.lower_single.set_plot(
            title=ipo.title,
            data=ipo.img[:, self.selected_y],
            editor_range=ipo.editor_range,
            color=ipo.color()
        )
        self.statusBar().showMessage('Editing {}, y={}'.format(ipo.title, self.selected_y))

    def select_pv(self, ipo, y):
        self.unselect_all()
        self.selected_y = self.all_plots.ipo_y_phase.select(y)
        self.selected_y = self.all_plots.ipo_y_velocity.select(y)

        self.selected_ipo = self.all_plots.ipo_y_phase

        self.all_plots.redraw()

        self.lower_single.hide()
        self.lower_double.show()
        self.lower_double.set_plot(
            data_velocity=self.all_plots.ipo_y_velocity.img[:, self.selected_y],
            data_phase=self.all_plots.ipo_y_phase.img[:, self.selected_y],
            editor_range_velocity=self.all_plots.ipo_y_velocity.editor_range,
            editor_range_phase=self.all_plots.ipo_y_phase.editor_range
        )
        self.statusBar().showMessage('Editing Note Phase/Velocity, y={}'.format(y))

    def edit_done(self):
        self.editing = False
        self.lower_double.hide()
        self.lower_single.hide()
        self.all_plots.redraw()

    def update_selection(self):
        if self.selected_ipo == self.all_plots.ipo_y_phase:
            self.select_pv(self.selected_ipo, self.selected_y)
        else:
            self.select(self.selected_ipo, self.selected_y)

    def keyPressEvent(self, e):
        key = e.key()
        if key == Qt.Key_Escape:
            self.close()
        elif key == Qt.Key_F:
            self.selected_y += 1
            self.update_selection()
        elif key == Qt.Key_S:
            self.selected_y -= 1
            self.update_selection()


if __name__ == "__main__":
    # if we don't pass the style, we'll get some warnings
    # https://github.com/therecipe/qt/issues/306
    qapp = QtWidgets.QApplication(['main.py', '--style', 'Fusion'])
    app = ApplicationWindow()
    app.show()
    qapp.exec_()
