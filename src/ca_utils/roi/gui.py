"""GUI for drawing ROIs."""

import sys
from qtpy import QtCore, QtWidgets
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication, QHBoxLayout, QSlider, QVBoxLayout, QWidget
import pyqtgraph as pg
import numpy as np
import ca_utils.io as ca
from ca_utils.roi import plot
import xarray as xr
import logging
import defopt


logging.basicConfig(level=logging.INFO)
pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)

# Size (in pixels) for ROI vertex/translate handles to improve usability
ROI_HANDLE_SIZE = 18


class DoubleSlider(QSlider):
    # create our our signal that we can connect to if necessary
    doubleValueChanged = QtCore.Signal(float)

    def __init__(self, decimals=3, *args, **kargs):
        super(DoubleSlider, self).__init__(*args, **kargs)
        self._multi = 10**decimals

        self.valueChanged.connect(self.emitDoubleValueChanged)

    def emitDoubleValueChanged(self):
        value = float(super(DoubleSlider, self).value()) / self._multi
        self.doubleValueChanged.emit(value)

    def value(self):
        return float(super(DoubleSlider, self).value()) / self._multi

    def setMinimum(self, value):
        return super(DoubleSlider, self).setMinimum(value * self._multi)

    def setMaximum(self, value):
        return super(DoubleSlider, self).setMaximum(value * self._multi)

    def setSingleStep(self, value):
        return super(DoubleSlider, self).setSingleStep(value * self._multi)

    def singleStep(self):
        return float(super(DoubleSlider, self).singleStep()) / self._multi

    def setValue(self, value):
        super(DoubleSlider, self).setValue(int(value * self._multi))


class StackView(pg.GraphicsLayoutWidget):
    image_data_changed = QtCore.Signal(object)
    time_changed = QtCore.Signal(object)
    layer_changed = QtCore.Signal(object)
    channel_changed = QtCore.Signal(object)
    trial_changed = QtCore.Signal(object)
    roi_changed = QtCore.Signal(object)

    def __init__(self, data_file, trial_number=0, parent=None, label=None):
        super().__init__(title="Ca", parent=parent)

        self.colors = "rgbcymw"
        self.rois = []
        self.roi_traces = []
        self.label = label

        self.da = self.load_data(data_file=data_file, trial_number=trial_number)

        self.current_time = min(self.da.time)
        self.current_layer = 0
        self.current_channel = "gcamp"

        # A plot area (ViewBox + axes) for displaying the image
        self.plt_image = self.addPlot(title="")

        # # Item for displaying image self.data
        self.img = pg.ImageItem()

        self._data = self.temporal_avg.sel(z=self.current_layer, channel=self.current_channel).data
        self.img.setImage(self.data)
        self.img.mouseClickEvent = self.mouseClicked

        self.plt_image.addItem(self.img)
        self.plt_image.hideAxis("left")
        self.plt_image.hideAxis("bottom")
        self.plt_image.setDefaultPadding(0)
        self.vb = self.plt_image.getViewBox()
        self.vb.setAspectLocked(True, ratio=1)
        self.img.setLevels(np.percentile(self.data, [1, 99]))
        # zoom to fit image
        self.plt_image.autoRange()

        self.image_data_changed.connect(self.on_image_data_changed)

        self.show_time_avg()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value
        self.image_data_changed.emit(value)

    def mouseClicked(self, event):
        """Show the position, pixel, and value under the mouse cursor."""
        pos = event.pos()
        ppos = self.img.mapToParent(pos)
        x, y = ppos.x(), ppos.y()
        roi = pg.PolyLineROI(
            [[x - 30, y - 30], [x + 30, y - 30], [x + 30, y + 30], [x - 30, y + 30]],
            closed=True,
            pen=pg.mkPen(color=self.colors[len(self.rois) % len(self.colors)], width=4),
            # maxBounds=QtCore.QRect(0, 0, *self.data.shape),
        )
        roi.move_handle = roi.addTranslateHandle([x, y], name="move_handle")
        # Enlarge handles to make them easier to grab
        self._enlarge_roi_handles(roi)
        # Ensure translate handle is also enlarged if not covered above
        try:
            roi.move_handle.setSize(ROI_HANDLE_SIZE)
        except Exception:
            try:
                if hasattr(roi.move_handle, "radius"):
                    roi.move_handle.radius = ROI_HANDLE_SIZE
            except Exception:
                pass
        roi.changed = True  # indicate mask needs update, which will trigger trace update
        roi.mask_changed = True  # indicate trace needs update
        roi.mask = None
        roi.trace = None
        roi.layer = self.current_layer
        print(self.current_layer)
        roi.sigRegionChangeFinished.connect(self.signal_update_traces)
        roi.setAcceptedMouseButtons(QtCore.Qt.MouseButton.RightButton)
        roi.sigClicked.connect(self.remove_roi)
        self.plt_image.addItem(roi)
        self.rois.append(roi)
        traces = self.get_traces()
        self.roi_changed.emit(traces)

    def signal_update_traces(self, roi=None):
        if roi is None:
            for roi in self.rois:
                roi.changed = True
        else:
            roi.changed = True
        traces = self.get_traces()
        self.roi_changed.emit(traces)

    def remove_roi(self, roi):
        logging.info(f"Removing ROI {self.rois.index(roi)}.")

        # remove right-clicked roi from list and plot
        self.plt_image.removeItem(roi)
        self.rois.remove(roi)

        # # remove all rois from plot
        for cnt, roi in enumerate(self.rois):
            (roi.setPen(color=self.colors[cnt % len(self.colors)], width=4))
            self.plt_image.removeItem(roi)
        # add all remaining rois to plot again - required to keep colors consistent between rois and traces
        for roi in self.rois:
            self.plt_image.addItem(roi)

        traces = self.get_traces()
        self.roi_changed.emit(traces)
        # self.image_data_changed.emit(self._data)

    def _enlarge_roi_handles(self, roi):
        """Increase the size of all handles of a given ROI.

        Tries multiple APIs for compatibility across pyqtgraph versions.
        """
        try:
            handles = roi.getHandles()
        except Exception:
            handles = []

        for h in handles:
            # Some pyqtgraph versions expose setSize on Handle
            try:
                h.setSize(ROI_HANDLE_SIZE)
                continue
            except Exception:
                pass
            # Fall back to setting radius directly if available
            try:
                if hasattr(h, "radius"):
                    h.radius = ROI_HANDLE_SIZE
                # Trigger a redraw if possible
                if hasattr(h, "update"):
                    h.update()
            except Exception:
                pass

    def get_masks(self):
        masks = []
        if len(self.rois):
            cols, rows = self.data.shape
            nb_layers = len(self.da.z)
            m = np.mgrid[:cols, :rows]
            possx = m[0, :, :]  # make the x pos array
            possy = m[1, :, :]  # make the y pos array
            possx.shape = cols, rows
            possy.shape = cols, rows
            for roi in self.rois:
                if roi.changed:
                    if roi.move_handle in roi.getHandles():
                        roi.removeHandle(roi.move_handle)
                    mpossx = roi.getArrayRegion(possx, self.img).astype(int)
                    mpossx = mpossx[np.nonzero(mpossx)]  # get the x pos from ROI
                    mpossy = roi.getArrayRegion(possy, self.img).astype(int)
                    mpossy = mpossy[np.nonzero(mpossy)]  # get the y pos from ROI
                    mask = np.zeros((nb_layers, cols, rows), dtype=bool)
                    mask[self.current_layer, mpossx, mpossy] = True
                    roi.mask = mask
                    roi.mask_changed = True
                    roi.changed = False
            masks = xr.DataArray(
                np.stack([roi.mask for roi in self.rois], axis=-1), name="rois", dims=["z", "x", "y", "roi"]
            )  # V, X, Y, N
        return masks

    def get_traces(self, masks=None):
        traces = None
        if len(self.rois):
            if masks is None:
                masks = self.get_masks()
            for cnt, roi in enumerate(self.rois):
                if roi.mask_changed:
                    roi.trace = plot.extract_one_trial(self.da, masks[..., cnt : cnt + 1])
                    roi.mask_changed = False
            traces = xr.concat((roi.trace for roi in self.rois), dim="roi")
        return traces

    def on_image_data_changed(self, data):
        self.img.setImage(self.data)
        self.img.setLevels(np.percentile(self.data, [1, 99]))  # auto contrast
        # only show rois in current layer
        for roi in self.rois:
            if roi.layer == self.current_layer:
                roi.show()
            else:
                roi.hide()

    def show_time_avg(self):
        self.data = self.temporal_avg.data[0, ..., 0]

    def update_time(self, new_time):
        self.current_time = new_time
        data = self.da.sel(time=self.current_time, method="nearest")
        data = data.sel(z=self.current_layer, method="nearest")
        self.data = data.sel(channel=self.current_channel).data

    def update_layer(self, slider_value):
        self.current_layer = int(slider_value)
        data = self.da.sel(time=self.current_time, method="nearest")
        data = data.sel(z=self.current_layer, method="nearest")
        self.data = data.sel(channel=self.current_channel).data

    def on_channel_changed(self, new_channel_name):
        self.current_channel = new_channel_name
        data = self.da.sel(time=self.current_time, method="nearest")
        data = data.sel(z=self.current_layer, method="nearest")
        self.data = data.sel(channel=self.current_channel).data

    def on_trial_changed(self, new_trial_index):
        self.current_trial = new_trial_index  # !! NEED TO CHANGE THE WIDGET TO A DROPDOWN WITH CHANNEL NAMES
        self.da = self.load_trial(self.current_trial)
        self.signal_update_traces()

    def load_data(self, state=True, data_file=None, trial_number=0):
        if data_file is None:
            data_file, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open DAQ file", "./", "DAQ file (*_daq.h5)")

        self.data_file = data_file.removesuffix("_daq.h5")
        self.session = ca.Session(self.data_file)
        da = self.load_trial(trial_number)
        self.clear_rois()
        return da

    def load_trial(self, trial_number):
        self.trial_log = self.session.log.loc[trial_number]

        da = self.session.stack(trial_number)
        self.label.setText(
            f"Session {self.data_file}, trial {trial_number}, data: {f"[{', '.join([f"{dim}: {length}" for dim, length in zip(da.dims, da.shape)])}]"}"
        )

        self.temporal_avg = da.mean(dim="time")
        self.show_time_avg()
        return da

    def clear_rois(self):
        for roi in self.rois:
            self.plt_image.removeItem(roi)

        self.rois = []
        traces = None
        self.roi_changed.emit(traces)

    def save_rois(self):
        if not len(self.rois):
            logging.warning("Nothing to save. No ROIs annotated yet.")
        else:
            masks = self.get_masks()
            save_file = self.data_file + "_rois.h5"
            logging.warning(f"Saving {masks.shape[0]} ROIs and traces to {save_file}.")
            masks.to_netcdf(save_file)


class TracesView(pg.GraphicsLayoutWidget):
    def __init__(self, stack_view):
        super().__init__(title="Traces")

        self.colors = "rgbcymw"
        self.mode = "raw"
        self.stack_view = stack_view
        self.plt_traces = self.addPlot(colspan=2, rowspan=2)
        self.plt_traces.setMaximumHeight(250)
        self.plt_traces.setLabel("bottom", "Time [seconds]")

        self.f0_region = None
        self.stim_region = None
        # stims
        self.stims = [k.rstrip("_onset_ms") for k in self.stack_view.trial_log.keys() if "_onset_ms" in k]
        self.stim = self.stims[0]

    def on_mode_changed(self, mode_value):
        self.mode = mode_value
        traces = self.stack_view.get_traces()
        self.update_traces(traces)

    def on_stim_changed(self, stim_value):
        self.stim = stim_value
        traces = self.stack_view.get_traces()
        self.plt_traces.removeItem(self.stim_region)

        self.update_traces(traces)

    def signal_update_traces(self):
        traces = self.stack_view.get_traces()
        self.update_traces(traces)

    def update_traces(self, traces):
        self.plt_traces.clearPlots()
        if traces is None:
            return
        if self.stim_region is not None:
            self.plt_traces.removeItem(self.stim_region)
        stim_onset_seconds = self.stack_view.trial_log[self.stim + "_onset_ms"] / 1_000
        stim_offset_seconds = self.stack_view.trial_log[self.stim + "_offset_ms"] / 1_000
        self.stim_region = pg.LinearRegionItem(values=(stim_onset_seconds, stim_offset_seconds), movable=False)
        pg.InfLineLabel(
            self.stim_region.lines[1], self.stim, position=0.95, angle=0, rotateAxis=(1, 0), anchor=(1, 1), movable=False
        )
        self.plt_traces.addItem(self.stim_region)

        if self.f0_region is None:
            self.f0_region = pg.LinearRegionItem(values=(traces.time[0], traces.time[-1] // 10))
            self.f0_region.sigRegionChangeFinished.connect(self.signal_update_traces)
            pg.InfLineLabel(self.f0_region.lines[1], "F0", position=0.95, angle=0, rotateAxis=(1, 0), anchor=(1, 1))
            self.plt_traces.addItem(self.f0_region)

        chan = 0  # TODO: make this string to addres gcamp/tdtom
        for roi_cnt in range(traces.shape[-1]):
            trace = traces[..., chan, roi_cnt]
            if self.mode != "raw":
                values = self.f0_region.getRegion()
                f0 = np.mean(trace.sel(time=slice(values[0], values[1])))

            if self.mode == "dF":
                trace = trace - f0
            if self.mode == "dF/F":
                trace = (trace - f0) / f0

            self.plt_traces.plot(trace.time, trace, skipFiniteCheck=True, pen=self.colors[roi_cnt % len(self.colors)])


class Widget(QWidget):
    def __init__(self, data_file, trial_number, parent=None):
        super(Widget, self).__init__(parent=parent)
        self.resize(1200, 600)

        # THIS LAYOUT IS A BIG CONFUSING MESS!!!!
        self.mainVertLayout = QVBoxLayout(self)
        self.main_label = QtWidgets.QLabel("Fluorescence normalization")
        self.main_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.mainVertLayout.addWidget(self.main_label)

        self.mainHorzLayout = QHBoxLayout(self)

        self.verticalLayout = QVBoxLayout(self)

        self.stack_view = StackView(data_file, trial_number, parent=self, label=self.main_label)
        self.verticalLayout.addWidget(self.stack_view)

        self.horzTimeLayout = QHBoxLayout(self)

        button = QtWidgets.QPushButton("&Show time avg")
        button.clicked.connect(self.stack_view.show_time_avg)
        self.horzTimeLayout.addWidget(button)

        # button = QtWidgets.QPushButton("&Show time avg")
        # button.clicked.connect(self.stack_view.show_time_avg)

        label = QtWidgets.QLabel("Time")
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.horzTimeLayout.addWidget(label)

        self.w1 = DoubleSlider(orientation=Qt.Horizontal)
        self.w1.setMinimum(int(min(self.stack_view.da.time)))
        self.w1.setMaximum(int(max(self.stack_view.da.time)))
        self.w1.doubleValueChanged.connect(self.stack_view.update_time)
        self.horzTimeLayout.addWidget(self.w1)

        self.verticalLayout.addLayout(self.horzTimeLayout)

        # Layer
        if len(self.stack_view.da.z) > 1:
            # self.w2 = QtWidgets.QSlider(orientation=Qt.Horizontal)
            # self.w2.setMinimum(0)
            # self.w2.setMaximum(len(self.stack_view.da.z) - 1)
            # self.verticalLayout.addWidget(self.w2)
            # self.w2.valueChanged.connect(self.stack_view.update_layer)
            self.w2 = DoubleSlider(orientation=Qt.Horizontal)
            self.w2.setMinimum(float(min(self.stack_view.da.z)))
            self.w2.setMaximum(float(max(self.stack_view.da.z)))
            self.verticalLayout.addWidget(self.w2)
            self.w2.doubleValueChanged.connect(self.stack_view.update_layer)

        # button row
        self.horizontalLayout = QHBoxLayout(self)
        # Channel
        label = QtWidgets.QLabel("Channel")
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.horizontalLayout.addWidget(label)

        self.w3 = QtWidgets.QComboBox()
        self.w3.addItems(self.stack_view.da.channel.data.tolist())
        self.w3.currentTextChanged.connect(self.stack_view.on_channel_changed)
        self.horizontalLayout.addWidget(self.w3)

        # Trial
        label = QtWidgets.QLabel("Trial")
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.horizontalLayout.addWidget(label)
        self.w4 = QtWidgets.QComboBox()
        self.w4.addItems([str(t) for t in range(len(self.stack_view.session.log))])
        self.w4.currentIndexChanged.connect(self.stack_view.on_trial_changed)
        self.horizontalLayout.addWidget(self.w4)

        button = QtWidgets.QPushButton("&Load data")
        button.clicked.connect(self.stack_view.load_data)
        self.horizontalLayout.addWidget(button)

        button = QtWidgets.QPushButton("&Save ROIs")
        button.clicked.connect(self.stack_view.save_rois)
        self.horizontalLayout.addWidget(button)

        button = QtWidgets.QPushButton("&Clear ROIs")
        button.clicked.connect(self.stack_view.clear_rois)
        self.horizontalLayout.addWidget(button)

        self.verticalLayout.addLayout(self.horizontalLayout)

        self.vertTracesLayout = QVBoxLayout(self)

        self.traces_view = TracesView(self.stack_view)
        # self.stack_view.update_traces = self.traces_view.update_traces
        self.stack_view.roi_changed.connect(self.traces_view.update_traces)
        self.vertTracesLayout.addWidget(self.traces_view)

        self.horzTracesLayout = QHBoxLayout(self)
        label = QtWidgets.QLabel("Mode")
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.horzTracesLayout.addWidget(label)
        self.w4 = QtWidgets.QComboBox()
        self.w4.addItems(["raw", "dF", "dF/F"])
        self.w4.currentTextChanged.connect(self.traces_view.on_mode_changed)
        self.horzTracesLayout.addWidget(self.w4)

        label = QtWidgets.QLabel("Stimulus")
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.horzTracesLayout.addWidget(label)
        self.w5 = QtWidgets.QComboBox()
        self.w5.addItems(self.traces_view.stims)
        self.w5.currentTextChanged.connect(self.traces_view.on_stim_changed)
        self.horzTracesLayout.addWidget(self.w5)

        self.vertTracesLayout.addLayout(self.horzTracesLayout)

        self.mainHorzLayout.addLayout(self.verticalLayout)
        self.mainHorzLayout.addLayout(self.vertTracesLayout)

        self.mainVertLayout.addLayout(self.mainHorzLayout)


def main(data_file: str, trial_number: int = 0):
    """_summary_

    Args:
        data_file (str): _description_
        trial_number (int): _description_. Defaults to 0.
    """
    app = QApplication(sys.argv)
    w = Widget(data_file=data_file, trial_number=trial_number)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    defopt.run(main)
