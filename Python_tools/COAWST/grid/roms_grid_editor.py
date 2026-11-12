#!/usr/bin/env python3
"""Interactive ROMS/COAWST mask and bathymetry editor (Python >=3.10).

Install: conda install -c conda-forge numpy "matplotlib>=3.8" netcdf4 pyqt
Or: python -m pip install numpy "matplotlib>=3.8" netCDF4 PyQt6
Run: python roms_grid_editor.py grid.nc --rivers rivers.nc
     python roms_grid_editor.py grid.nc --output grid_edited.nc --vmax 15
Spyder: select a Qt graphics backend (or run %matplotlib qt), then:
    from roms_grid_editor import launch
    editor = launch('grid.nc', rivers='rivers.nc')

Controls
--------
Point: left-click a cell; Area: drag a rectangle (mask operations only).
Modes: Inspect, Toggle mask, Set land, Set sea, Set depth.
For depth editing, enter meters in 'Depth (m)', then click an individual cell.
Depth changes do not change the land/sea mask. Depth must be finite and >0.
Use the Matplotlib toolbar for pan/zoom; disable it before editing.
Undo/Redo affect individual actions. Revert restores the last saved state.
Checkboxes show/hide rivers, depth contours, and the translucent mask.
Set Min depth / Max depth and click Apply range (or press Enter in either
field) to highlight that interval. This updates colors and contour levels
only; depths outside the interval use the end colors and remain editable.
All locations use zero-based (i,j) grid coordinates, without lon/lat labels.
River o -> * symbols follow mean transport; x means no mean direction.

Save writes to grid_edited.nc by default. Existing output files require
--overwrite-output on the first save. Input and output must be distinct.
The source is copied byte-for-byte to a temporary file, then h, mask_rho,
mask_u, mask_v, mask_psi and history are updated before atomic replacement.
Other variables/attributes are preserved. Existing output is backed up on
EVERY save. Derived masks are created if absent. Arrays are oriented using
NetCDF dimension names rather than MATLAB/Python memory order.

Depth changes are local: no smoothing, vertical-coordinate regeneration,
or initial/boundary-condition regeneration is performed. Review bathymetry
and consistency of dependent model inputs before running the model.
No external coastline download is needed: the mask outline is displayed.
No auto-save: use Save before closing (closing prints an unsaved warning).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import os
import shutil
import tempfile
import warnings

import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, RadioButtons, RectangleSelector, TextBox


def read_2d(var):
    """Return an eta-by-xi array and its canonical dimension order."""
    dims = var.dimensions
    eta = [d for d in dims if d.startswith('eta_')]
    xi = [d for d in dims if d.startswith('xi_')]
    if len(dims) != 2 or len(eta) != 1 or len(xi) != 1:
        raise ValueError(f'{var.name}: expected two named eta/xi dimensions, got {dims}')
    order = (eta[0], xi[0])
    data = np.ma.asarray(var[:], dtype=float).filled(np.nan)
    return data.transpose([dims.index(d) for d in order]), order


def staggered_masks(mask):
    return {'mask_u': mask[:, :-1] * mask[:, 1:],
            'mask_v': mask[:-1, :] * mask[1:, :],
            'mask_psi': mask[:-1, :-1] * mask[1:, :-1] *
                        mask[:-1, 1:] * mask[1:, 1:]}


class Grid:
    def __init__(self, filename):
        self.source = Path(filename).expanduser().resolve()
        self.source_stamp = self.stamp()
        with Dataset(self.source) as ds:
            self.depth, self.dims = read_2d(ds['h'])
            self.mask, mask_dims = read_2d(ds['mask_rho'])
            if self.depth.shape != self.mask.shape or self.dims != mask_dims:
                raise ValueError('h and mask_rho must share the same rho grid.')
        if min(self.mask.shape) < 2 or not np.isin(self.mask, [0, 1]).all():
            raise ValueError('mask_rho must be a finite binary array at least 2 x 2.')
        self.saved = (self.depth.copy(), self.mask.copy())
        self.undo_stack, self.redo_stack = [], []

    def stamp(self):
        stat = self.source.stat()
        return stat.st_mtime_ns, stat.st_size

    @property
    def dirty(self):
        return not (np.array_equal(self.depth, self.saved[0], equal_nan=True) and
                    np.array_equal(self.mask, self.saved[1]))

    def change(self, field, indices, values):
        array = getattr(self, field)
        indices = np.asarray(indices, dtype=int).ravel()
        old = array.flat[indices].copy()
        new = np.broadcast_to(values, old.shape).copy()
        different = ~((old == new) | (np.isnan(old) & np.isnan(new)))
        if not different.any():
            return
        indices, old, new = indices[different], old[different], new[different]
        array.flat[indices] = new
        self.undo_stack.append((field, indices, old, new))
        self.redo_stack.clear()

    def undo(self):
        if self.undo_stack:
            action = self.undo_stack.pop()
            field, indices, old, new = action
            getattr(self, field).flat[indices] = old
            self.redo_stack.append(action)

    def redo(self):
        if self.redo_stack:
            action = self.redo_stack.pop()
            field, indices, old, new = action
            getattr(self, field).flat[indices] = new
            self.undo_stack.append(action)

    def revert(self):
        self.depth[:], self.mask[:] = self.saved
        self.undo_stack.clear()
        self.redo_stack.clear()

    def save(self, output, allow_overwrite=False):
        output = Path(output).expanduser().resolve()
        if output == self.source:
            raise ValueError('Choose an output filename different from the input grid.')
        if self.stamp() != self.source_stamp:
            raise ValueError('Input grid changed on disk. Reopen the editor before saving.')
        if output.exists() and not allow_overwrite:
            raise FileExistsError(f'{output} exists; use --overwrite-output to authorize replacement.')
        wet = self.mask == 1
        #if not np.all(np.isfinite(self.depth[wet]) & (self.depth[wet] > 0)):
            #raise ValueError('All wet cells need finite positive depths before saving.')
        output.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix='.roms_edit_', suffix='.nc', dir=output.parent)
        os.close(fd)
        temporary = Path(name)
        backup = None
        try:
            shutil.copy2(self.source, temporary)
            with Dataset(temporary, 'r+') as ds:
                fields = {'h': self.depth, 'mask_rho': self.mask,
                          **staggered_masks(self.mask)}
                for key, array in fields.items():
                    if key not in ds.variables:
                        suffix = key.split('_', 1)[1]
                        dims = (f'eta_{suffix}', f'xi_{suffix}')
                        for dim, size in zip(dims, array.shape):
                            if dim not in ds.dimensions:
                                ds.createDimension(dim, size)
                            elif len(ds.dimensions[dim]) != size:
                                raise ValueError(f'{dim}: inconsistent dimension size')
                        var = ds.createVariable(key, 'f8', dims)
                        var.long_name = f'land/sea mask at {suffix} points'
                        var.flag_values = np.array([0., 1.])
                        var.flag_meanings = 'land water'
                    var = ds[key]
                    existing, canonical = read_2d(var)
                    if existing.shape != array.shape:
                        raise ValueError(f'{key}: incompatible shape')
                    # Reject integer depth storage rather than silently rounding edits.
                    if key == 'h' and np.dtype(var.dtype).kind not in 'fc':
                        raise ValueError('h must use floating-point storage for this editor.')
                    var[:] = array.transpose([canonical.index(d) for d in var.dimensions])
                    actual, _ = read_2d(var)
                    if not np.allclose(actual, array, rtol=1e-6, atol=1e-6, equal_nan=True):
                        raise ValueError(f'{key}: storage precision/packing cannot retain edits.')
                entry = (datetime.now(timezone.utc).isoformat() +
                         ': roms_grid_editor.py updated bathymetry and land/sea masks')
                ds.history = str(getattr(ds, 'history', '')).rstrip() + '\n' + entry
            if output.exists():
                tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                backup = output.with_name(output.name + f'.{tag}.bak')
                shutil.copy2(output, backup)
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
        self.saved = (self.depth.copy(), self.mask.copy())
        self.undo_stack.clear()
        self.redo_stack.clear()
        return backup


def read_rivers(filename, shape):
    """Read horizontal river faces with the supplied MATLAB map_rivers convention."""
    with Dataset(filename) as ds:
        def vector(key):
            return np.ma.asarray(ds[key][:], dtype=float).filled(np.nan).ravel()
        ids, x, y, direction = [vector(k) for k in
            ('river', 'river_Xposition', 'river_Eposition', 'river_direction')]
        if not (len(ids) == len(x) == len(y) == len(direction)):
            raise ValueError('River variable lengths do not match.')
        dim = ds['river_Xposition'].dimensions[0]
        var = ds['river_transport']
        q = np.ma.asarray(var[:], dtype=float).filled(np.nan)
        q = np.moveaxis(q, var.dimensions.index(dim), 0).reshape(len(ids), -1)
        q[~np.isfinite(q)] = np.nan
        count = np.sum(np.isfinite(q), axis=1)
        mean = np.divide(np.nansum(q, axis=1), count,
                         out=np.full(len(ids), np.nan), where=count > 0)
    ny, nx = shape
    result = []
    for r, (rid, i, j, d, flow) in enumerate(zip(ids, x, y, direction, mean)):
        valid = np.isfinite([i, j]).all() and i == int(i) and j == int(j)
        valid = valid and ((d == 0 and 1 <= i < nx and 0 <= j < ny) or
                           (d == 1 and 0 <= i < nx and 1 <= j < ny))
        if not valid:
            warnings.warn(f'Skipping river {rid:g}: invalid face or unsupported direction {d}.')
            continue
        if np.any(q[r] > 0) and np.any(q[r] < 0):
            warnings.warn(f'River {rid:g} reverses flow; arrow shows mean direction.')
        hi = np.array([i, j]); lo = hi - [d == 0, d == 1]
        tail, head = (lo, hi) if flow > 0 else (hi, lo)
        result.append((rid, tail, head, bool(np.isfinite(flow) and flow != 0)))
    return result


class Editor:
    def __init__(self, grid, output, rivers=None, vmin=0, vmax=15, overwrite=False):
        self.grid, self.output = grid, Path(output).expanduser().resolve()
        if self.output == grid.source:
            raise ValueError('Output must differ from input.')
        self.overwrite = overwrite
        self.mode, self.tool = 'Inspect', 'Point'
        self.contour = self.outline = None
        self.river_artists = []
        self.fig = plt.figure(figsize=(13, 8))
        self.ax = self.fig.add_axes([.07, .13, .65, .80])
        self.ax.set(xlabel='i (zero-based)', ylabel='j (zero-based)')
        self.image = self.ax.imshow(grid.depth, origin='lower', cmap='viridis_r',
                                    vmin=vmin, vmax=vmax, interpolation='nearest')
        self.overlay = self.ax.imshow(self.rgba(), origin='lower', interpolation='nearest')
        self.fig.colorbar(self.image, cax=self.fig.add_axes([.735, .28, .017, .55]),
                          label='Depth (m)')
        self.status = self.fig.text(.07, .035, '', fontsize=9)
        self.position = self.fig.text(.07, .075, 'Select a tool; pan/zoom with toolbar.', fontsize=10)
        self.controls = []
        self.radio = RadioButtons(self.fig.add_axes([.80, .65, .19, .25]),
            ['Inspect', 'Toggle mask', 'Set land', 'Set sea', 'Set depth'])
        self.radio.on_clicked(self.set_mode)
        self.tools = RadioButtons(self.fig.add_axes([.80, .52, .19, .11]), ['Point', 'Area'])
        self.tools.on_clicked(self.set_tool)
        self.depth_box = TextBox(self.fig.add_axes([.87, .465, .11, .035]), 'Depth (m) ', initial='5')
        self.checks = CheckButtons(self.fig.add_axes([.80, .34, .19, .11]),
                                  ['Rivers', 'Contours', 'Mask'], [True, True, True])
        self.checks.on_clicked(self.toggle)
        self.fig.text(.80, .32, 'Min depth (m)', fontsize=9)
        self.fig.text(.895, .32, 'Max depth (m)', fontsize=9)
        self.min_box = TextBox(self.fig.add_axes([.80, .28, .085, .035]), '',
                               initial=f'{vmin:g}')
        self.max_box = TextBox(self.fig.add_axes([.895, .28, .085, .035]), '',
                               initial=f'{vmax:g}')
        self.min_box.on_submit(self.apply_range)
        self.max_box.on_submit(self.apply_range)
        self.range_button = Button(self.fig.add_axes([.80, .235, .18, .035]),
                                   'Apply range')
        self.range_button.on_clicked(self.apply_range)
        for label, rect, callback in [
            ('Undo', [.80, .195, .085, .04], lambda e: self.action('undo')),
            ('Redo', [.895, .195, .085, .04], lambda e: self.action('redo')),
            ('Revert', [.80, .14, .085, .04], lambda e: self.action('revert')),
            ('Save', [.895, .14, .085, .04], self.save),
            ('Home', [.80, .085, .18, .04], self.home)]:
            button = Button(self.fig.add_axes(rect), label)
            button.on_clicked(callback); self.controls.append(button)
        self.selector = RectangleSelector(self.ax, self.area, button=[1],
                                          useblit=True, minspanx=3, minspany=3,
                                          spancoords='pixels', interactive=False)
        self.selector.set_active(False)
        self.fig.canvas.mpl_connect('button_press_event', self.click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.hover)
        self.fig.canvas.mpl_connect('close_event', self.close)
        if rivers:
            try:
                for rid, tail, head, directed in read_rivers(rivers, grid.mask.shape):
                    face = (tail + head) / 2
                    if directed:
                        self.river_artists.extend(self.ax.plot(*np.array([tail, head]).T, 'r-', lw=1.5, zorder=6))
                        self.river_artists.extend(self.ax.plot(*tail, 'ro', ms=6, mfc='none', zorder=6))
                        self.river_artists.extend(self.ax.plot(*head, 'r*', ms=10, zorder=6))
                    else:
                        self.river_artists.extend(self.ax.plot(*face, 'rx', ms=8, zorder=6))
                    self.river_artists.append(self.ax.text(*(face + .15), f'{rid:g}',
                                                          color='r', fontsize=9, clip_on=True, zorder=7))
            except Exception as exc:
                warnings.warn(f'River overlay not loaded: {exc}')
        self.home()
        self.refresh('Ready. Save writes: ' + str(self.output))

    def rgba(self):
        ny, nx = self.grid.mask.shape
        checker = np.indices((ny, nx)).sum(axis=0) % 2
        colors = np.array([[1, .9254, .5451], [.8039, .7451, .4392],
                           [.2509, .8784, 1], [.95, .97, 1]])
        indices = self.grid.mask.astype(int) * 2 + checker
        return np.dstack((colors[indices], np.where(self.grid.mask == 0, .35, .08)))

    def navigation(self):
        toolbar = getattr(self.fig.canvas, 'toolbar', None)
        return bool(toolbar and toolbar.mode)

    def set_mode(self, mode):
        self.mode = mode
        self.selector.set_active(self.tool == 'Area' and mode in ['Toggle mask', 'Set land', 'Set sea'])
        self.status.set_text('Set depth: enter meters, then click one cell.' if mode == 'Set depth'
                             else f'{mode}: {self.tool.lower()} selection.')
        self.fig.canvas.draw_idle()

    def set_tool(self, tool):
        self.tool = tool
        self.set_mode(self.mode)

    def cell(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return None
        i, j = int(np.floor(event.xdata + .5)), int(np.floor(event.ydata + .5))
        ny, nx = self.grid.mask.shape
        return (j, i) if 0 <= i < nx and 0 <= j < ny else None

    def hover(self, event):
        cell = self.cell(event)
        if cell is not None:
            j, i = cell
            self.position.set_text(f'i={i}, j={j}    h={self.grid.depth[j, i]:.4g} m    '
                                   f'mask={int(self.grid.mask[j, i])}')
            self.fig.canvas.draw_idle()

    def click(self, event):
        if event.button != 1 or self.navigation() or self.cell(event) is None:
            return
        j, i = self.cell(event)
        if self.mode == 'Inspect':
            self.hover(event)
        elif self.mode == 'Set depth' or self.tool == 'Point':
            self.edit(np.array([j * self.grid.mask.shape[1] + i]))

    def area(self, start, end):
        if self.navigation() or self.mode not in ['Toggle mask', 'Set land', 'Set sea']:
            return
        ny, nx = self.grid.mask.shape
        i0, i1 = sorted([start.xdata, end.xdata]); j0, j1 = sorted([start.ydata, end.ydata])
        ii = np.arange(max(0, int(np.ceil(i0))), min(nx-1, int(np.floor(i1)))+1)
        jj = np.arange(max(0, int(np.ceil(j0))), min(ny-1, int(np.floor(j1)))+1)
        if ii.size and jj.size:
            self.edit((jj[:, None] * nx + ii).ravel())

    def edit(self, indices):
        try:
            if self.mode == 'Set depth':
                value = float(self.depth_box.text)
                if not np.isfinite(value) or value <= 0:
                    raise ValueError('Depth must be a finite positive number in meters.')
                self.grid.change('depth', indices[:1], value)
            elif self.mode in ['Toggle mask', 'Set land', 'Set sea']:
                old = self.grid.mask.flat[indices]
                values = 1-old if self.mode == 'Toggle mask' else int(self.mode == 'Set sea')
                self.grid.change('mask', indices, values)
            self.refresh(f'{self.mode}: {len(indices)} cell(s).')
        except ValueError as exc:
            self.status.set_text(str(exc)); self.fig.canvas.draw_idle()

    def refresh(self, message=''):
        self.image.set_data(self.grid.depth)
        self.overlay.set_data(self.rgba())
        limits = self.ax.get_xlim(), self.ax.get_ylim()
        if self.contour is not None:
            self.contour.remove(); self.contour = None
        if self.outline is not None:
            self.outline.remove(); self.outline = None
        if self.checks.get_status()[1]:
            low, high = self.image.get_clim()
            self.contour = self.ax.contour(self.grid.depth, levels=np.linspace(low, high, 9),
                                           colors='.25', linewidths=.45, zorder=3)
        if np.any(self.grid.mask == 0) and np.any(self.grid.mask == 1):
            self.outline = self.ax.contour(self.grid.mask, levels=[.5], colors='k',
                                           linewidths=.7, zorder=4)
        self.ax.set_xlim(limits[0]); self.ax.set_ylim(limits[1])
        self.ax.set_title('ROMS mask / bathymetry editor' + (' — unsaved changes' if self.grid.dirty else ''))
        if message:
            self.status.set_text(message)
        self.fig.canvas.draw_idle()

    def toggle(self, label):
        river, contours, mask = self.checks.get_status()
        for artist in self.river_artists:
            artist.set_visible(river)
        self.overlay.set_visible(mask)
        self.refresh()

    def apply_range(self, event=None):
        try:
            lo, hi = float(self.min_box.text), float(self.max_box.text)
            if not np.isfinite([lo, hi]).all() or hi <= lo:
                raise ValueError
            self.image.set_clim(lo, hi)
            self.refresh(f'Highlighting {lo:g} to {hi:g} m; grid depths unchanged.')
        except ValueError:
            self.status.set_text('Enter finite depths with maximum greater than minimum. Range unchanged.')
            self.fig.canvas.draw_idle()

    def action(self, name):
        getattr(self.grid, name)()
        self.refresh(name.capitalize() + ' complete.')

    def home(self, event=None):
        ny, nx = self.grid.mask.shape
        self.ax.set_xlim(-.5, nx-.5); self.ax.set_ylim(-.5, ny-.5)
        self.fig.canvas.draw_idle()

    def save(self, event=None):
        try:
            backup = self.grid.save(self.output, allow_overwrite=self.overwrite)
            self.overwrite = True
            message = f'Saved {self.output.name}'
            if backup:
                message += f'; backup: {backup.name}'
            self.refresh(message)
        except Exception as exc:
            self.status.set_text(f'Save failed: {exc}')
            self.fig.canvas.draw_idle()

    def close(self, event):
        if self.grid.dirty:
            warnings.warn('Closed with unsaved edits. No grid file was changed by closing.')


def launch(grid_file, rivers=None, output=None, vmin=0, vmax=15, overwrite=False):
    if not np.isfinite([vmin, vmax]).all() or vmax <= vmin:
        raise ValueError('Display maximum must exceed minimum.')
    grid = Grid(grid_file)
    if output is None:
        output = grid.source.with_name(grid.source.stem + '_edited.nc')
    editor = Editor(grid, output, rivers, vmin, vmax, overwrite)
    # Retain widgets even when launch() is called without assigning its result.
    editor.fig._roms_editor = editor
    plt.show()
    return editor


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('grid', help='ROMS NetCDF grid with h and mask_rho')
    parser.add_argument('--rivers', help='Optional ROMS river forcing NetCDF file')
    parser.add_argument('--output', help='Output grid; default <input>_edited.nc')
    parser.add_argument('--vmin', type=float, default=0)
    parser.add_argument('--vmax', type=float, default=15)
    parser.add_argument('--overwrite-output', action='store_true', help='Allow replacing existing output (with backup)')
    args = parser.parse_args()
    launch(args.grid, args.rivers, args.output, args.vmin, args.vmax, args.overwrite_output)
