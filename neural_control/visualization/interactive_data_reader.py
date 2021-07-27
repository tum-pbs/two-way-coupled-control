import json
import time
from re import U
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
from tkinter import *
from tkinter import font
import os
from natsort import natsorted
from itertools import chain
from Plotter import Plotter
from torch.nn.functional import pad


class Interactive_Data_Reader(Tk):
    def __init__(self, log_path: str, window_size: str):
        """
        Initialize GUI

        Params:
            log_path: path to log file that contain information of previous runs
            window_size: size of GUI window
        """
        super(Interactive_Data_Reader, self).__init__()
        self.log_variables = ["dataPath", "sModel", "sVar", "sAxis", "sPlotType", "sKwargs"]
        self.title("Interactive_Data_Reader")
        self.nAxis = 6
        self.nMaxLoadedVariables = 30
        self.minsize(500, 400)
        self.geometry(window_size)
        # String vars
        self.sModel = StringVar(self, " ")
        self.sTest = StringVar(self, " ")
        self.dataPath = StringVar(self, "/home/ramos/work/PhiFlow2/PhiFlow/storage/")
        self.sVar = [StringVar(self, " ") for _ in range(self.nMaxLoadedVariables)]
        self.sKwargs = [StringVar(self, " ") for _ in range(self.nMaxLoadedVariables)]
        self.sAxis = [StringVar(self, " ") for _ in range(self.nMaxLoadedVariables)]
        self.plot_options = ["imshow", "plot", "probes", "plot", "arrows", "point", "torque"]
        self.sPlotType = [StringVar(self, " ") for _ in range(self.nMaxLoadedVariables)]
        # Other variables
        self.plotHandle = [[]] * self.nMaxLoadedVariables
        self.last_update = {'time': time.time(), 'mouse_x': self.winfo_pointerx()}
        self.plotter = Plotter()
        self.set_default()
        self.create_layout()
        self.log_path = log_path
        self.load_log()
        self.bind_functions()
        self.update_solutions()
        self.refresh_plots()

    def check_time(func):
        """
        Decorator that avoid too frequent updates

        """

        def wrapper_time(self, *args, **kwargs):
            delta = time.time() - self.last_update['time']
            if delta < .1: return
            self.last_update['time'] = time.time()
            func(self, *args, **kwargs)
        wrapper_time.__wrapped__ = func
        return wrapper_time

    def set_default(self):
        """
        Override tkinter default values/options

        """
        # Creating a Font object of "TkDefaultFont"
        defaultFont = font.nametofont("TkDefaultFont")
        # Overriding default-font with custom settings
        # i.e changing font-family, size and weight
        defaultFont.configure(size=15)

    def load_log(self):
        """
        Load log file containing information of previous runs

        """
        if os.path.isfile(self.log_path + "/log.json"):
            with open(self.log_path + "/log.json", "r") as f:
                try:
                    inputs = json.load(f)
                except json.JSONDecodeError:
                    print("Could not load json file")
                    pass
                    return
                for attribute in self.log_variables:
                    value = getattr(self, attribute)
                    try:
                        if isinstance(value, list):
                            for i in range(len(value)):
                                value[i].set(inputs[attribute][i])
                        else:
                            value.set(inputs[attribute])
                    except:
                        print(f"Could not load {attribute}")
                        pass
            self.update_solutions()

    def save_log(self):
        """
        Save information to log file

        """
        with open(self.log_path + "/log.json", "w") as f:
            export_dict = {}
            for attribute in self.log_variables:
                value = getattr(self, attribute)
                export_dict[attribute] = value.get() if not isinstance(value, list) else [v.get() for v in value]
            json.dump(export_dict, f, indent="    ")

    def create_layout(self):
        """
        Create the widgets layout

        """
        # Canva for input stuff
        clicks = Canvas(self, width=300, height=500)
        # clicks.config(highlightbackground="black")
        clicks.pack(side=LEFT, expand=True, padx=30, pady=30)
        slides = Canvas(self, width=300, height=500)
        slides.pack(side=BOTTOM, expand=True)
        # Entry for datapath
        self.eEntry = Entry(clicks, textvariable=self.dataPath, width=100)
        clicks.create_window(200, 200, window=self.eEntry)
        self.eEntry.grid(columnspan=5, row=0)
        # Button for updating filepaths
        self.bUpdatePaths = Button(clicks, text="Update paths")
        self.bUpdatePaths.grid(row=2, column=0)
        # Button for updating matplotlib plot
        self.bUpdatePlot = Button(clicks, text="Update plot")
        self.bUpdatePlot.config(bg="#cf5f5f")
        self.bUpdatePlot.grid(row=2, column=1)
        # Button for updating/reloading matplotlib plot
        self.bReloadVariables = Button(clicks, text="Reload plots")
        self.bReloadVariables.grid(row=2, column=3)
        # Button for updating/reloading matplotlib plot
        self.bToggleRefresh = Button(clicks, text="Refresh all")
        self.bToggleRefresh.grid(row=2, column=2)
        # Simulation selector
        self.oModelSelector = OptionMenu(clicks, self.sModel, " ")
        self.oModelSelector.config(width=20)
        self.oModelSelector.grid(column=0, row=3)
        # Test selector
        self.oTestSelector = OptionMenu(clicks, self.sTest, " ")
        self.oTestSelector.config(width=20)
        self.oTestSelector.grid(column=1, row=3)
        # Checkbutton for test data
        self.iUseTestData = IntVar()
        self.iUseTestData.set(0)
        checkButton = Checkbutton(clicks, variable=self.iUseTestData, text="Test Data")
        checkButton.grid(column=2, row=3)
        # Overal widgets that are present in more than one line
        self.oVarSelector = []
        self.eKwargs = []
        self.iUpdatePlot = []
        self.oAxisSelector = []
        self.oPlotType = []
        self.bRefresh = []
        for i in range(self.nMaxLoadedVariables):
            self.oVarSelector += [OptionMenu(clicks, self.sVar[i], " ")]
            self.oVarSelector[i].config(width=20)
            self.oVarSelector[i].grid(column=1, row=4 + i)
            self.iUpdatePlot += [IntVar()]
            self.iUpdatePlot[i].set(1)
            checkButton = Checkbutton(clicks, variable=self.iUpdatePlot[i])
            checkButton.grid(column=2, row=i + 4)
            self.bRefresh += [Button(clicks, text='Refresh')]
            self.oAxisSelector += [OptionMenu(clicks, self.sAxis[i], *[str(x) for x in range(self.nAxis)])]
            self.oAxisSelector[i].grid(column=3, row=i + 4)
            self.bRefresh[i].grid(row=i + 4, column=4)
            self.oPlotType += [OptionMenu(clicks, self.sPlotType[i], *self.plot_options)]
            self.oPlotType[i].grid(row=i + 4, column=5, columnspan=1)
            self.eKwargs += [Entry(clicks, textvariable=self.sKwargs[i], width=20)]
            self.eKwargs[i].grid(row=i + 4, column=6, columnspan=1)
        # Slider to choose simulation case
        # label.place(x=20)
        local_canvas1 = Canvas(slides)
        self.sCaseSelector = Scale(local_canvas1, orient=HORIZONTAL, length=450)
        self.bNextCase = Button(local_canvas1, text=">")
        self.bPreviousCase = Button(local_canvas1, text="<")
        self.bPreviousCase.pack(side=LEFT, )
        self.sCaseSelector.pack(side=LEFT, )
        self.bNextCase.pack(side=LEFT, )
        # Slider for different solutions
        local_canvas2 = Canvas(slides)
        self.sSnapshotSelector = Scale(local_canvas2, orient=HORIZONTAL, length=450)
        self.bNextSnapshot = Button(local_canvas2, text=">")
        self.bPreviousSnapshot = Button(local_canvas2, text="<")
        self.bPreviousSnapshot.pack(side=LEFT, )
        self.sSnapshotSelector.pack(side=LEFT, )
        self.bNextSnapshot.pack(side=LEFT, )
        label = Label(slides, text="Case ")
        label.grid(row=0, column=0)
        label = Label(slides, text="Snapshot ")
        label.grid(row=0, column=1)
        local_canvas1.grid(row=1, column=0, padx=30)
        local_canvas2.grid(row=1, column=1, padx=30)
        # Slider for different solutions
        # Button for playing animations
        self.bPlay = Button(clicks, text="Play")
        self.bPlay.grid(row=i + 1 + 7, column=0)
        # Button for stop playing animations
        self.bStop = Button(clicks, text="Stop")
        self.bStop.grid(row=i + 1 + 7, column=1)
        # Button for skipping snapshots when playing
        self.ePlaySkip = Entry(clicks, text="Skip")
        self.ePlaySkip.grid(row=i + 1 + 7, column=2)
        # Export check
        self.iExport = IntVar()
        self.iExport.set(0)
        cExport = Checkbutton(clicks, text="Export")
        cExport.grid(row=i + 1 + 7, column=3)
        # Matplotlib stuff
        self.fig, self.axes = plt.subplots(int(self.nAxis / 2), int(2), figsize=(10, 10))
        self.axes = self.axes.reshape(-1)
        self.fig.tight_layout()
        self.display = FigureCanvasTkAgg(self.fig, master=self)  # A tk.DrawingArea.
        self.display.draw()
        self.toolbar = NavigationToolbar2Tk(self.display, self, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side=BOTTOM, fill=X)
        self.display.get_tk_widget().pack(side=RIGHT, expand=True)
        # Hack so that I can use plotter class
        for i in range(self.nAxis):
            self.plotter.check_plot_id(f"plot{i}", constrained_layout=False)
            self.plotter.figs[f"plot{i}"]["ax"] = self.axes[i]

    def bind_functions(self):
        """
        Bind widgets functionalities to the proper functions

        """
        self.bUpdatePaths.bind("<Button-1>", self.update_solutions)
        self.bUpdatePlot.bind("<Button-1>", self.toggle)
        self.bReloadVariables.bind("<Button-1>", self.reset_plot)
        self.sSnapshotSelector.configure(command=self.update_snapshot)
        self.bNextSnapshot.bind("<Button-1>", self.next_snapshot)
        self.bPreviousSnapshot.bind("<Button-1>", self.previous_snapshot)
        self.sCaseSelector.bind("<ButtonRelease-1>", self.update_case)
        self.bNextCase.bind("<Button-1>", self.next_case)
        self.bPreviousCase.bind("<Button-1>", self.previous_case)
        self.sModel.trace("w", lambda *args: self.reset_plot())
        for i, (var, kwargs, axis, plotType, refresh) in enumerate(zip(self.sVar, self.sKwargs, self.sAxis, self.sPlotType, self.bRefresh)):
            var.trace("w", lambda *args, i=i: self.reset_plot(i=i))
            kwargs.trace("w", lambda *args, i=i: self.reset_plot(i=i))
            axis.trace("w", lambda *args, i=i: self.reset_plot(i=i))
            plotType.trace("w", lambda *args, i=i: self.reset_plot(i=i))
            refresh.bind("<Button-1>", lambda *args, i=i: self.reset_plot(i=i))
        self.bPlay.configure(command=self.play)
        self.bStop.configure(command=self.stop)
        self.bToggleRefresh.bind("<Button-1>", self.toggle_refresh)

    def toggle_refresh(self, event=None):
        """
        Toggle all refreshs

        """
        for value in self.iUpdatePlot:
            value.set(1)

    def stop(self, event=None):
        """
        Stop animation

        """
        self.isPlaying = False

    def next_snapshot(self, event=None):
        """
        Select next snapshot

        """
        i = self.sSnapshotSelector.get()
        self.sSnapshotSelector.set(i + 1)
        self.update_case()

    def next_case(self, event=None):
        """
        Select next case

        """
        i = self.sCaseSelector.get()
        self.sCaseSelector.set(i + 1)
        self.update_case()

    def previous_snapshot(self, event=None):
        """
        Select previous snapshot

        """
        i = self.sSnapshotSelector.get()
        self.sSnapshotSelector.set(i - 1)
        self.update_case()

    def previous_case(self, event=None):
        """
        Select previous case

        """
        i = self.sCaseSelector.get()
        self.sCaseSelector.set(i - 1)
        self.update_case()

    def play(self, event=None):
        """ "
        Play animation

        """
        self.isPlaying = True
        maximum = self.sSnapshotSelector.cget("to")
        current = int(self.sSnapshotSelector.get())
        if current >= maximum:
            current = self.sSnapshotSelector.cget("from")
        while current <= maximum and self.isPlaying:
            self.sSnapshotSelector.set(current)
            self.update_snapshot.__wrapped__(self)
            current += int(self.ePlaySkip.get())
            if self.iExport:
                path = f'{self.dataPath.get()}/movies/'
                if not os.path.exists(f'{path}'): os.mkdir(path)
                self.fig.savefig(f'{path}{self.sSnapshotSelector.get()}.png', facecolor='w', edgecolor='w', dpi=200)

    def update_solutions(self, event=None):
        """
        Get list of solutions contained in dataPath and update dropwdowns

        """

        def update_dropdown(dropdown, stringvar, newentries):
            menu = dropdown["menu"]
            menu.delete(0, "end")
            for string in newentries:
                menu.add_command(label=string, command=lambda value=string: stringvar.set(value))

        path = self.dataPath.get()
        # Models
        models = os.listdir(path)
        models = [file for file in models if "." not in file]  # Get folders only
        models = natsorted(models, key=lambda x: x.lower())
        update_dropdown(self.oModelSelector, self.sModel, models)
        if self.sModel.get() not in models:
            self.sModel.set(models[0])
        # Tests
        if os.path.exists(f"{path}/{self.sModel.get()}/tests/"):
            tests = os.listdir(f"{path}/{self.sModel.get()}/tests/")
        else: tests = ['']
        update_dropdown(self.oTestSelector, self.sTest, tests)
        if self.sTest.get() not in tests:
            self.sTest.set(tests[0])
        # Data folder
        if self.iUseTestData.get():
            self.datafolder = f"/tests/{self.sTest.get()}/data/"
        else:
            self.datafolder = "/data/"
        variables = os.listdir(path + "/" + self.sModel.get() + self.datafolder)
        variables = natsorted(list(set([file.split("_case")[0] for file in variables])))
        for var, oVarSelector, checkBox in zip(self.sVar, self.oVarSelector, self.iUpdatePlot):
            if var.get() not in variables:
                checkBox.set(0)
            update_dropdown(oVarSelector, var, variables)
        # Update cases selector dropdown
        allCases = os.listdir(path + "/" + self.sModel.get() + self.datafolder + variables[0])
        cases = natsorted(list(set([file.split("case")[1][:4] for file in allCases])))
        snapshots = natsorted([int(case.split("_")[-1][: 4]) for case in natsorted(allCases)
                               if f"{variables[0]}_case{self.sCaseSelector.get():04d}" in case])
        # Update axis selector dropdown
        for selector in self.sAxis:
            if selector.get() == " ":
                selector.set("0")
        # Update sliders
        try:
            self.sCaseSelector.configure(to=int(cases[-1]))
            self.sCaseSelector.configure(from_=int(cases[0]))
            self.sSnapshotSelector.configure(to=snapshots[-1])
            self.sSnapshotSelector.configure(from_=snapshots[0])
            self.save_log()
        except:
            print("Could not get variables for slides correctly")
            pass
        # self.refresh_plots()

    def toggle(self, event):
        """
        Toggle between update plot or not

        """
        # Toggle button functionality
        button = event.widget
        if button.config("bg")[-1] == "#68bf21":
            button.config(bg="#cf5f5f")
        else:
            button.config(bg="#68bf21")
            self.automatic_plot_reset()

    def automatic_plot_reset(self, *args):
        """
        Function that keeps refreshing plots

        """
        if self.bUpdatePlot.config("bg")[-1] == "#68bf21":
            self.sCaseSelector.set(self.sCaseSelector.cget("to"))
            self.reset_plot()
            self.play()
            self.after(2000, self.automatic_plot_reset)

    def reset_plot(self, event=None, i=None):
        """
        Reset plots

        """
        self.update_solutions()
        # Get all variables for this plot
        # Remove colorbars
        if i is None:
            for i, ax in enumerate(self.axes):
                for image in ax.images:
                    if image:
                        image.colorbar.remove()
            plt.tight_layout()
            # Remove lines and images
            self.plotHandle = [[] for _ in self.plotHandle]
            # Clear axes
            for ax in self.axes:
                ax.clear()
            self.refresh_plots()
        else:
            ids = [j for j, axis in enumerate(self.sAxis) if axis.get() == self.sAxis[i].get()]
            for id in ids:
                axis = self.axes[int(self.sAxis[id].get())]
                for obj in axis.get_children():
                    if obj in self.plotHandle[id]:
                        try:
                            obj.colorbar.remove()
                        except:
                            pass
                        obj.remove()
                        del obj
                self.plotHandle[id] = []
            axis.clear()
            # plt.tight_layout()
            self.refresh_plots(i=ids)

    @ check_time
    def update_snapshot(self, event=None):
        """
        Update plots according to current snapshot

        """
        path = self.dataPath.get()
        snapshot = self.sSnapshotSelector.get()
        model = self.sModel.get()
        for i in range(self.nMaxLoadedVariables):
            if not self.iUpdatePlot[i].get():
                continue
            var = self.sVar[i].get()
            case = self.sCaseSelector.get()
            filepath = f"{path}/{model}/{self.datafolder}/{var}/{var}_case{case:04d}_{snapshot:04d}.npy"
            if not os.path.exists(filepath): continue  # Make sure file exists
            plotType = self.sPlotType[i].get()
            kwargs = {}
            strings = self.sKwargs[i].get().split(", ")
            for string in strings:
                if string == " " or string == "":
                    continue
                kwargs[string.split("=")[0].strip()] = eval(string.split("=")[1].strip())
            if plotType == "imshow":
                data = np.load(filepath)
                self.plotHandle[i][0].set_data(data.transpose())
            if plotType == "plot":
                data = np.load(filepath)
                if data.size == 2:
                    data = [[value] for value in data]
                elif data.size == 1:
                    data = [[snapshot], [data]]
                else:
                    print(f"Invalid plot for variable {i}")
                    continue
                self.plotHandle[i][0].set_data(data)
            if plotType == 'point':
                vars_set = {'x', 'y'}
                if not (vars_set <= kwargs.keys()):
                    continue
                x = np.load(filepath)
                y = np.load(f"{path}/{model}/{self.datafolder}/{kwargs['y']}/{kwargs['y']}_case{case:04d}_{snapshot:04d}.npy")
                data = [x, y][::1 - 2 * (kwargs['x'] == False)]  # check if order should be inverted
                kwargs.pop('x', None)
                kwargs.pop('y', None)
                self.plotHandle[i][0].set_data(data)
            if plotType == "probes":
                data = np.load(filepath)
                self.plotHandle[i][0].set_data(data)
            if plotType == "arrows":
                vars_set = {'u', 'v'}
                if not (vars_set <= kwargs.keys()):
                    continue
                data_xy = np.load(filepath)
                data_u = np.load(f"{path}/{model}/{self.datafolder}/{kwargs['u']}/{kwargs['u']}_case{case:04d}_{snapshot:04d}.npy")
                data_v = np.load(f"{path}/{model}/{self.datafolder}/{kwargs['v']}/{kwargs['v']}_case{case:04d}_{snapshot:04d}.npy")
                self.plotHandle[i][0].set_UVC(data_u, data_v)
                self.plotHandle[i][0].set_offsets(data_xy)
            # Torque
            if plotType == "torque":
                vars_set = {'torque', 'offset', 'angle'}
                if not (vars_set <= kwargs.keys()):
                    continue
                data_xy = np.load(filepath)
                angle = np.load(f"{path}/{model}/{self.datafolder}/{kwargs['angle']}/{kwargs['angle']}_case{case:04d}_{snapshot:04d}.npy")
                torque = np.load(f"{path}/{model}/{self.datafolder}/{kwargs['torque']}/{kwargs['torque']}_case{case:04d}_{snapshot:04d}.npy")
                offset = kwargs['offset']
                anchor_x = np.array([np.cos(-angle) * offset, -np.cos(-angle) * offset])
                anchor_y = np.array([np.sin(-angle) * offset, -np.sin(-angle) * offset])
                data_v = [np.cos(angle) * torque, -np.cos(angle) * torque]
                data_u = [np.sin(angle) * torque, -np.sin(angle) * torque]
                anchor_x += data_xy[0]
                anchor_y += data_xy[1]
                kwargs.pop('torque', None)
                kwargs.pop('offset', None)
                kwargs.pop('angle', None)
                self.plotHandle[i][0].set_UVC(data_u, data_v)
                new_xy = ((anchor_x[0, 0], anchor_y[0, 0]), (anchor_x[1, 0], anchor_y[1, 0]))
                self.plotHandle[i][0].set_offsets(new_xy)
        self.display.draw()
        self.update()

    @ check_time
    def update_case(self, event=None):
        """
        Update plots based on case

        """
        path = self.dataPath.get()
        snapshot = self.sSnapshotSelector.get()
        model = self.sModel.get()
        for i in range(self.nMaxLoadedVariables):
            if not self.iUpdatePlot[i].get():
                continue
            var = self.sVar[i].get()
            case = self.sCaseSelector.get()
            plotType = self.sPlotType[i].get()
            filepath = f"{path}/{model}/{self.datafolder}/{var}/{var}_case{case:04d}_{snapshot:04d}.npy"
            if not os.path.exists(filepath): continue  # Make sure file exists
            kwargs = {}
            strings = self.sKwargs[i].get().split(", ")
            for string in strings:
                if string == " " or string == "":
                    continue
                kwargs[string.split("=")[0].strip()] = eval(string.split("=")[1].strip())
            # Update lines
            if plotType == "plot":
                snapshots = np.arange(int(self.sSnapshotSelector.cget("from")), int(self.sSnapshotSelector.cget("to")) + 1)
                data = np.ones((2, int(snapshots[-1] - snapshots[0] + 1))) * snapshots
                for j, snapshot_local in enumerate(snapshots):
                    filepath = f"{path}/{model}/{self.datafolder}/{var}/{var}_case{case:04d}_{int(snapshot_local):04d}.npy"
                    loaded_data = np.load(filepath)
                    data[2 - loaded_data.size:, j] = loaded_data
                self.plotHandle[i][0].set_data(data)
                self.plotHandle[i][1].set_data(data)
        self.update_snapshot()

    def refresh_plots(self, event=None, i=None):
        """
        Update matplotlib plot with parameters from GUI

        """
        if i:
            if isinstance(i, (list, tuple)):
                loop = i
            else:
                loop = [i]
        else:
            loop = range(self.nMaxLoadedVariables)
        path = self.dataPath.get()
        snapshot = self.sSnapshotSelector.get()
        model = self.sModel.get()
        for i in loop:
            if not self.iUpdatePlot[i].get():
                continue
            var = self.sVar[i].get()
            case = self.sCaseSelector.get()
            axis = self.sAxis[i].get()
            plotType = self.sPlotType[i].get()
            filepath = f"{path}/{model}/{self.datafolder}/{var}/{var}_case{case:04d}_{snapshot:04d}.npy"
            kwargs = {}
            strings = self.sKwargs[i].get().split(", ")
            try:
                # Kwargs
                for string in strings:
                    if string == " " or string == "":
                        continue
                    kwargs[string.split("=")[0].strip()] = eval(string.split("=")[1].strip())
                # Imshow
                if plotType == "imshow":
                    data = np.load(filepath)
                    self.plotter.add_data([data], [var])
                    image, *_ = self.plotter.imshow([var], f"plot{axis}", **kwargs)
                    self.plotHandle[i] = [image]
                # Plot
                if plotType == "plot":
                    data = np.load(filepath)
                    if data.size == 2:
                        data = [[value] for value in data]
                        should_create_grid = False
                    elif data.size == 1:
                        data = [[snapshot], [data]]
                        should_create_grid = True
                    else:
                        print(f"Invalid plot for variable {i}")
                        continue
                    self.plotter.add_data([data], [var])
                    dot_handle, *_ = self.plotter.plot([var], f"plot{axis}", marker="o", color="black", create_legend=False)  # Plot dot for current snapshot
                    snapshots = np.arange(int(self.sSnapshotSelector.cget("from")), int(self.sSnapshotSelector.cget("to")) + 1)
                    data = np.ones((2, int(snapshots[-1] - snapshots[0] + 1))) * snapshots
                    for j, snapshot_local in enumerate(snapshots):
                        filepath = f"{path}/{model}/{self.datafolder}/{var}/{var}_case{case:04d}_{int(snapshot_local):04d}.npy"
                        loaded_data = np.load(filepath)
                        data[2 - loaded_data.size:, j] = loaded_data
                    self.plotter.add_data([data], [var])
                    lines_handle, *_ = self.plotter.plot([var], f"plot{axis}", create_grid=should_create_grid, **kwargs)
                    self.plotHandle[i] = [*dot_handle, *lines_handle]
                # Point
                if plotType == "point":
                    vars_set = {'x', 'y'}
                    if not (vars_set <= kwargs.keys()):
                        continue
                    x = np.load(filepath)
                    y = np.load(f"{path}/{model}/{self.datafolder}/{kwargs['y']}/{kwargs['y']}_case{case:04d}_{snapshot:04d}.npy")
                    data = [[x], [y]][::1 - 2 * (kwargs['x'] == False)]  # check if order should be inverted
                    kwargs.pop('x', None)
                    kwargs.pop('y', None)
                    self.plotter.add_data([data], [var])
                    dot_handle = self.plotter.plot([var], f"plot{axis}", create_grid=False, create_legend=False, marker="o", color="tab:gray", ** kwargs)
                    self.plotHandle[i], *_ = dot_handle
                # Probes
                if plotType == "probes":
                    data = np.load(filepath)
                    self.plotter.add_data([data], [var])
                    self.plotHandle[i], *_ = self.plotter.plot([var], f"plot{axis}", marker="x",
                                                               color="black", linestyle='None', create_legend=False)  # Plot crosses on probes points
                # Arrows
                if plotType == "arrows":
                    vars_set = {'u', 'v'}
                    if not (vars_set <= kwargs.keys()):
                        continue
                    data_xy = np.load(filepath)
                    data_u = np.load(f"{path}/{model}/{self.datafolder}/{kwargs['u']}/{kwargs['u']}_case{case:04d}_{snapshot:04d}.npy")
                    data_v = np.load(f"{path}/{model}/{self.datafolder}/{kwargs['v']}/{kwargs['v']}_case{case:04d}_{snapshot:04d}.npy")
                    kwargs.pop('u', None)
                    kwargs.pop('v', None)
                    self.plotHandle[i] = [self.axes[int(axis)].quiver(*data_xy, data_u, data_v, **kwargs)]
                # Torque
                if plotType == "torque":
                    vars_set = {'torque', 'offset', 'angle'}
                    if not (vars_set <= kwargs.keys()):
                        continue
                    data_xy = np.load(filepath)
                    angle = np.load(f"{path}/{model}/{self.datafolder}/{kwargs['angle']}/{kwargs['angle']}_case{case:04d}_{snapshot:04d}.npy")
                    torque = np.load(f"{path}/{model}/{self.datafolder}/{kwargs['torque']}/{kwargs['torque']}_case{case:04d}_{snapshot:04d}.npy")
                    offset = kwargs['offset']
                    anchor_x = np.array([np.cos(-angle) * offset, -np.cos(-angle) * offset])
                    anchor_y = np.array([np.sin(-angle) * offset, -np.sin(-angle) * offset])
                    data_v = [np.cos(angle) * torque, -np.cos(angle) * torque]
                    data_u = [np.sin(angle) * torque, -np.sin(angle) * torque]
                    anchor_x += data_xy[0]
                    anchor_y += data_xy[1]
                    kwargs.pop('torque', None)
                    kwargs.pop('offset', None)
                    kwargs.pop('angle', None)
                    self.plotHandle[i] = [self.axes[int(axis)].quiver(anchor_x, anchor_y, data_u, data_v, **kwargs)]
            except:
                print(f"Error for plot: {plotType}, var {i}: {var} ")
                self.iUpdatePlot[i].set(0)
            self.display.draw()
            self.update()


if __name__ == '__main__':
    # TODO create test
    log_path = "/home/ramos/work/PhiFlow2/PhiFlow/neural_control/visualization/"
    gui = Interactive_Data_Reader(log_path, "2600x1200")
    gui.mainloop()
