import argparse
from collections import OrderedDict, defaultdict
from itertools import cycle
import json
import os
from InputsManager import InputsManager
from natsort.natsort import natsorted
import numpy as np
# import plotly.express as px
# from PIL import ImageColor
# import plotly.graph_objects as go
# import plotly.io as pio
from Plotter import Plotter
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['text.usetex'] = True

hatches = cycle(['/', '\\', '|', '-', '+', 'x', '.'])
colors_hash = dict(
    Online='#1f77b4',
    PID='#ff7f0e',
    Supervised='#2ca02c',
    LS='#d62728',
    RL='#9467bd',
)
colors_hash['Seed 1'] = colors_hash['Online']
colors_hash['Seed 2'] = colors_hash['PID']
colors_hash['Seed 3'] = colors_hash['Supervised']
colors_hash['Seed 4'] = colors_hash['LS']
x_scale_factor = dict(
    test1=0.1,
    test2=0.05,
    test3=0.1,
    test4=0.05,
    test5=0.05,
    test6=0.05
)
# Load figs json
with open(os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/figs.json'), 'r') as f:
    figs_attrs = json.load(f)
# Update figs_attrs to contain only requested figs
plt.rcParams.update({'font.size': figs_attrs['global']['font_size']})
plt.rc('legend', fontsize=figs_attrs['global']['font_size_legend'])
parser = argparse.ArgumentParser(description='Plot metrics')
parser.add_argument('--figs', type=str, nargs='+', required=False, default=None)  # Default is to plot all figs
figs_to_plot = parser.parse_args().figs
if figs_to_plot is not None:
    for fig in list(figs_attrs.keys()):
        if fig in [*figs_to_plot, 'global']: continue
        else: figs_attrs.pop(fig)
# Add metrics data to plotter
loop_dict = {key: value for key, value in figs_attrs.items() if 'global' not in key}
for fig_name, attrs in loop_dict.items():
    stdd = {}
    p = Plotter((figs_attrs['global']['width'], figs_attrs['global']['height']))
    data_ids = []
    for run_label, run_path in attrs['runs'].items():
        # Pre process variables
        tests = natsorted([test for test in os.listdir(run_path + '/tests') if attrs['test'] in test])
        for test in tests:
            # If not requested, only last model will be plotted
            if not attrs.get('all_models'): model_id = ''
            else: model_id = test.split('_')[-1]
            datapath = f"{run_path}/tests/{test}/"
            try:
                with open(f"{datapath}/metrics.json", "r") as f:
                    data = json.load(f)
            except FileNotFoundError:
                print(f"\n\n Did not find metrics.json in {datapath} \n\n")
                continue
            metrics = defaultdict(lambda: 0)
            metrics.update(data)
            # Store standard deviation
            # Load metrics
            for params in attrs['metrics']:
                metric_name = params['name']
                value = np.squeeze(metrics[metric_name])
                label = f"{run_label}_{model_id}_/{metric_name}"
                if len(value.shape) == 1:
                    x = np.arange(value.shape[0]) * x_scale_factor[test.split("_")[0]]
                    value = [x, value]
                p.add_data([value], [label])
                stdd[label] = metrics[metric_name + "_stdd"]
                data_ids.append(label)
    # Get rid of possible duplicates
    data_ids = natsorted(list(set(data_ids)))
    # Assign colors to metrics that do not have color in kwargs
    exclude = []
    for params in attrs['metrics']:
        if params['args']['kwargs'].get('color'): exclude.append(params['name'])
    # Assign colos per run and model
    colors = {}
    run_models = [id.split("_/")[0] for id in data_ids]
    run_models = list(OrderedDict.fromkeys(run_models))  # Remove duplicates
    # colors_hash = {run_model: next(p.colors) for run_model in run_models}
    for id in data_ids:
        if any([metric in id for metric in exclude]): continue
        for run_model, color in colors_hash.items():
            if run_model in id: colors[id] = color
    # Create filling for stdd
    if attrs.get('stdd'):
        fill_params = defaultdict(dict)
        for id in data_ids:
            fill_params[id]['offset'] = np.squeeze(stdd[id])
            fill_params[id]['kwargs'] = dict(attrs['stdd'])
            fill_params[id]['kwargs']['hatch'] = next(hatches)
    for run_label, run_path in attrs['runs'].items():
        for params in attrs['metrics']:
            func = getattr(p, params["plot_type"])
            args = dict(params['args'])
            args['plot_id'] = fig_name  # Assign plot id
            args['labels'] = {string: f"{string.split('_')[0]} {string.split('_')[1]}" for string in data_ids}
            # Prepare stdd
            # Prepare color
            if attrs.get('stdd'): args['fill_between'] = fill_params
            kwargs = args.pop('kwargs')
            args['colors'] = colors
            args['data_ids'] = [id for id in data_ids if run_label in id and params['name'] == id.split("/")[1]]
            _, fig, ax = func(**args, **kwargs)
    ylabel = attrs.get('ylabel', '')
    xlabel = attrs.get('xlabel', '')
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    p.remove_repeated_labels(fig_name)
    p.set_export_path(figs_attrs['global']['export_path'])
    p.export(fig, fig_name)
    # p.show()
    p.clear()
