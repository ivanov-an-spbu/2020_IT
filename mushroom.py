#!/usr/bin/env python3
"""
The probability calculator based on the attributes from the dataset.

See for the dataset: https://archive.ics.uci.edu/ml/datasets/Mushroom.

Requirements: python>=3.8; matplotlib>=3.3.2; numpy>=1.19.2.

Usage: `<python> mushroom.py <dataset-path>`.

Note: `<dataset-path>` is optional, defaults to `agaricus-lepiota.data`.
"""

# Import the system utilities.
import sys

# Import MatPlotLib for type classes.
import matplotlib as mpl

# Import MatPlotLib PyPlot for GUI.
import matplotlib.pyplot as plt

# Import MatPlotLib widgets.
import matplotlib.widgets as wdg

# Import NumPy for calculations.
import numpy as npy

# Import the type utilities.
import typing as typ

# Import the iteration utilities.
import itertools as itr

def load_dataset() -> npy.ndarray:
    """Loads the dataset data from the dataset file."""

    # Get the dataset path from the CLI or use default one.
    path = sys.argv[1] if len(sys.argv) > 1 else "agaricus-lepiota.data"

    # Return the dataset data.
    return npy.loadtxt(path, "str", "#", ",")

def get_avg_prob(probs: typ.Tuple[float]) -> float:
    """Finds the average probability of the probabilities."""

    # Return the average probability.
    return npy.average(probs)

def get_pos(data: npy.ndarray) -> typ.Tuple[npy.ndarray]:
    """Gets the list of possible values for each attribute."""

    # Filter the possible unique attribute values from the data.
    return tuple(npy.unique(data[:, i]) for i in range(data.shape[1]))

def get_probs(data: npy.ndarray, attrs: typ.Tuple[str]) -> typ.Tuple[float]:
    """Gets the probabilities for the provided attribute values."""

    # Set all and poisonous row filters.
    fa, fp = data[:, 1:] == attrs, npy.transpose([data[:, 0] == "p"])

    # Calculate the amounts of all and poisonous attribute values occurences.
    a, p = npy.sum(fa, 0), npy.sum(npy.logical_and(fa, fp), 0)

    # Do not show warnings on invalid arithmetical operations.
    with npy.errstate(invalid="ignore"):

        # Generate the probabilities (1.0 is set if 0 by 0 division is done).
        return tuple(npy.nan_to_num(p / a, nan=1.0))

def gen_rnd_mushroom(pos: typ.Tuple[npy.ndarray]) -> str:
    """Generates a random mushroom using the possible attributes."""

    # Set the string piece generator function.
    f = lambda v: f"{npy.random.choice(v)},"

    # Return the string with random attribute values separated by comma.
    return "".join(f(v) for v in itr.islice(pos, 1, None))[:-1]

class GUI:
    """The user interface based on the matplotlib package."""

    # Set the grid options.
    __GRID = {
        # The line color.
        "color": "darkgrey",

        # The line type.
        "linestyle": "dashed",

        # The line width.
        "linewidth": 1,
    }

    # Set the legend options.
    __LEGEND = {
        # The relative to location position.
        "bbox_to_anchor": (1.0, 0.0),

        # The font size (here: extra small).
        "fontsize": "x-small",

        # The location (here: lower left).
        "loc": 3,
    }

    # Set all the attribute names.
    __NAMES = (
        # Name in agaricus-lepiota.names: cap-shape.
        "cap shape",

        # Name in agaricus-lepiota.names: cap-surface.
        "cap surface",

        # Name in agaricus-lepiota.names: cap-color.
        "cap color",

        # Name in agaricus-lepiota.names: bruises?.
        "bruises?",

        # Name in agaricus-lepiota.names: odor.
        "odor",

        # Name in agaricus-lepiota.names: gill-attachment.
        "gill attachment",

        # Name in agaricus-lepiota.names: gill-spacing.
        "gill spacing",

        # Name in agaricus-lepiota.names: gill-size.
        "gill size",

        # Name in agaricus-lepiota.names: gill-color.
        "gill color",

        # Name in agaricus-lepiota.names: stalk-shape.
        "stalk shape",

        # Name in agaricus-lepiota.names: stalk-root.
        "stalk root",

        # Name in agaricus-lepiota.names: stalk-surface-above-ring.
        "stalk surface above ring",

        # Name in agaricus-lepiota.names: stalk-surface-below-ring.
        "stalk surface below ring",

        # Name in agaricus-lepiota.names: stalk-color-above-ring.
        "stalk color above ring",

        # Name in agaricus-lepiota.names: stalk-color-below-ring.
        "stalk color below ring",

        # Name in agaricus-lepiota.names: veil-type.
        "veil type",

        # Name in agaricus-lepiota.names: veil-color.
        "veil color",

        # Name in agaricus-lepiota.names: ring-number.
        "ring number",

        # Name in agaricus-lepiota.names: ring-type.
        "ring type",

        # Name in agaricus-lepiota.names: spore-print-color.
        "spore print color",

        # Name in agaricus-lepiota.names: population.
        "population",

        # Name in agaricus-lepiota.names: habitat.
        "habitat",

        # Name in agaricus-lepiota.names: (no name, result).
        "result (average)",
    )

    # Set the attribute names range.
    __NRANGE = range(len(__NAMES))

    def __init__(self, data: npy.ndarray) -> None:
        """Initializes the GUI by preparing matplotlib window."""

        # Set the matplotlib window size.
        plt.rcParams["figure.figsize"] = (8, 6)

        # Set the data as parameter.
        self.__data = data

        # Set the possible attribute values.
        self.__pos = get_pos(data)

        # Initialize the axes.
        self.__init_axes()

        # Initialize the UI components.
        self.__init_cont()

        # Submit for the first time.
        self.__cb_sub(None)

        # Show the plot window.
        plt.show(block=True)

    def __cb_hnd(self, cmap: mpl.colors.Colormap, i: int) -> None:
        """Generates a rectangle used as a legend handler."""

        # Return the rectangle of color from the color map.
        return plt.Rectangle((0, 0), 1, 1, color=cmap(i))

    def __cb_rnd(self, _: mpl.backend_bases.MouseEvent) -> None:
        """Sets the random mushroom and submits it."""

        # Update the text box text and automatically submit.
        self.__cb_sub(self.__txt.set_val(gen_rnd_mushroom(self.__pos)))

    def __cb_sub(self, _: mpl.backend_bases.MouseEvent) -> None:
        """Draws a bar chart when the input is sibmitted."""

        # Get the input attribute values.
        attrs = self.__txt.text.split(",")

        # Get the probabilities as a list.
        probs = list(get_probs(self.__data, attrs))

        # Append the average probability.
        probs.append(get_avg_prob(probs))

        # Get the probabilities list length.
        length = len(probs)

        # Draw the plot base.
        self.__draw_bar_base(length)

        # Draw the plot content.
        self.__draw_bar_cont(probs, length)

        # Redraw all the axes.
        plt.draw()

    def __draw_bar_base(self, length: int) -> None:
        """Draws the bar chart plot base: coordinates and attributes."""

        # Clear the axes beforehand.
        self.__bar_ax.clear()

        # Set the X axis label.
        self.__bar_ax.set_xlabel("attributes")

        # Set the Y axis label.
        self.__bar_ax.set_ylabel("probability")

        # Set the X axis ticks.
        self.__bar_ax.set_xticks(npy.arange(1, length, 2))

        # Set the Y axis ticks.
        self.__bar_ax.set_yticks(npy.arange(0.0, 1.1, 0.2))

    def __draw_bar_cont(self, probs: typ.List[float], length: int) -> None:
        """Draws the bar chart plot content: the grid, the bar chart, ..."""

        # Set the axes grid properties.
        self.__bar_ax.grid(**self.__GRID)

        # Create the bar color map.
        cmap = plt.cm.get_cmap("rainbow", length)

        # Generate the various colors.
        cols = [cmap(i) for i in range(0, length)]

        # Draw the bar chart of all probabilities.
        self.__bar_ax.bar(range(1, length + 1), probs, color=cols)

        # Generate the legend handles.
        hand = [self.__cb_hnd(cmap, i) for i in self.__NRANGE]

        # Draw the bar chart legend.
        self.__bar_ax.legend(hand, self.__NAMES, **self.__LEGEND)

    def __init_axes(self) -> None:
        """Initializes the axes on which UI and graphics will be drawn."""

        # Set the bar chart graphic.
        self.__bar_ax = plt.axes((0.1, 0.30, 0.62, 0.6))

        # Set the submit button "graphic".
        self.__sub_ax = plt.axes((0.1, 0.03, 0.8, 0.05))

        # Set the random button "graphic".
        self.__rnd_ax = plt.axes((0.1, 0.08, 0.8, 0.05))

        # Set the text box "graphic".
        self.__txt_ax = plt.axes((0.1, 0.13, 0.8, 0.05))

    def __init_cont(self) -> None:
        """Initializes the UI content - inputs, buttons, ..."""

        # Generate a random mushroom.
        mushroom = gen_rnd_mushroom(self.__pos)

        # Set the random button.
        self.__rnd = wdg.Button(self.__rnd_ax, "random")

        # Set the submit button.
        self.__sub = wdg.Button(self.__sub_ax, "submit")

        # Set the text box with the mushroom.
        self.__txt = wdg.TextBox(self.__txt_ax, "input", mushroom)

        # Bind the submit button to the bar chart graphing function.
        self.__sub.on_clicked(self.__cb_sub)

        # Bind the random button to the random mushroom function.
        self.__rnd.on_clicked(self.__cb_rnd)

# Load the data from the dataset and load GUI if the script is run directly.
# Wow, you have found the hidden stick figure: :)-|--<! Take care of it.
if __name__ == "__main__": GUI(load_dataset())
