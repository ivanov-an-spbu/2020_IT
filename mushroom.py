#!/usr/bin/env python3
"""
The probability calculator based on the attributes from the dataset.

See for the dataset: https://archive.ics.uci.edu/ml/datasets/Mushroom.

Requirements: python>=3.8; matplotlib>=3.3.2; numpy>=1.19.2.

Usage: `<python> mushroom.py <dataset-path>`.

Note: `<dataset-path>` is optional, defaults to `agaricus-lepiota.data`.
"""

# Import MatPlotLib for classes.
import matplotlib as mp

# Import MatPlotLib PyPlot for GUI.
import matplotlib.pyplot as pt

# Import NumPy for calculations.
import numpy as np

# Import the iteration cycle.
from itertools import cycle

# Import JSON encoder for debug printing.
from json import dumps

# Import the command line arguments.
from sys import argv

# Import MatPlotLib button and text box widgets.
from matplotlib.widgets import Button, TextBox

class Probabilities:
    """Probabilities for each attribute value."""

    def __init__(self) -> None:
        """Calculates and saves the probabilities."""

        # Get the dataset path from the CLI or use default one.
        path = argv[1] if len(argv) > 1 else "agaricus-lepiota.data"

        # Load the dataset data.
        data = np.loadtxt(path, delimiter=",", dtype="str")

        # Create the base dictionaries list.
        self.__probabilities = [dict() for _ in range(data.shape[1] - 1)]

        # Go through the data rows.
        np.apply_along_axis(self.__cb_each, axis=1, arr=data)

        # Go through the attributes.
        for attribute in self.__probabilities:

            # Go through the attribute values.
            for key, value in attribute.items():

                # Replace the [common, poisonous] array with probability.
                attribute[key] = value[1] / value[0]

    def average(self, values: list) -> float:
        """Calculates the average probability."""

        # Return the average probability.
        return sum(values) / len(values)

    def filter(self, values: str) -> list:
        """Filters the probability by its keys."""

        # Parse the input.
        values = values.split(",")

        # Go through the parsed input indexes.
        for i in range(len(values)):

            # Check if the attribute value is not found.
            if values[i] not in self.__probabilities[i]:

                # Set the related probability to 1.0.
                values[i] = 1.0

                # Continue the iterations.
                continue

            # Set the correct probability.
            values[i] = self.__probabilities[i][values[i]]

        # Return the probability list.
        return values

    def get(self) -> list:
        """Gets the calculated probabilities."""

        # Just return the calculted probabilities.
        return self.__probabilities

    def log(self) -> None:
        """Logs the probabilities in STDOUT."""

        # Just print the probabilities in STDOUT.
        print(dumps(self.__probabilities, indent=2))

    def random(self) -> str:
        """Generates a random mushroom as a data string."""

        # Set the initial empty data string.
        string = ""

        # Go through the attributes.
        for dict in self.__probabilities:

            # Get the attribute values.
            keys = list(dict.keys())

            # Add the random value to the data string.
            string += keys[np.random.randint(0, len(keys))] + ","

        # Return the generated data string.
        return string[:-1]

    def __cb_each(self, row):
        """Calculates the further probabilities with `row`."""

        # Go through the attribute indexes.
        for i in range(1, len(row)):

            # Skip the unknown value.
            if row[i] == "?": continue

            # Set the pointer to the current attribute.
            pointer = self.__probabilities[i - 1]

            # Set the default attribute values.
            pointer.setdefault(row[i], np.zeros(2, dtype=int))

            # Set the pointer to the attribute value.
            pointer = pointer[row[i]]

            # Increase the poisonous attribute values amount.
            if row[0] == "p": pointer[1] += 1

            # Increase the common attribute values amount.
            pointer[0] += 1

class UserInterface:
    """The user interface made via MatPlotLib."""

    def __init__(self, probabilities: Probabilities) -> None:
        """Initializes the GUI by creating graphics."""

        # Set the Pyplot window size.
        pt.rcParams["figure.figsize"] = (8, 6)

        # Set the initial random mushroom.
        mushroom = probabilities.random()

        # Pass the probabilities.
        self.__probabilities = probabilities

        # Make the bar chart graphic.
        self.__bar_axes = pt.axes([0.1, 0.30, 0.62, 0.6])

        # Make the button "graphic".
        self.__button_axes = pt.axes([0.1, 0.03, 0.8, 0.05])

        # Make the random button "graphic".
        self.__random_axes = pt.axes([0.1, 0.08, 0.8, 0.05])

        # Make the text box "graphic".
        self.__text_box_axes = pt.axes([0.1, 0.13, 0.8, 0.05])

        # Create the submit button.
        self.__button = Button(self.__button_axes, "submit")

        # Create the raddom button.
        self.__random = Button(self.__random_axes, "random")

        # Create the text box.
        self.__text_box = TextBox(self.__text_box_axes, "input", mushroom)

        # Bind the submit button to the graphing function.
        self.__button.on_clicked(self.__cb_submit)

        # Bind the random button to the random mushroom function.
        self.__random.on_clicked(self.__cb_random)

        # Automatically submit for the first time.
        self.__cb_submit(None)

        # Show the plot window.
        pt.show()

    def __cb_random(self, _) -> None:
        """Sets the random mushroom to the text box."""

        # Update the text box text.
        self.__text_box.set_val(self.__probabilities.random())

        # Automatically submit.
        self.__cb_submit(None)

    def __cb_submit(self, _) -> None:
        """Draws a bar chart when the input is sibmitted."""

        # Get the probabilities as a list.
        probs = self.__probabilities.filter(self.__text_box.text)

        # Get the list length.
        length = len(probs) + 1

        # Create the bar color map.
        cmap = pt.cm.get_cmap("jet", length)

        # Generate the var colors.
        colors = [cmap(i) for i in range(0, length)]

        # Set the probabilities names.
        names = [
            "cap-shape",
            "cap-surface",
            "cap-color",
            "bruises?",
            "odor",
            "gill-attachment",
            "gill-spacing",
            "gill-size",
            "gill-color",
            "stalk-shape",
            "stalk-root",
            "stalk-surface-above-ring",
            "stalk-surface-below-ring",
            "stalk-color-above-ring",
            "stalk-color-below-ring",
            "veil-type",
            "veil-color",
            "ring-number",
            "ring-type",
            "spore-print-color",
            "population",
            "habitat",
            "result (overall)"
        ]

        # Append the average probability.
        probs.append(self.__probabilities.average(probs))

        # Increase the length because or reindex.
        length += 1

        # Clear the bar axes before drawing.
        self.__bar_axes.clear()

        # Set the axes grid properties.
        self.__bar_axes.grid(color="darkgrey", linestyle="--", linewidth=1)

        # Set the X axis label.
        self.__bar_axes.set_xlabel("attributes")

        # Set the Y axis label.
        self.__bar_axes.set_ylabel("probability")

        # Set the X axis ticks.
        self.__bar_axes.set_xticks(np.arange(1, length, 2))

        # Set the Y axis ticks.
        self.__bar_axes.set_yticks(np.arange(0.0, 1.1, 0.2))

        # Draw the bar chart of all probabilities.
        self.__bar_axes.bar(range(1, length), height=probs, color=colors)

        # Draw the bar chart legend.
        self.__bar_axes.legend(
            # The handles (colors).
            handles=[self.__handle(cmap, i) for i in range(0, len(names))],

            # The other options.
            bbox_to_anchor=(1., 0.), labels=names, fontsize='x-small', loc=3
        )

        # Redraw the axes.
        pt.draw()

    def __handle(self, cmap: mp.colors.Colormap, i: int) -> pt.Rectangle:
        """Gets the legend colored rectangle handle."""

        # Return the rectangle handle.
        return pt.Rectangle((0, 0), 1, 1, color=cmap(i))

# Check if the script is run directly.
if __name__ == "__main__":
    # Create the probabilities instance.
    # Wow, you have found a stick figure: :)-|--< Take care of it.
    probabilities = Probabilities()

    # Create the user interface instance.
    user_interface = UserInterface(probabilities)
