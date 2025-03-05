# Orange3 Example Add-on

This is an example add-on for [Orange3](https://orangedatamining.com/) data mining software package.

## Installation

### Prerequisites

You need to have Orange3 and Python installed. The add-on also requires PyQt5.

### Installation from source

To install the add-on from source, run:

```bash
# Clone the repository and move into it
git clone https://github.com/yourusername/orange3-example-addon.git
cd orange3-example-addon

# Install the add-on in development mode
pip install -e .
```

Alternatively, you can install all dependencies at once:

```bash
pip install -r requirements.txt
pip install -e .
```

### Troubleshooting

If you encounter the error `ImportError: PyQt4, PyQt5, PySide or PySide2 are not available for import`, install PyQt5:

```bash
pip install PyQt5
```

## Features

This add-on includes the AutoGluon Leaderboard widget for evaluating time series forecasting models.

## Usage

After the installation, the widget from this add-on is registered with Orange. To run Orange from the terminal,
use

    orange-canvas

or

    python -m Orange.canvas

The new widget appears in the toolbox bar under the section Example.

![screenshot](https://github.com/biolab/orange3-example-addon/blob/master/screenshot.png)
