"""Run ECMWF AIFS Single v2 and AIFS-ENS v2 on Apple silicon (MPS / Metal).

The two models need incompatible ``anemoi-models`` versions, so each has a thin runtime
project under ``runtimes/`` that pins its own; everything else -- initial conditions, the
MPS patches, the Zarr layout and the streaming writer -- lives here and is shared.
"""

__version__ = "0.1.0"
