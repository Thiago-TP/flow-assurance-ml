"""Raw-data visualization: instance histories, well histories, signatures.

Plots work directly on the raw 3W parquet files (not on the extracted
features), so they show exactly what the pipeline consumes. Each family writes
one PDF per fault or per well into its own directory:

=========================  ============================  ======================
Plot                       Directory                     Instances
=========================  ============================  ======================
``plot_fault``             ``INSTANCE_FIGURES_DIR``      real only
``plot_well_history``      ``WELL_HISTORY_FIGURES_DIR``  real only
``plot_faults_per_well``   ``WELL_HISTORY_FIGURES_DIR``  real only
``plot_fault_signatures``  ``SIGNATURE_FIGURES_DIR``     real, simulated, drawn
=========================  ============================  ======================

Everything keyed to a physical well is restricted to real instances
(``WELL-*`` files), since simulated and hand-drawn ones have no well to belong
to. Signatures are about the shape of an event rather than about a well, so
they cover the three instance sources, one subdirectory each.

One module per family, plus their common ground:

==============  ==============================================================
Module          Contents
==============  ==============================================================
``common``      palettes, ``dataset.ini`` readers, instance discovery, label
                bands and shading, the min/max envelope, PDF writing
``instances``   ``plot_fault``: every sensor of every real instance of a fault
``signatures``  ``plot_fault_signatures``: the variables that identify a fault
``wells``       ``plot_well_history``: a well's instances joined in time
``timeline``    ``plot_faults_per_well``: how a well's instances tile its history
==============  ==============================================================

Sensor units are read from the 3W ``dataset.ini`` shipped with the dataset.
"""

from flowml.visualization.common import (
    FAULT_COLORS,
    LABEL_COLORS,
    STATE_COLORS,
    list_instances,
    list_well_ids,
    load_sensor_names,
    load_sensor_units,
    resolve_fault,
)
from flowml.visualization.instances import plot_fault
from flowml.visualization.signatures import (
    FAULT_SIGNATURES,
    plot_all_fault_signatures,
    plot_fault_signatures,
)
from flowml.visualization.timeline import plot_faults_per_well
from flowml.visualization.wells import load_well_history, plot_well_history, plot_wells_histories

__all__ = [
    "FAULT_COLORS",
    "FAULT_SIGNATURES",
    "LABEL_COLORS",
    "STATE_COLORS",
    "list_instances",
    "list_well_ids",
    "load_sensor_names",
    "load_sensor_units",
    "load_well_history",
    "plot_all_fault_signatures",
    "plot_fault",
    "plot_fault_signatures",
    "plot_faults_per_well",
    "plot_well_history",
    "plot_wells_histories",
    "resolve_fault",
]
