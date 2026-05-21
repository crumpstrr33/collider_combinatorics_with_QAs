from typing import Annotated, TypeAlias

from numpy import floating
from numpy.typing import NDArray

NDFloat = NDArray[floating]

# Array of multiple `evt_type`s
EvtsType: TypeAlias = Annotated[NDFloat, ("shape", (..., "num_fsp", 4))]
# Shape of the 4-momenta for a single event
EvtType: TypeAlias = Annotated[NDFloat, ("shape", ("num_fsp", 4))]
# A single Jij/Pij-like matrix
CoeffType: TypeAlias = Annotated[NDFloat, ("shape", ("num_fsp", "num_fsp"))]
# Array of Jij/Pij-like matrices
CoeffsType: TypeAlias = Annotated[NDFloat, ("shape", (..., "num_fsp", "num_fsp"))]
# Data storage: {invm_bin: {datum: [...]}}
# where `invm_bin` are the values in INVMS[:-1] in constants.py
# and datum are the keys for all the data collected in a run in jobrunner.py
DatumType: TypeAlias = dict[float, dict[str, NDArray[floating | str]]]
