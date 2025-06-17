from typing import TypeVar

from numpy import float64
from numpy.typing import NDArray
from typing_extensions import Annotated

num_fsp = TypeVar("numFSP", bound=int)
evts_type = Annotated[NDArray[float64], ("shape", (..., num_fsp, 4))]
evt_type = Annotated[NDArray[float64], ("shape", (num_fsp, 4))]
Jijs_type = Annotated[NDArray[float64], ("shape", (..., num_fsp, num_fsp))]
Pijs_type = Annotated[NDArray[float64], ("shape", (..., num_fsp, num_fsp))]
datum_type = dict[float, dict[str, NDArray]]
