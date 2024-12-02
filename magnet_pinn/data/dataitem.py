
from dataclasses import dataclass, field
from typing import Optional

import numpy.typing as npt
import numpy as np


@dataclass
class DataItem:
    simulation: str
    input: npt.NDArray[np.float32]
    field: npt.NDArray[np.float32]
    subject: npt.NDArray[np.bool_]
    positions: Optional[npt.NDArray[np.float32]] = field(default_factory=list)
    phase: Optional[npt.NDArray[np.float32]] = field(default_factory=list)
    mask: Optional[npt.NDArray[np.bool_]] = field(default_factory=list)
    coils: Optional[npt.NDArray[np.bool_]] = field(default_factory=list)
    dtype: Optional[str] = field(default_factory=str)
    truncation_coefficients: Optional[npt.NDArray] = field(default_factory=list)