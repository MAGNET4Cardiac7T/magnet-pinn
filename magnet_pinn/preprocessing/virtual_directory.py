import warnings

from pathlib import Path
from typing import List, Union


class VirtualDirectory:
    """
    A class to represent a virtual directory consisting of a collection of physical directories.
    Implements the Path interface for iterating over the files (iterdir), checking if empty, etc.
    """
    def __init__(self, paths: Union[str, List[str]]):
        """
            Parameters
            ----------
            paths : Union[str, List[str]]
                A list of paths (or a single path) to the physical directories.
        """
        warnings.warn(
            "Deprecated from the version 0.0.7 while it was replaced with the list of Paths",
            FutureWarning,
            stacklevel=2
        )
        if not isinstance(paths, list):
            paths = [paths]
        self.paths = [Path(path) for path in paths]
        self.max_display_len = 5

    def iterdir(self):
        for path in self.paths:
            yield from path.iterdir()

    def exists(self):
        return all(path.exists() for path in self.paths)
    
    def is_dir(self):
        return all(path.is_dir() for path in self.paths)
    
    def is_file(self):
        return any(path.is_file() for path in self.paths)

    def __str__(self):
        return f"VirtualDirectory({', '.join(str(path) for path in self.paths[:self.max_display_len])})"
    
    def __truediv__(self, simulation):
        for path in self.paths:
            if (path / simulation).exists():
                return path / simulation
