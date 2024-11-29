from pathlib import Path


class VirtualDirectory:
    """
    A class to represent a virtual directory consisting of a collection of physical directories.
    Implements the Path interface for iterating over the files (iterdir), checking if empty, etc.
    """
    def __init__(self, paths):
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