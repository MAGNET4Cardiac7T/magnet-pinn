from magnet_pinn.data._base import BaseTransform
from magnet_pinn.data.dataitem import DataItem


class FirstAugmentation(BaseTransform):
        def __call__(self, simulation: DataItem) -> DataItem:
            simulation.simulation += "1"
            return simulation


class SecondAugmentation(BaseTransform):
    def __call__(self, simulation: DataItem) -> DataItem:
        simulation.simulation += "2"
        return simulation


class ThirdAugmentation(BaseTransform):
    def __call__(self, simulation: DataItem) -> DataItem:
        simulation.simulation += "3"
        return simulation
