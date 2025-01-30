import numpy as np

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
    

def check_items_datatypes(result, random_item):
    assert type(result.simulation) == type(random_item.simulation)
    assert result.input.dtype == random_item.input.dtype
    assert result.field.dtype == random_item.field.dtype
    assert result.subject.dtype == random_item.subject.dtype
    assert type(result.positions) == type(random_item.positions)
    assert result.phase.dtype == random_item.phase.dtype
    assert result.mask.dtype == random_item.mask.dtype
    assert result.coils.dtype == random_item.coils.dtype
    assert result.dtype == random_item.dtype
    assert type(result.dtype) == type(random_item.dtype)
    assert result.truncation_coefficients.dtype == random_item.truncation_coefficients.dtype


def check_cropped_shapes(result):
    assert result.input.shape == (3, 10, 10, 10)
    assert result.field.shape == (2, 2, 3, 10, 10, 10, 8)
    assert result.subject.shape == (10, 10, 10)
    assert len(result.positions) == 0
    assert result.phase.shape == (8,)
    assert result.mask.shape == (8,)
    assert result.coils.shape == (10, 10, 10, 8)
    assert result.truncation_coefficients.shape == (3,)


def check_items_shapes_suppsed_to_be_equal(result, input_item):
    assert result.input.shape == input_item.input.shape
    assert result.field.shape == input_item.field.shape
    assert result.subject.shape == input_item.subject.shape
    assert len(result.positions) == len(input_item.positions)
    assert result.phase.shape == input_item.phase.shape
    assert result.mask.shape == input_item.mask.shape
    assert result.coils.shape == input_item.coils.shape
    assert result.truncation_coefficients.shape == input_item.truncation_coefficients.shape


def check_elements_not_changed_by_crop(result, input_item):
    assert result.simulation == input_item.simulation
    assert result.positions == input_item.positions
    assert np.equal(result.phase, input_item.phase).all()
    assert np.equal(result.mask, input_item.mask).all()
    assert result.dtype == input_item.dtype
    assert np.equal(result.truncation_coefficients, input_item.truncation_coefficients).all()


def check_constant_shapes_not_changed_by_phase_shift(result, item): 
    assert len(result.simulation) == len(item.simulation)
    assert result.input.shape == item.input.shape
    assert result.subject.shape == item.subject.shape
    assert len(result.positions) == len(item.positions)
    assert result.phase.shape == item.phase.shape
    assert result.mask.shape == item.mask.shape
    assert len(result.dtype) == len(item.dtype)
    assert result.truncation_coefficients.shape == item.truncation_coefficients.shape


def check_constant_values_not_changed_by_phase_shift(result, item):
    assert result.simulation == item.simulation
    assert np.equal(result.input, item.input).all()
    assert np.equal(result.subject, item.subject).all()
    assert result.positions == item.positions
    assert result.dtype == item.dtype
    assert np.equal(result.truncation_coefficients, item.truncation_coefficients).all()


def check_default_transform_resulting_shapes(result, item):
    assert result.input.shape == item.input.shape
    assert result.subject.shape == item.subject.shape
    assert len(result.positions) == len(item.positions)
    assert result.phase.shape == item.phase.shape
    assert result.mask.shape == item.mask.shape

    assert result.field.shape == item.field.shape[:-1]
    assert result.coils.shape == tuple([2] + list(item.coils.shape[:-1]))


def check_default_transform_resulting_values(result, item):
    assert result.simulation == item.simulation
    assert np.equal(result.input, item.input).all()
    assert np.equal(result.subject, item.subject).all()
    assert result.positions == item.positions
    assert result.dtype == item.dtype
    assert np.equal(result.truncation_coefficients, item.truncation_coefficients).all()

    assert np.equal(result.field, np.sum(item.field, axis=-1)).all()

    coils_num = item.coils.shape[-1]
    assert np.equal(result.phase, np.zeros(coils_num, dtype=item.phase.dtype)).all()
    assert np.equal(result.mask, np.ones(coils_num, dtype=item.mask.dtype)).all()

    expected_coils = np.stack([
        np.sum(item.coils, axis=-1),
        np.zeros(item.coils.shape[:-1], dtype=item.coils.dtype)
    ], axis=0)
    assert np.equal(result.coils, expected_coils).all()
