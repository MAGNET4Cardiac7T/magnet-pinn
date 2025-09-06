.. _Normalization:

-----------------------
Normalization
-----------------------

^^^^^^^^^^^^^^^^^^
Prerequisites:
^^^^^^^^^^^^^^^^^^

- Preprocessed the Data (Examples 1 or 2)

Normalization is an important step in the preprocessing pipeline to ensure that the data is in a suitable range for training models.
The `StandardNormalizer` class can be used to normalize the data.
By using the iterator created in the previous example and additionally setting the `num_samples=10` this snippet will produce the normalization json files.
Note, that the normalization needs to be done for both input and target data.
The following code snippet shows how to normalize the data:

.. code-block:: python 
    
    from magnet_pinn.utils import MinMaxNormalizer, StandardNormalizer
    import einops

    class Iterator:
        def __init__(self, path, iterator):
            self.path = path
            self.iterator = iterator

        def __len__(self):
            return len(self.iterator)

        def __iter__(self):
            for batch in self.iterator:
                input = np.concatenate([batch['input'], batch['coils']], axis=0)
                target = einops.rearrange(batch['field'], 'he reim xyz ... -> (he reim xyz) ...')
                yield {
                    'input': input,
                    'target': target,
                }

    iterator = Iterator("data/processed/train/grid_voxel_size_4_data_type_float32", iterator)

    normalizer = StandardNormalizer()
    normalizer.fit_params(iterator, key='input', axis=0)
    normalizer.save_as_json("data/processed/train/grid_voxel_size_4_data_type_float32/normalization/normalization.json")