.. _PINN_training:

-----------------------
Training a ML model
-----------------------

Once the data is preprocessed and ready, you can use it to train a model.
The following code snippet shows how to train the 3D-UNet using the preprocessed data:

.. code-block:: python

    import torch
    import einops
    from magnet_pinn.losses import MSELoss
    from magnet_pinn.utils import StandardNormalizer
    from magnet_pinn.data.utils import worker_init_fn
    from magnet_pinn.models import UNet3D

    # Set the base directory where the preprocessed data is stored
    BASE_DIR = "data/processed/train/grid_voxel_size_4_data_type_float32"
    target_normalizer = StandardNormalizer.load_from_json(f"{BASE_DIR}/normalization/target_normalization.json")
    input_normalizer = StandardNormalizer.load_from_json(f"{BASE_DIR}/normalization/input_normalization.json")

    # Create a DataLoader for the preprocessed data
    train_loader = torch.utils.data.DataLoader(iterator, batch_size=4, num_workers=16, worker_init_fn=worker_init_fn)

    # Create the model
    model = UNet3D(5, 12, f_maps=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = MSELoss()
    subject_lambda = 10.0
    space_lambda = 0.01

    for epoch in range(10):
        model.train()
        for i, batch in enumerate(train_loader):
            properties, phase, field, subject_mask = batch['input'], batch['coils'], batch['field'], batch['subject']
            x = torch.cat([properties, phase], dim=1)
            y = einops.rearrange(field, 'b he reim xyz ... -> b (he reim xyz) ...')
            # normalize input and target
            x = input_normalizer(x)
            y = target_normalizer(y)
            optimizer.zero_grad()
            y_hat = model(x)
            # calculate loss
            subject_loss = criterion(y_hat, y, subject_mask)
            space_loss = criterion(y_hat, y, ~subject_mask)
            loss = subject_loss*subject_lambda + space_loss*space_lambda

            loss.backward()
            optimizer.step()
            print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")