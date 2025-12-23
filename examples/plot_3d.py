from magnet_pinn.data.grid import MagnetGridIterator
import numpy as np
import matplotlib.pyplot as plt
from magnet_pinn.data.transforms import Crop, GridPhaseShift, Compose, DefaultTransform
from matplotlib.colors import LogNorm

BASE_DIR = "data/processed/train/grid_voxel_size_4_data_type_float32"

em_FIELD = ['e-field', 'b-field'] # 'e-field' or 'b-field'

augmentation = Compose(
    [
        Crop(crop_size=(100, 100, 100)),
        GridPhaseShift(num_coils=8)
    ]
)

iterator = MagnetGridIterator(
    BASE_DIR,
    transforms=augmentation,
    num_samples=100
)

item = next(iter(iterator))


for EM_FIELD in em_FIELD:
    if EM_FIELD == 'e-field':
        field_channel = 0
    else:
        field_channel = 1

    # get the field data and coils
    field = np.linalg.norm(item['field'][field_channel], axis=(0, 1))*item['subject']
    coils_real = item['coils'][0, :, :, :]
    coils_imag = item['coils'][1, :, :, :]

    # Define grid
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    z = np.linspace(-1, 1, 100)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Choose y slices to visualize
    y_slices = [-0.45, -0.3, -0.15, 0.0, 0.15, 0.3, 0.45]  # positions along y-axis

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')

    scaling_factor = 1.0  # Adjust this factor to scale the field values for better visualization

    for y_val in y_slices:
        # Find the index closest to y_val
        idx = (np.abs(y - y_val)).argmin()

        # Extract the slice (X-Z plane at fixed y)
        X_slice = X[:, idx, :]
        Z_slice = Z[:, idx, :]
        field_slice = field[:, idx, :]

        visual_y = scaling_factor*y[idx]

        # Use LogNorm for colors
        v_min = float(np.min(field)+1e-10)
        v_max = float(np.max(field)+2e-10)
        norm = LogNorm(vmin=v_min+0.05*v_max, vmax=v_max, clip=False)
        cmap = plt.get_cmap('viridis')

        # Plot as a surface
        ax.plot_surface(X_slice, visual_y*np.ones_like(X_slice), Z_slice,
                        facecolors=cmap(norm(field_slice)),
                        rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)

    ax.view_init(elev=30, azim=150)
    ax.set_xlabel('X', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z', fontsize=12, fontweight='bold')

    s = 0.5
    s2 = 0.5
    ax.set_xlim(-s, s)
    ax.set_ylim(-s2, s2)
    ax.set_zlim(-s, s)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='z', which='major', labelsize=12)
    ax.xaxis.labelpad = 13
    ax.yaxis.labelpad = 13
    ax.zaxis.labelpad = 13
    ax.set_xticks(np.linspace(-s, s, 3))
    ax.set_yticks(np.linspace(-s2, s2, 3))
    ax.set_zticks(np.linspace(-s, s, 3))
    ax.set_box_aspect([1, 2, 1])  # Optional: keep proportions equal

    # Hide panes (background) and grid
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True)
    ax.set_position([-0.4, 0.1, 1.8, 0.7])

    # Hide axes lines
    plt.axis('on')
    if EM_FIELD == 'e-field':
        ax.set_title('Absolute E-field', fontsize=18, fontweight='bold')
        fig.canvas.manager.set_window_title("3D E-Field Visualization")
    else:
        ax.set_title('Absolute B-field', fontsize=18, fontweight='bold')
        fig.canvas.manager.set_window_title("3D B-Field Visualization")
    plt.show()
