import open3d as o3d
import numpy as np
from IPython.display import Image

# Load the PCD file
pcd = o3d.io.read_point_cloud(r"Dataset\train\2021_08_16_22_26_54\641\000069.pcd")

# Create a visualizer and add the geometry
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd)

# Capture the rendered image
image = vis.capture_screen_float_buffer(do_render=True)
vis.destroy_window()  # Close the visualizer window

# Convert the image to a displayable format
image_np = np.asarray(image)
