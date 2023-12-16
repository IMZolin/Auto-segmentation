import os

source_image_path = os.path.join(".", "AutoSegmentation", "beads_merged.tif")
blured_image_path = os.path.join(
    ".", "AutoSegmentation", "results", "blurred_image.tif"
)
binarized_image_path = os.path.join(
    ".", "AutoSegmentation", "results", "binarized_image.tif"
)
beads_path = os.path.join(".", "AutoSegmentation", "results", "beads")
average_bead_path = os.path.join(".", "AutoSegmentation", "results", "average_bead.tif")
mark_bead_path = os.path.join(".", "AutoSegmentation", "results", "mark_beads.tif")
