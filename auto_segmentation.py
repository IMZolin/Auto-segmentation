import os
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import cv2
from skimage.metrics import structural_similarity as ssim
from config import (
    binarized_image_path,
    blured_image_path,
    beads_path,
    average_bead_path,
    mark_bead_path,
)


class AutoSegmentation:
    def __init__(self):
        self.img_manager = None
        self.source_img = None
        self.mid_layer_idx = None
        self.blur_img = None
        self.bin_img = None
        self.analysis = None
        self.bead_centers = None
        self.extracted_beads = None
        self.box_size = None
        self.averaged_bead = None
        self.mark_beads_img = None
        self.avg_bead_3d_projection = None
        self.avg_bead_2d_projections = None
        self.GAUSS_BLUR_RAD = (7, 7)

    def load_image(self, image_path: str) -> None:
        """Load the source image."""
        try:
            if image_path is None:
                raise Exception("Source image path is None.")

            ret, images = cv2.imreadmulti(image_path, [], cv2.IMREAD_ANYCOLOR)
            if not ret or len(images) == 0:
                raise Exception("Failed to read the source image.")

            self.source_img = np.asarray(images)
            self.mid_layer_idx = self.source_img.shape[0] // 2

        except Exception as e:
            print(f"Error in load_image: {str(e)}")

    def binarize(self, is_show: bool = False) -> None:
        """Binarize the loaded source image."""
        try:
            if self.source_img is None:
                raise Exception("Source image is not loaded. Run load_image first.")

            mono_img = cv2.cvtColor(self.source_img[self.mid_layer_idx], cv2.COLOR_BGR2GRAY)
            self.blur_img = cv2.GaussianBlur(mono_img, self.GAUSS_BLUR_RAD, 0)
            self.bin_img = cv2.threshold(
                self.blur_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
            )[1]
            self.bin_img = cv2.bitwise_not(self.bin_img)

            if is_show:
                cv2.imshow("Image", self.source_img[self.mid_layer_idx])
                cv2.imshow("Binarized image", self.bin_img)
                cv2.waitKey(0)

        except Exception as e:
            print(f"Error in binarize: {str(e)}")

    def find_bead_centers(self) -> None:
        """Find bead centers in the binarized image."""
        try:
            if self.blur_img is None:
                raise Exception(
                    "Blur image is not available. Run cv2_preprocess first."
                )

            self.analysis = cv2.connectedComponentsWithStats(self.bin_img, 4, cv2.CV_32S)
            (totalLabels, label_ids, values, centroid) = self.analysis

            if len(centroid) > 1:
                self.bead_centers = np.array(centroid[1:], dtype=int)
            else:
                raise Exception("No connected components found in the image.")

        except Exception as e:
            print(f"Error in find the bead centers: {str(e)}")

    def filter_points(self, max_area: int = 120) -> None:
        """Filter bead centers based on area."""
        try:
            if self.bead_centers is None:
                raise Exception(
                    "Bead centers are not available. Run find_bead_centers first."
                )

            (totalLabels, label_ids, values, centroid) = self.analysis

            filtered_centers = []
            for i in range(1, totalLabels):
                area = values[i, cv2.CC_STAT_AREA]
                if 0 < area < max_area:
                    filtered_centers.append(self.bead_centers[i - 1])
            self.bead_centers = np.array(filtered_centers, dtype=int)

        except Exception as e:
            print(f"Error in filter_points: {str(e)}")

    def extract_beads(self, box_size: int) -> None:
        """Extract beads based on bead centers and box size."""
        try:
            if self.source_img is None:
                raise Exception("Source image is not available.")
            if self.bead_centers is None:
                raise Exception("Bead centers are not available. Run find_bead_centers first.")

            self.extracted_beads = []
            self.box_size = box_size
            new_bead_centers = []

            for center in self.bead_centers:
                x, y = center
                half_size = box_size // 2

                if (
                        half_size <= y < self.source_img.shape[1] - half_size
                        and half_size <= x < self.source_img.shape[2] - half_size
                ):
                    bead = self.source_img[
                           :, y - half_size: y + half_size, x - half_size: x + half_size, :
                           ]
                    if bead.shape[0] == bead.shape[1]:
                        self.extracted_beads.append(bead)
                        new_bead_centers.append(center)

            self.bead_centers = np.array(new_bead_centers, dtype=int)

        except Exception as e:
            print(f"Error in extract_beads: {str(e)}")

    def extract_single_bead(self, center: Tuple[int, int]) -> np.ndarray:
        """Extract a single bead based on its center coordinates."""
        try:
            if self.source_img is None:
                raise Exception("Source image is not available.")
            if center is None:
                raise Exception("Center coordinates are not provided.")

            x, y = center
            half_size = self.box_size // 2
            if (
                    half_size <= y < self.source_img.shape[1] - half_size
                    and half_size <= x < self.source_img.shape[2] - half_size
            ):
                single_bead = self.source_img[
                              :, y - half_size: y + half_size, x - half_size: x + half_size
                              ]
                print(single_bead.shape[1], single_bead.shape[2], single_bead.shape[0])
                if single_bead.shape[1] == single_bead.shape[2]:
                    return single_bead
                else:
                    raise ValueError("Invalid bead dimensions.")
            else:
                raise ValueError("Invalid bead center coordinates.")

        except Exception as e:
            print(f"Error in extract_single_bead: {str(e)}")

    def mark_beads(self, is_show: bool = False) -> None:
        """Mark beads on the source image."""
        try:
            if self.source_img is None:
                raise Exception("Source image is not available.")
            if self.bead_centers is None:
                raise Exception(
                    "Bead centers are not available. Run find_bead_centers first."
                )
            if self.box_size is None:
                raise Exception(
                    "Box size is not available. Set the box_size before marking beads."
                )
            self.mark_beads_img = self.source_img[self.mid_layer_idx].copy()
            for center in self.bead_centers:
                x, y = center
                half_size = self.box_size // 2
                top_left = (x - half_size, y - half_size)
                bottom_right = (x + half_size, y + half_size)
                cv2.rectangle(
                    self.mark_beads_img,
                    top_left,
                    bottom_right,
                    color=(255, 0, 0),
                    thickness=2,
                )
            if is_show:
                cv2.imshow("Marked Beads", self.mark_beads_img)
                cv2.waitKey(0)

        except Exception as e:
            print(f"Error in mark_beads: {str(e)}")

    def average_bead(self, is_show: bool = False) -> None:
        """Calculate and show the average bead."""
        try:
            if self.extracted_beads is None or len(self.extracted_beads) == 0:
                raise ValueError("No extracted beads available.")

            self.averaged_bead = np.mean(self.extracted_beads, axis=0).astype("uint8")
            if is_show:
                cv2.imshow("Marked Beads", self.mark_beads_img)
                cv2.waitKey(0)
        except Exception as e:
            print(f"Error in average_bead: {str(e)}")

    def save_results(self) -> None:
        """Save binarized, blurred, averaged beads, and marked beads images."""
        try:
            if self.bin_img is not None:
                cv2.imwrite(binarized_image_path, self.bin_img)
            if self.blur_img is not None:
                cv2.imwrite(blured_image_path, self.blur_img)
            if self.averaged_bead is not None:
                cv2.imwritemulti(average_bead_path, self.averaged_bead)
            if self.mark_beads_img is not None:
                cv2.imwrite(mark_bead_path, self.mark_beads_img)
        except Exception as e:
            print(f"Error in save_results: {str(e)}")

    def save_beads(self, folder_path: str = beads_path) -> None:
        """Save extracted beads to a folder."""
        try:
            if self.extracted_beads is None:
                raise Exception("Extracted beads are not available. Run extract_beads first.")

            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            for i, bead in enumerate(self.extracted_beads):
                if bead.ndim == 3:
                    bead_rgb = cv2.cvtColor(bead, cv2.COLOR_BGR2RGB)
                    cv2.imwritemulti(os.path.join(folder_path, f"bead_{i}.tif"), bead_rgb)
                else:
                    cv2.imwritemulti(os.path.join(folder_path, f"bead_{i}.tif"), bead)
            print(f"Beads saved to folder: {folder_path}")

        except Exception as e:
            print(f"Error in save_beads: {str(e)}")

    def measure_processing_time(self, image_path: str) -> None:
        """Measure the processing time for all algorithms"""
        try:
            start_time = time.time()
            self.load_image(image_path=image_path)
            if self.source_img is not None:
                self.binarize(is_show=False)
                self.find_bead_centers()
                self.filter_points(max_area=500)
                self.extract_beads(box_size=36)
                self.average_bead()

            end_time = time.time()
            processing_time = end_time - start_time
            print(f"Processing Time: {processing_time} seconds")

        except Exception as e:
            print(f"Error in measure_processing_time: {str(e)}")

    def calculate_sim_histogram(self, center: Tuple[int, int]) -> None:
        """Calculate and show a histogram of the similarity indices with the marked beads."""
        # TODO: correct histogram for 3d average bead
        try:
            if self.mark_beads_img is None:
                raise Exception("Marked beads image is not available.")
            if center is None:
                raise Exception("Center coordinates are not provided.")

            single_bead = self.extract_single_bead(center)

            similarity_values = []
            for bead in self.extracted_beads:
                similarity_index = ssim(single_bead, bead, full=True)[0]
                similarity_values.append(similarity_index)

            plt.hist(
                similarity_values, bins=50, density=True, alpha=0.75, color="b"
            )
            plt.xlabel("Structural Similarity Index (SSIM)")
            plt.ylabel("Frequency")
            plt.title(
                "Histogram of Bead Similarity"
            )
            plt.show()

        except Exception as e:
            print(
                f"Error in calculate_similarity_histogram_with_single_bead: {str(e)}"
            )

    def generate_2d_projections(self, is_show: bool = False) -> None:
        """Generate 2D projections (xy, xz, yz) and save them in self.avg_bead_projection."""
        try:
            if self.averaged_bead is not None:
                result = np.where(self.averaged_bead == np.amax(self.averaged_bead))
                projections_coord = [result[0][0], result[1][0], result[2][0]]
                fig, axs = plt.subplots(3, 1, sharex=False, figsize=(2, 6))

                # XY Projection
                # TODO: make independence from the channel;
                # TODO: make layer switching for projections
                axs[0].imshow(self.averaged_bead[projections_coord[0], :, :, 1], cmap='jet',
                              vmin=np.min(self.averaged_bead), vmax=np.max(self.averaged_bead))
                axs[0].set_title('XY Projection')

                # XZ Projection
                axs[1].imshow(self.averaged_bead[:, projections_coord[1], :, 1], cmap='jet',
                              vmin=np.min(self.averaged_bead), vmax=np.max(self.averaged_bead))
                axs[1].set_title('XZ Projection')

                # YZ Projection
                axs[2].imshow(self.averaged_bead[:, :, projections_coord[2], 1], cmap='jet',
                              vmin=np.min(self.averaged_bead), vmax=np.max(self.averaged_bead))
                axs[2].set_title('YZ Projection')

                self.avg_bead_2d_projections = fig
                if is_show:
                    plt.show()
            else:
                raise ValueError("No averaged bead available.")
        except Exception as e:
            print(f"Error in generate_2d_projections: {str(e)}")

    def create_3d_projection(self, is_show: bool = False) -> None:
        """Create and save the 3D projection for the averaged bead."""
        # TODO: correct 3d(4d) intensity chart for 3d average bead
        try:
            if self.averaged_bead is None:
                raise ValueError("Averaged bead is not available. Run average_bead first.")

            r_channel = self.averaged_bead[:, :, 0]
            g_channel = self.averaged_bead[:, :, 1]
            b_channel = self.averaged_bead[:, :, 2]

            x, y = np.meshgrid(np.arange(r_channel.shape[1]), np.arange(r_channel.shape[0]))
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(x, y, r_channel, c='red', marker='o', label='Red Channel')
            ax.scatter(x, y, g_channel, c='green', marker='o', label='Green Channel')
            ax.scatter(x, y, b_channel, c='blue', marker='o', label='Blue Channel')

            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.set_zlabel('Intensity')
            ax.set_title('3D Projection of Averaged Bead')
            self.avg_bead_3d_projection = fig
            if is_show:
                plt.show()

        except Exception as e:
            print(f"Error in create_3d_projection: {str(e)}")
