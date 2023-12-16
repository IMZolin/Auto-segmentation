from auto_segmentation import AutoSegmentation
from config import source_image_path, beads_path


def main():
    auto_segm = AutoSegmentation()
    auto_segm.load_image(image_path=source_image_path)
    auto_segm.binarize(is_show=False)
    auto_segm.find_bead_centers()
    auto_segm.filter_points(max_area=500)
    auto_segm.extract_beads(box_size=36)
    auto_segm.average_bead()
    auto_segm.mark_beads(is_show=False)
    auto_segm.save_beads(folder_path=beads_path)
    auto_segm.save_results()
    auto_segm.generate_2d_projections(is_show=True)
    # auto_segm.create_3d_projection(is_show=False)
    # auto_segm.calculate_sim_histogram(center=(1894, 1952))


def time_measure():
    auto_segm = AutoSegmentation()
    auto_segm.measure_processing_time(image_path=source_image_path)


if __name__ == "__main__":
    main()
