import tkinter as tk
from tkinter import filedialog
from auto_segmentation import AutoSegmentation
from PIL import Image, ImageTk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class GUIExecutor:
    def __init__(self, master):
        self.master = master
        self.master.title("Auto-segmentation beads")

        self.auto_segm = AutoSegmentation()

        self.menu_bar = tk.Menu(master)
        master.config(menu=self.menu_bar)

        # File Menu
        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Load Image", command=self.load_image)
        self.file_menu.add_command(label="Segmentate beads", command=self.process_image)
        self.file_menu.add_command(label="Exit", command=master.destroy)

        # Frame for Display Images
        self.image_frame = tk.Frame(master)
        self.image_frame.pack(pady=10)

        # Display Images
        self.original_label = tk.Label(self.image_frame, text="Source Image")
        self.original_label.grid(row=0, column=0, padx=10)

        self.marked_label = tk.Label(self.image_frame, text="Processed Image")
        self.marked_label.grid(row=0, column=1, padx=10)

        self.average_label = tk.Label(self.image_frame, text="Averaged Bead")
        self.average_label.grid(row=0, column=2, padx=10)

        # Canvas for 3D Projection
        self.projection_canvas = tk.Canvas(self.image_frame)
        self.projection_canvas.grid(row=0, column=3, padx=10)

        # Slider for Averaged Beads
        self.slider_label = tk.Label(self.image_frame, text="Averaged Bead Layer:")
        self.slider_label.grid(row=1, column=0, columnspan=3, pady=5)

        self.slider = tk.Scale(self.image_frame, from_=0, to=0, orient="horizontal", command=self.slider_callback)
        self.slider.grid(row=2, column=0, columnspan=3, pady=5)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.tif;*.tiff")])
        if file_path:
            self.auto_segm.load_image(image_path=file_path)
            self.show_images()

    def process_image(self):
        if self.auto_segm.source_img is not None:
            self.auto_segm.binarize(is_show=False)
            self.auto_segm.find_bead_centers()
            self.auto_segm.filter_points(max_area=500)
            self.auto_segm.extract_beads(box_size=36)
            self.auto_segm.average_bead()
            self.auto_segm.mark_beads(is_show=False)
            self.show_images()

            # Update slider range based on the number of averaged beads
            num_averaged_beads = len(self.auto_segm.averaged_bead) if (self.auto_segm.averaged_bead is not None and len(self.auto_segm.averaged_bead) > 0) else 0
            self.slider.config(to=num_averaged_beads - 1)

    def show_images(self):
        if self.auto_segm.source_img is not None:
            original_img = self.auto_segm.source_img[self.auto_segm.mid_layer_idx]
            original_img = Image.fromarray(original_img.astype('uint8'))
            original_img.thumbnail((400, 400))
            original_img = ImageTk.PhotoImage(original_img)
            self.original_label.config(image=original_img)
            self.original_label.image = original_img

            if self.auto_segm.mark_beads_img is not None:
                marked_img = self.auto_segm.mark_beads_img
                marked_img = Image.fromarray(marked_img.astype('uint8'))
                marked_img.thumbnail((400, 400))
                marked_img = ImageTk.PhotoImage(marked_img)
                self.marked_label.config(image=marked_img)
                self.marked_label.image = marked_img

            if self.auto_segm.averaged_bead is not None and len(self.auto_segm.averaged_bead) > 0:
                # Display the first averaged bead initially
                self.show_averaged_bead(0)
                self.slider.config(to=len(self.auto_segm.averaged_bead) - 1)

                self.auto_segm.generate_2d_projections(is_show=False)
                if hasattr(self.auto_segm,
                           'avg_bead_2d_projections') and self.auto_segm.avg_bead_2d_projections is not None:
                    for widget in self.projection_canvas.winfo_children():
                        widget.destroy()

                    canvas = FigureCanvasTkAgg(self.auto_segm.avg_bead_2d_projections, master=self.projection_canvas)
                    canvas.draw()
                    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def slider_callback(self, value):
        # Callback function for the slider
        value = int(value)
        self.show_averaged_bead(value)

    def show_averaged_bead(self, index):
        if self.auto_segm.averaged_bead is not None and len(self.auto_segm.averaged_bead) > 0:
            averaged_img = self.auto_segm.averaged_bead[index]
            averaged_img = Image.fromarray(np.uint8(averaged_img))
            averaged_img = averaged_img.resize((200, 200))
            averaged_img = ImageTk.PhotoImage(averaged_img)
            self.average_label.config(image=averaged_img)
            self.average_label.image = averaged_img


if __name__ == "__main__":
    root = tk.Tk()
    gui_executor = GUIExecutor(root)
    root.mainloop()
