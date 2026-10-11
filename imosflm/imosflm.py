import argparse
import os
import sys
import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk, ImageOps, ImageEnhance
import fabio  # Import FabIO
import matplotlib.pyplot as plt
import cv2
import numpy as np
import h5py"} "

# Set environment variables
os.environ['MAINTAINER'] = 'mosflm@mrc-lmb.cam.ac.uk'
os.environ['MOSFLM_VERSION_REQUIRED'] = '7.4'
os.environ['TKIMAGELOAD'] = '0'
os.environ['EXPERTDETECTORSETTINGS'] = '0'
os.environ['SPIRAL'] = '0'
os.environ['HDF5file'] = '0'

# Debugging setup
debugging = os.getenv('MOSFLM_DEBUG', '0') == '1'

# CCP4i2 mode setup
ccp4i2 = os.getenv('CCP4I2', '0') == '1'
if ccp4i2:
    print("ccp4i2 mode: ON")

# Fastload mode setup
fastload = os.getenv('FASTLOAD', '0') == '1'
if fastload:
    print("fastload mode: ON")

# Check Tcl/Tk version
tcl_tk_version = tk.Tcl().eval('info patchlevel')
if tcl_tk_version == "8.4.13":
    print(f"\nYour Tcl/Tk version has been determined to be {tcl_tk_version}")
    print("Unfortunately in this version, the time taken to display images is unacceptably long.")
    sys.exit()
else:
    print(f"Tcl platform is {sys.platform}")
    print(f"TclTk version from info patchlevel is {tcl_tk_version}")

# Main application class
class IMosflmApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("iMosflm")
        self.geometry("800x1000")
        self.setup_gui()
        self.lift()  # Raise the window to the top
        self.attributes('-topmost', True)  # Keep it on top
        self.after_idle(self.attributes, '-topmost', False)  # Allow other windows to be on top later
        self.current_image = None  # Store the current image

    def setup_gui(self):
        # Create a menu bar
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open", command=self.open_file)
        file_menu.add_command(label="Save", command=self.save_file)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)

        # Add a text area with a specific height
        self.text_area = tk.Text(self, wrap='word', height=5)
        self.text_area.pack(expand=0, fill='x')  # Adjust fill to 'x' to prevent vertical expansion

        # Add a button frame
        button_frame = tk.Frame(self)
        button_frame.pack(fill='x')
        run_button = tk.Button(button_frame, text="Run", command=self.run_process)
        run_button.pack(side='left', padx=5, pady=5)

        # Add an invert button
        invert_button = tk.Button(button_frame, text="Invert", command=self.invert_image)
        invert_button.pack(side='left', padx=5, pady=5)

        # Add a histogram button
        histogram_button = tk.Button(button_frame, text="Histogram", command=self.show_histogram)
        histogram_button.pack(side='left', padx=5, pady=5)

        # Add an image display area
        self.image_label = tk.Label(self)
        self.image_label.pack(expand=1, fill='both')

        self.contrast_slider = tk.Scale(
            button_frame, from_=0.5, to=3.0, resolution=0.1, orient=tk.HORIZONTAL,
            label="Contrast", command=self.update_contrast
        )
        self.contrast_slider.set(1.0)  # Default contrast
        self.contrast_slider.pack(fill=tk.X)

    def open_file(self):
        # Logic to open a file
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.gif *.tif *.tiff *.cbf")])
        if file_path:
            if file_path.lower().endswith('.cbf'):
                self.display_cbf_image(file_path)
            else:
                self.display_image(file_path)

    def save_file(self):
        # Logic to save a file
        file_path = filedialog.asksaveasfilename(defaultextension=".txt")
        if file_path:
            with open(file_path, 'w') as file:
                content = self.text_area.get(1.0, tk.END)
                file.write(content)

    def show_about(self):
        # Display an about message
        messagebox.showinfo("About", "iMosflm\nVersion 7.4.0\nDeveloped by MRC-LMB")

    def run_process(self):
        # Placeholder for running a process
        self.debug("Run button clicked")

    def display_image(self, file_path):
        # Load and display the image
        image = Image.open(file_path)
        self.current_image = image  # Store the current image
        self.rescale_and_display(image)

    def display_cbf_image(self, file_path):
        """Load and display the CBF image using FabIO."""
        try:
            cbf_image = fabio.open(file_path)
            data = cbf_image.data
            image = Image.fromarray(data)
            self.current_image = image.convert("L")  # Convert to grayscale
            self.rescale_and_display(self.current_image)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CBF image: {e}")

    def display_hdf5_image(self, image):
        """Load and display the HDF5 image using FabIO."""
        try:
            #hdf5_image = fabio.open(image)
            #data = hdf5_image.data
            #image = Image.fromarray(data)

            # Open the HDF5 file
            with h5py.File(image, "r") as f:
                # Explore dataset keys
                print(list(f.keys()))

                # Load an image dataset (adjust key as needed)
                image_data = f["/entry/data/data"][...]
            self.current_image = image_data.convert("L")  # Convert to grayscale
            self.rescale_and_display(self.current_image)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load HDF5 image: {e}")

    def rescale_and_display(self, image):
        """Resize and display the image."""
        if image is None:
            return

        # Get the window width
        window_width = self.winfo_width()

        # Calculate new height to maintain aspect ratio
        aspect_ratio = image.height / image.width
        new_height = int(window_width * aspect_ratio)

        # Resize the image
        resized_image = image.resize((window_width, new_height), Image.LANCZOS)

        # Convert to PhotoImage and display
        photo = ImageTk.PhotoImage(resized_image)
        self.image_label.config(image=photo)
        self.image_label.image = photo  # Keep reference to avoid garbage collection

    def invert_image(self):
        # Invert the current image
        if self.current_image:
            inverted_image = ImageOps.invert(self.current_image.convert("RGB"))
            self.current_image = inverted_image
            self.rescale_and_display(inverted_image)

    def enhance_contrast_clahe(image):
        """Apply CLAHE (Adaptive Histogram Equalization) to enhance contrast."""
        if image.mode not in ("L", "RGB"):
            image = image.convert("L")  # Convert to grayscale

        img_array = np.array(image)  # Convert to numpy array
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_img = clahe.apply(img_array)
        return Image.fromarray(enhanced_img)

    def update_contrast(self, value):
        """Adjust the contrast based on the slider value while keeping inversion applied."""
        if self.current_image:
            contrast_factor = float(value)
            enhancer = ImageEnhance.Contrast(self.current_image)
            adjusted_image = enhancer.enhance(contrast_factor)
            self.rescale_and_display(adjusted_image)
            self.current_image = adjusted_image  # Update the stored image

    def show_histogram(self):
        # Display the histogram of the current image
        if self.current_image:
            # Convert the image to grayscale
            grayscale_image = self.current_image.convert("L")
            image_array = np.array(grayscale_image) / 255.0  # Normalize to [0, 1]

            # Create a mask (optional, here we use the entire image)
            mask = np.ones_like(image_array, dtype=bool)

            # Calculate the histogram
            histogram, bin_edges = np.histogram(image_array[mask], bins=256, range=(0.0, 1.0))

            # Configure and draw the histogram figure
            fig, ax = plt.subplots()
            ax.set_title("Grayscale Histogram")
            ax.set_xlabel("Grayscale value")
            ax.set_ylabel("Pixel count")
            ax.set_xlim([0.0, 1.0])
            ax.plot(bin_edges[0:-1], histogram)
            plt.show()

    def debug(self, message):
        if debugging:
            print(f"iMosflm: {message}")

def parse_arguments():
    parser = argparse.ArgumentParser(description='iMosflm Application')
    parser.add_argument('--ccp4i2', action='store_true', help='Autosaves an iMosflm session file upon exit')
    parser.add_argument('--debug', '-d', action='store_true', help='Creates a large output file for debugging purposes')
    parser.add_argument('--expert', '-e', action='store_true', help='Permits access to advanced detector settings')
    parser.add_argument('--fastload', '-f', action='store_true', help='Attempt to speed loading of many images offline')
    parser.add_argument('--version', '-v', action='store_true', help='Displays the program version and required Mosflm version, then exits')
    # Remove the explicit --help argument
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Set environment variables based on arguments
    if args.ccp4i2:
        os.environ['CCP4I2'] = '1'
        print("* CCP4i2 mode on - be VERY sure you want this")

    if args.debug:
        os.environ['MOSFLM_DEBUG'] = '1'
        print("* full debugging turned on")

    if args.expert:
        os.environ['EXPERTDETECTORSETTINGS'] = '1'
        print("* Expert mode on - be VERY sure you want this")

    if args.fastload:
        os.environ['FASTLOAD'] = '1'
        print("* FASTLOAD mode on")

    if args.version:
        print(f"* {os.environ.get('IMOSFLM_VERSION', 'Unknown version')}")
        sys.exit()

    # Initialize and run the application
    app = IMosflmApp()
    app.mainloop()

if __name__ == "__main__":
    main()