import os
import json
import numpy as np
from PIL import Image as PILImage
import threading

from kivy.app import App
from kivy.config import Config
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.carousel import Carousel
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.checkbox import CheckBox
from kivy.uix.slider import Slider
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.animation import Animation
from kivy.properties import NumericProperty, StringProperty
from kivy.utils import get_color_from_hex
from kivy.uix.button import Button
from kivy.graphics import Color, RoundedRectangle
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.image import Image
from kivy.uix.screenmanager import Screen
from kivy.uix.scrollview import ScrollView
from kivy.uix.label import Label
from kivy.core.image import Image as CoreImage
from pdf2image import convert_from_path
from kivy.clock import Clock
from kivy.graphics import Color, Rectangle
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.floatlayout import FloatLayout
from rembg import remove
from picamera2 import Picamera2
import tflite_runtime.interpreter as tflite


#--Round Buttons--#
class ImageButton(ButtonBehavior, Image):
    pass

class RoundedButton(Button):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Remove the default background image so the color is visible..,l
        self.background_normal = ''
        self.background_color = [0, 1, 1, 1]  # Your desired color
        
        with self.canvas.before:
            # Draw a rounded rectangle with the same color.
            self.bg_color = Color(rgba=self.background_color)
            self.bg_rect = RoundedRectangle(pos=self.pos, size=self.size, radius=[15])
        
        # Update the position and size of the rounded rectangle when the button changes.
        self.bind(pos=self.update_rect, size=self.update_rect)

    def update_rect(self, *args):
        self.bg_rect.pos = self.pos
        self.bg_rect.size = self.size

# ----------------------- Backend: TFLite Model and Camera Setup ----------------------- #
class RiceLeafApp(App):
    def build(self):
        # Force full screen (no window controls)
        Config.set('graphics', 'fullscreen', 'auto')
        
        # Initialize global variables
        self.image_counter = 1
        self.results = []  # List of dicts with keys: 'image' and 'diagnosis'
        
        # Define gallery folder and JSON data file paths
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.gallery_folder = os.path.join(base_dir, "gallery")

        if not os.path.exists(self.gallery_folder):
            os.makedirs(self.gallery_folder)
        self.gallery_data_file = os.path.join(self.gallery_folder, "gallery_data.json")
        
        # Load persisted gallery data (if available)
        self.load_gallery_data()
        
        # Initialize and start the Raspberry Pi camera
# Initialize and start the Raspberry Pi cameras
        self.camera = Picamera2()

        # 1) Preview at low resolution for smooth high‑fps live view
        self.preview_conf = self.camera.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )

        # 2) Still at full resolution for capture
        self.still_conf   = self.camera.create_still_configuration(
            main={"size": (2592, 1944), "format": "RGB888"}
        )

        # Start in preview mode once
        self.camera.configure(self.preview_conf)
        self.camera.start()


        # Load the TFLite DenseNet-121 model and class labels
        self.load_model()

        # Build the ScreenManager and add our screens.
        self.sm = ScreenManager(transition=FadeTransition())
        self.sm.app = self  # Attach a reference to the app for access in screens
        self.sm.add_widget(HomeScreen(name='home'))
        self.sm.add_widget(TutorialScreen(name='tutorial'))
        self.sm.add_widget(MainScreen(name='main'))
        self.sm.add_widget(PreviousResultsScreen(name='results'))
        self.sm.add_widget(DocumentationScreen(name='documentation'))
        self.sm.add_widget(LoadingScreen(name='loading'))

        return self.sm

    def load_model(self):
        # Adjust model_path to the location of your TFLite model file.
        model_path = "ZAMBALI_rice_disease_model_V4.tflite"
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Load labels from file or define manually
        labels_path = "labels.txt"  # Ensure this file exists in the same directory
        if os.path.exists(labels_path):
            with open(labels_path, "r") as f:
                self.class_labels = [line.strip() for line in f.readlines()]
        else:
            # Define manually if file not available; adjust as needed.
            self.class_labels = ["Bacterial Blight", "Brown Spot", "Healthy", "Leaf Blast", "Not a Rice Leaf"]

    def load_gallery_data(self):
        if os.path.exists(self.gallery_data_file):
            try:
                with open(self.gallery_data_file, "r") as f:
                    self.results = json.load(f)
                # Update image_counter to avoid overwriting
                if self.results:
                    # Assuming filenames are of the format captured_image_<n>.jpg
                    indices = [int(item['raw_image'].split('_')[-1].split('.')[0]) for item in self.results if 'captured_image' in item['raw_image']]
                    self.image_counter = max(indices) + 1 if indices else 1
            except Exception as e:
                print("Error loading gallery data:", e)
                self.results = []
        else:
            self.results = []

    def save_gallery_data(self):
        try:
            with open(self.gallery_data_file, "w") as f:
                json.dump(self.results, f)
        except Exception as e:
            print("Error saving gallery data:", e)

    def classify_image(self, image_path):
        # Open the image and convert it to RGB (to ensure 3 channels)
        img = PILImage.open(image_path).convert("RGB").resize((224, 224))
        img = np.array(img, dtype=np.float32)
        img = img / 255.0  # normalize
        img = np.expand_dims(img, axis=0)
        
        # Run inference.
        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()
        
        # Get the output tensor and make a copy to avoid referencing internal data.
        output_data = self.interpreter.get_tensor(self.output_details[0]['index']).copy()
        # Assuming the output shape is [1, num_classes]
        output_flat = output_data[0]
        
        # Get the predicted class index and confidence.
        predicted_index = np.argmax(output_flat)
        confidence = output_flat[predicted_index] * 100  # as percentage
        
        # Build a dictionary for all class confidences (flatten the output).
        self.all_confidences = {label: float(conf * 100) for label, conf in zip(self.class_labels, output_flat)}
        
        # Retrieve the class name.
        if 0 <= predicted_index < len(self.class_labels):
            diagnosis = self.class_labels[predicted_index]
        else:
            diagnosis = "Unknown"
        
        return diagnosis, confidence

    def on_stop(self):
        # Stop the camera when the app is closed.
        self.camera.stop()
        # Save gallery data on exit
        self.save_gallery_data()

class LoadingScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = FloatLayout()

        # Set the white background
        with layout.canvas.before:
            Color(1, 1, 1, 1)  # White
            self.rect = Rectangle(pos=layout.pos, size=layout.size)
            layout.bind(size=self.update_rect, pos=self.update_rect)

        # Add a spinner image (ensure spinner.gif is available in your project folder)
        spinner = Image(source="spinner.gif",
                        size_hint=(0.2, 0.2),
                        pos_hint={'center_x': 0.5, 'center_y': 0.6})
        layout.add_widget(spinner)

        # Add a processing label with black text
        self.label = Label(text="Processing...",
                           font_size='32sp',
                           color=(0, 0, 0, 1),  # Black
                           size_hint=(0.8, 0.2),
                           pos_hint={'center_x': 0.5, 'center_y': 0.4})
        layout.add_widget(self.label)

        self.add_widget(layout)

    def update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

#--Gradient Rounded Buttons--#
class GradientRoundedButton(Button):
    def __init__(self, gradient_start=(1, 1, 1, 1), gradient_end=(0, 0, 0, 1), **kwargs):
        super().__init__(**kwargs)
        # Save gradient colors.
        self.gradient_start = gradient_start
        self.gradient_end = gradient_end
        # Remove the default background.
        self.background_normal = ''
        self.background_color = [0, 0, 0, 0]

        with self.canvas.before:
            # Initialize with a placeholder radius.
            self.rect = RoundedRectangle(pos=self.pos, size=self.size, radius=[self.height / 2] * 4)
        self.bind(pos=self.update_rect, size=self.update_rect)
        self._update_texture()

    def _update_texture(self):
        # Create a horizontal gradient texture: 64 pixels wide, 1 pixel tall.
        width, height = 64, 1
        texture = Texture.create(size=(width, height), colorfmt='rgba')
        buf = []
        for x in range(width):
            t = x / float(width - 1)
            # Linear interpolation for each channel.
            r = self.gradient_start[0] * (1 - t) + self.gradient_end[0] * t
            g = self.gradient_start[1] * (1 - t) + self.gradient_end[1] * t
            b = self.gradient_start[2] * (1 - t) + self.gradient_end[2] * t
            a = self.gradient_start[3] * (1 - t) + self.gradient_end[3] * t
            buf.extend([int(r * 255), int(g * 255), int(b * 255), int(a * 255)])
        buf = bytes(buf)
        texture.blit_buffer(buf, colorfmt='rgba', bufferfmt='ubyte')
        texture.wrap = 'clamp_to_edge'
        self.rect.texture = texture

    def update_rect(self, *args):
        # Update rectangle position and size.
        self.rect.pos = self.pos
        self.rect.size = self.size
        # Set radius to half the height for a fully rounded (pill) shape.
        self.rect.radius = [self.height / 2] * 4
        # Stretch the texture so that the gradient spans the whole button.
        if self.rect.texture:
            self.rect.texture.uvsize = (self.width / self.rect.texture.width, self.height / self.rect.texture.height)

# ----------------------- Home Screen ----------------------- #
class HomeScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = FloatLayout()

        background = Image(source="bg.png",
                           allow_stretch=True,
                           keep_ratio=False,
                           size_hint=(1, 1),
                           pos_hint={'x': 0, 'y': 0},
                           disabled=True)  # This makes it unclickable.
        layout.add_widget(background)

        logo = Image(source="logo.png",
                     allow_stretch=False,
                     keep_ratio=True,
                     size_hint=(0.7, 0.7),
                     pos_hint={'x': -0.07, 'y': 0.15},
                     disabled=True)
        layout.add_widget(logo)
        
        title = Label(
            text="[i][/i]",
            markup=True,
            font_name="Poppins-BoldItalic.ttf",  # Ensure the TTF file is available.
            font_size='42sp',
            color=(0.1, 0.1, 0.2, 1),  # Deep navy tone (RGBA).
            size_hint=(0.8, 0.1),
            pos_hint={'center_x': 0.5, 'top': 0.8}
        )
        layout.add_widget(title)

        # Documentation Button with a gradient from red to yellow.
        doc_button = GradientRoundedButton(
            text="Documentation",
            size_hint=(0.3, 0.2),
            pos_hint={'x': 0.6, 'top': 0.6},
            gradient_start=(149/255, 199/255, 52/255, 1),  # Converted fromrgba(87, 156, 30, 0.96)
            gradient_end=(75/255, 133/255, 59/255, 1)      # Converted fromrgba(31, 86, 8, 0.94)
        )
        doc_button.bind(on_release=self.open_documentation)
        layout.add_widget(doc_button)
        
        # How It Works Button with a gradient from cyan to blue.
        how_button = GradientRoundedButton(
            text="How it works",
            size_hint=(0.3, 0.2),
            pos_hint={'x': 0.6, 'top': 0.35},
            gradient_start=(149/255, 199/255, 52/255, 1),  # Converted fromrgba(110, 175, 50, 0.97)
            gradient_end=(75/255, 133/255, 59/255, 1)      # Converted from #4B853B
        )
        how_button.bind(on_release=self.go_to_tutorial)
        layout.add_widget(how_button)
        
        # Get Started Button with a gradient from yellow to orange.
        get_started = GradientRoundedButton(
            text="START",
            size_hint=(0.35, 0.2),
            pos_hint={'x': 0.6, 'top': 0.85},
            gradient_start=(65/255, 123/255, 20/255, 0.94),  # Converted fromrgba(65, 123, 20, 0.94)
            gradient_end=(31/255, 86/255, 8/255, 0.94)      # Converted from #4B853B
        )
        get_started.bind(on_release=self.go_to_main)
        layout.add_widget(get_started)
        
        self.add_widget(layout)
        
    def go_to_tutorial(self, instance):
        self.manager.current = 'tutorial'
    
    def go_to_main(self, instance):
        self.manager.current = 'main'

    def open_documentation(self, instance):
        self.manager.current = 'documentation'

    def go_to_tutorial(self, instance):
        self.manager.current = 'tutorial'

    def go_to_main(self, instance):
        self.manager.current = 'main'

#------------------------ Documentation Screen-----------------------#
class DocumentationScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Create a vertical BoxLayout
        layout = BoxLayout(orientation='vertical', padding=20, spacing=20)
        
        # Set a white background by adding a canvas instruction.
        with self.canvas.before:
            Color(1, 1, 1, 1)  # White color
            self.rect = Rectangle(pos=self.pos, size=self.size)
            self.bind(pos=self.update_rect, size=self.update_rect)
        
        # Add the QR code image
        qr_image = Image(
            source="qr_code.png",  # Path to your saved QR code image
            allow_stretch=True,
            keep_ratio=True,
            size_hint=(1.2, 0.6)
        )
        layout.add_widget(qr_image)
        
        # Add label with text "Scan Me to see documentation"
        info_label = Label(
            text="SCAN QR CODE TO SEE DOCUMENTATION",
            font_size='24sp',
            color=(0, 0, 0, 1),  # Black text
            size_hint=(1, 0.2)
        )
        layout.add_widget(info_label)

        # Add a Close button at the top
        close_button = Button(
            text="Close",
            size_hint=(0.2, 0.15),
            pos_hint={'right': 1, 'y': 0}
        )
        close_button.bind(on_release=self.go_home)
        layout.add_widget(close_button)
        
        self.add_widget(layout)
    
    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size
        
    def go_home(self, instance):
        self.manager.current = 'home'

# ----------------------- Tutorial Screen ----------------------- #
class TutorialScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = FloatLayout()
        
        # Carousel for the 6 tutorial images.
        self.carousel = Carousel(direction='right', loop=False)
        for i in range(1, 7):
            img = Image(source=f"tutorial{i}.jpg", allow_stretch=True)
            img.bind(on_touch_down=self.next_slide)
            self.carousel.add_widget(img)
        layout.add_widget(self.carousel)
        self.add_widget(layout)

        # "Skip" text at top-right.
        skip = Button(text="SKIP",
                      size_hint=(0.2, 0.1),
                      pos_hint={'right': 1, 'top': 1},
                      background_color=[1, 1, 0, 1])
        skip.bind(on_release=self.go_to_main)
        layout.add_widget(skip)
    
    def next_slide(self, instance, touch):
        if instance.collide_point(*touch.pos):
            if self.carousel.index < len(self.carousel.slides) - 1:
                self.carousel.load_next(mode='next')
            else:
                self.manager.current = 'main'
    
    def go_to_main(self, instance):
        self.manager.current = 'main'

# ----------------------- Tutorial Screen ----------------------- #
class TutorialScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = FloatLayout()
        
        # Carousel for the 6 tutorial images.
        self.carousel = Carousel(direction='right', loop=False)
        for i in range(1, 7):
            img = Image(source=f"tutorial{i}.jpg", allow_stretch=True)
            img.bind(on_touch_down=self.next_slide)
            self.carousel.add_widget(img)
        layout.add_widget(self.carousel)
        self.add_widget(layout)

        # "Skip" text at top-right.
        skip = Button(text="SKIP",
                      size_hint=(0.2, 0.1),
                      pos_hint={'right': 1, 'top': 1},
                      background_color = [1,1,0,1])
        skip.bind(on_release=self.go_to_main)
        layout.add_widget(skip)
    
    def next_slide(self, instance, touch):
        if instance.collide_point(*touch.pos):
            if self.carousel.index < len(self.carousel.slides) - 1:
                self.carousel.load_next(mode='next')
            else:
                self.manager.current = 'main'
    
    def go_to_main(self, instance):
        self.manager.current = 'main'


# ----------------------- Main Screen ----------------------- #
class MainScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_processing = False
        self.update_event = None

        # Layout & widgets
        layout = FloatLayout()
        self.camera_preview = Image(size_hint=(1,1), pos_hint={'x':0,'y':0})
        layout.add_widget(self.camera_preview)

        self.zoom_slider = Slider(min=1, max=3, value=1,
                                  orientation='vertical',
                                  size_hint=(0.1, 0.5),
                                  pos_hint={'x': 0, 'center_y': 0.5})
        self.zoom_slider.bind(value=self.on_zoom_change)
        layout.add_widget(self.zoom_slider)
        
        gallery_button = ImageButton(
            source="gallery.png",  # Replace with your image path
            size_hint=(0.1, 0.1),
            pos_hint={'right': 1, 'y': 0}
        )
        gallery_button.bind(on_release=self.go_to_gallery)
        layout.add_widget(gallery_button)
        
        home_button = ImageButton(source="favicon.ico",
            size_hint=(0.1, 0.1),
            pos_hint={'right': 1, 'top': 1})
        home_button.bind(on_release=self.go_home)
        layout.add_widget(home_button)
        
        capture_button = ImageButton(source="capture.png",
            size_hint=(0.1, 0.15),
            pos_hint={'right': 1, 'center_y': 0.5})
        capture_button.bind(on_release=self.capture_image)
        layout.add_widget(capture_button)
        
        self.add_widget(layout)
    
    def on_enter(self):
        # Just schedule the preview refresh at 60 fps — camera is already running
        self.update_event = Clock.schedule_interval(self.update_camera, 1.0 / 60.0)
    
    def on_leave(self):
        if self.update_event:
            self.update_event.cancel()
            self.update_event = None
    
    def update_camera(self, dt):
        try:
            frame = self.manager.app.camera.capture_array("main")
            buf = frame.tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
            texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
            texture.flip_vertical()
            self.camera_preview.texture = texture
        except Exception as e:
            print("Camera update error:", e)
    
    def on_zoom_change(self, instance, value):
        try:
            # Get the full resolution from the camera's preview configuration.
            # (Assuming the preview resolution is the full sensor resolution you want to zoom on)
            sensor_width, sensor_height = self.manager.app.camera.preview_configuration.main.size
            
            # Calculate the new width and height by dividing by the zoom factor.
            new_w = int(sensor_width / value)
            new_h = int(sensor_height / value)
            
            # Calculate the top-left corner (x, y) to center the crop.
            x = int((sensor_width - new_w) / 2)
            y = int((sensor_height - new_h) / 2)
            
            # Apply the crop rectangle via the camera controls.
            self.manager.app.camera.set_controls({"ScalerCrop": (x, y, new_w, new_h)})
        except Exception as e:
            print("Zoom update error:", e)

    def capture_image(self, instance):
        # Prevent duplicate processing if a capture is already in progress.
        if getattr(self, 'is_processing', False):
            return
        self.is_processing = True

        app = self.manager.app

        # Switch to the loading screen
        self.manager.current = 'loading'
        
        # Save the raw image in the gallery folder.
        filename = f"captured_image_{app.image_counter}.jpg"
        raw_image_path = os.path.join(app.gallery_folder, filename)
        #app.camera.capture_file(raw_image_path)
        app.image_counter += 1

        app.camera.switch_mode_and_capture_file(self.still_conf, raw_image_path)

        def process_image():
            try:
                # --- Background Removal using rembg tailored for rice leaves ---
                # Open the raw image in binary mode.
                with open(raw_image_path, 'rb') as i:
                    input_data = i.read()
                # Remove the background using rembg.
                result_data = remove(input_data)
                # Save the processed image as PNG (to preserve transparency).
                processed_filename = f"processed_{filename[:-4]}.png"  # change extension to PNG
                processed_path = os.path.join(app.gallery_folder, processed_filename)
                with open(processed_path, 'wb') as o:
                    o.write(result_data)

                # --- Classification ---
                # Classify the processed image (background removed).
                diagnosis, conf = app.classify_image(processed_path)

                # Save the result as a dictionary with both raw and processed image paths.
                app.results.insert(0, {
                    'raw_image': raw_image_path,
                    'processed_image': processed_path,
                    'diagnosis': diagnosis,
                    'confidence': conf,
                    "all_confidences": app.all_confidences
                })
                # Persist the updated results.
                app.save_gallery_data()
            except Exception as e:
                print("Error in process_image:", e)
            finally:
                # Clear the processing flag on the main thread.
                Clock.schedule_once(lambda dt: setattr(self, 'is_processing', False), 0)
                # Switch to the results screen from the main thread after processing.
                Clock.schedule_once(lambda dt: setattr(self.manager, 'current', 'results'), 0)
        
        # Run the processing in a separate thread.
        threading.Thread(target=process_image).start()

    def go_to_gallery(self, instance):
        self.manager.current = 'results'
    
    def go_home(self, instance):
        self.manager.current = 'home'

# ----------------------- Previous Results Screen ----------------------- #
class PreviousResultsScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = FloatLayout()
        
        self.carousel = Carousel(direction='right', loop=False, size_hint=(1, 0.9), pos_hint={'x': 0, 'y': 0.1})
        layout.add_widget(self.carousel)
        
        back_button = ImageButton(source="back.png",
                             size_hint=(0.2, 0.1),
                             pos_hint={'right': 1, 'y': 0.01})
        back_button.bind(on_release=self.go_back)
        layout.add_widget(back_button)
        
        self.add_widget(layout)
    
    def on_pre_enter(self):
        self.carousel.clear_widgets()
        # Load persisted images from the gallery.
        for result in self.manager.app.results:
            slide = ResultSlide(result)
            self.carousel.add_widget(slide)
    
    def go_back(self, instance):
        self.manager.current = 'main'


class ResultSlide(BoxLayout):
    rotation_y = NumericProperty(0)  # Needed for animation
    raw_image = StringProperty("")
    processed_image = StringProperty("")
    diagnosis = StringProperty("")
    confidence = NumericProperty(0)

    def __init__(self, result, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        # Store raw image, processed image, diagnosis, and confidence.
        self.raw_image = result.get('raw_image', "")
        self.processed_image = result.get('processed_image', "")
        self.diagnosis = result.get('diagnosis', "")
        self.confidence = float(result.get('confidence', 0))
        self.all_confidences = result.get("all_confidences", {})
        self.display_image_view()

    def display_image_view(self, instance=None):
        self.clear_widgets()
        
        # Create a horizontal layout for images.
        images_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.7))
        
        proc_img = Image(source=self.processed_image, allow_stretch=True)
        raw_img = Image(source=self.raw_image, allow_stretch=True)
        images_layout.add_widget(proc_img)
        images_layout.add_widget(raw_img)
        self.add_widget(images_layout)
        
        diag_text = f"Diagnosis: {self.diagnosis}"
        if self.confidence:
            diag_text += f" ({self.confidence:.1f}%)"
        diag_label = Label(text=diag_text, size_hint=(1, 0.1))
        self.add_widget(diag_label)
        
        more_info_button = Button(text="More Info", size_hint=(0.2, 0.1))
        more_info_button.bind(on_release=self.flip_view)
        self.add_widget(more_info_button)


    def flip_view(self, instance):
        """Flips to the 'More Info' screen with class confidence levels and recommendations."""
        anim = Animation(rotation_y=90, duration=0.3) + Animation(rotation_y=0, duration=0.3)
        anim.start(self)
        self.clear_widgets()

        # Show confidence levels for all classes
        if hasattr(self, 'all_confidences') and self.all_confidences:
            conf_text = "[b]Confidence Levels:[/b]\n"
            for label, conf in self.all_confidences.items():
                conf_text += f"{label}: {conf:.2f}%\n"
        else:
            conf_text = "Confidence data not available.\n"

        conf_label = Label(text=conf_text.strip(), markup=True, halign="left", size_hint=(1, 0.3))
        self.add_widget(conf_label)

        # Diagnosis-based recommendations
        if self.diagnosis == "Bacterial Blight":
            eng_msg = ("Bacterial Blight detected.\n"
                    "- Remove and destroy severely infected plants.\n"
                    "- Apply a copper-based bactericide as recommended.\n"
                    "- Maintain strict field hygiene.")
            fil_msg = ("Nadiskubre ang Bacterial Blight.\n"
                    "- Alisin at sirain ang labis na apektadong halaman.\n"
                    "- Gumamit ng copper-based bactericide.\n"
                    "- Panatilihin ang kalinisan ng taniman.")
        elif self.diagnosis in ["Blast", "Leaf Blast"]:
            eng_msg = ("Leaf Blast detected.\n"
                    "- Apply an effective fungicide.\n"
                    "- Adjust fertilizer use to avoid excessive growth.\n"
                    "- Improve plant spacing to enhance air circulation.\n"
                    "- Consult local agricultural services for guidance.")
            fil_msg = ("Nadiskubre ang Leaf Blast.\n"
                    "- Gamitin ng fungicide.\n"
                    "- Baguhin ang pataba para hindi magsobra ang paglaki.\n"
                    "- Ayusin ang pagitan ng mga halaman para sa mas maayos na sirkulasyon ng hangin.\n"
                    "- Kumonsulta sa lokal na agricultural service para sa payo.")
        elif self.diagnosis == "Brown Spot":
            eng_msg = ("Brown Spot detected.\n"
                    "- Use an appropriate fungicide.\n"
                    "- Avoid over-fertilization.\n"
                    "- Remove dead plant material to reduce spread.\n"
                    "- Follow good cultural practices.")
            fil_msg = ("Nadiskubre ang Brown Spot.\n"
                    "- Gamitin ang tamang fungicide.\n"
                    "- Iwasan ang paglalagay ng sobrang pataba.\n"
                    "- Alisin ang mga patay na bahagi ng halaman.\n"
                    "- Panatilihin ang tamang pamamahala ng taniman.")
        elif self.diagnosis == "Healthy":
            eng_msg = ("Your rice leaf appears healthy.\n"
                    "- Keep up your current cultivation practices and monitor regularly.")
            fil_msg = ("Malusog ang iyong dahon ng palay.\n"
                    "- Ipagpatuloy ang kasalukuyang pamamaraan at regular na i-monitor ang taniman.")
        elif self.diagnosis == "Not a Rice Leaf":
            eng_msg = ("The image does not seem to be of a rice leaf.\n"
                    "- Please scan a proper rice leaf for an accurate diagnosis.")
            fil_msg = ("Ang larawan ay hindi mukhang dahon ng palay.\n"
                    "- Mangyaring mag-scan ng wastong dahon ng palay para sa maayos na diagnosis.")
        else:
            eng_msg = "No specific recommendations available."
            fil_msg = ""

        if fil_msg:
            fil_msg = f"[i]{fil_msg}[/i]"

        combined_text = f"Recommendations:\n{eng_msg}\n\n{fil_msg}"
        rec_label = Label(text=combined_text, markup=True, halign="center", size_hint=(1, 0.6))
        self.add_widget(rec_label)

        back_button = Button(text="See Image", size_hint=(0.3, 0.1))
        back_button.bind(on_release=self.display_image_view)
        self.add_widget(back_button)


# ----------------------- Run the Application ----------------------- #
if __name__ == '__main__':
    RiceLeafApp().run()