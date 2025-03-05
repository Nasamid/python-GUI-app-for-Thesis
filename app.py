import os
import numpy as np
from PIL import Image as PILImage

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

# Import Picamera2 (assumed installed and working on your Raspberry Pi)
from picamera2 import Picamera2

# Import TFLite interpreter – adjust to your installation (tflite_runtime or tensorflow.lite)
import tflite_runtime.interpreter as tflite

# ----------------------- Backend: TFLite Model and Camera Setup ----------------------- #
# In the main app we load the DenseNet-121 model using TFLite.
# You may need to adjust input image size, normalization, and label mapping.
# Also note that camera zoom or live preview methods may need to be adapted to your setup.

class RiceLeafApp(App):
    def build(self):
        # Force full screen (no window controls)
        Config.set('graphics', 'fullscreen', 'auto')
        
        # Initialize global variables
        self.image_counter = 1
        self.results = []  # List of dicts with keys: 'image' and 'diagnosis'
        
        # Initialize and start the Raspberry Pi camera
        self.camera = Picamera2()
        self.camera.preview_configuration.main.size = (2592, 1944)
        self.camera.preview_configuration.main.format = "RGB888"
        self.camera.configure("preview")
        self.camera.start()

        # Load the TFLite DenseNet-121 model
        self.load_model()

        # Build the ScreenManager and add our screens.
        self.sm = ScreenManager(transition=FadeTransition())
        self.sm.app = self  # Attach a reference to the app for access in screens
        self.sm.add_widget(HomeScreen(name='home'))
        self.sm.add_widget(TutorialScreen(name='tutorial'))
        self.sm.add_widget(MainScreen(name='main'))
        self.sm.add_widget(PreviousResultsScreen(name='results'))

        return self.sm

    def load_model(self):
        # Adjust model_path to the location of your TFLite model file.
        model_path = "rice_disease_model_V1-3.tflite"
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def classify_image(self, image_path):
        # Load and preprocess image.
        img = PILImage.open(image_path).resize((224, 224))
        img = np.array(img, dtype=np.float32)
        # Normalize as required by your model (this is a placeholder)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        # For demonstration, return the index of the highest score.
        diagnosis = np.argmax(output_data)
        # In practice, map the index to a class label.
        return f"Class {diagnosis}"

    def on_stop(self):
        # Stop the camera when the app is closed.
        self.camera.stop()

# ----------------------- Home Screen ----------------------- #
class HomeScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = FloatLayout()
        
        # Title at the top-middle.
        title = Label(text="[b]Rice Leaf Disease Detector[/b]",
                      markup=True,
                      font_size='32sp',
                      size_hint=(0.8, 0.1),
                      pos_hint={'center_x': 0.5, 'top': 1})
        layout.add_widget(title)
        
        # Clickable texts:
        doc_button = Button(text="Documentation",
                            size_hint=(0.2, 0.1),
                            pos_hint={'x': 0.1, 'top': 0.9})
        doc_button.bind(on_release=self.open_documentation)
        layout.add_widget(doc_button)
        
        how_button = Button(text="How it works",
                            size_hint=(0.2, 0.1),
                            pos_hint={'x': 0.7, 'top': 0.9})
        how_button.bind(on_release=self.go_to_tutorial)
        layout.add_widget(how_button)
        
        # Get Started Button
        get_started = Button(text="Get Started",
                             size_hint=(0.3, 0.1),
                             pos_hint={'center_x': 0.5, 'y': 0.4})
        get_started.bind(on_release=self.go_to_main)
        layout.add_widget(get_started)
        
        # "Don't show this again" checkbox with label.
        checkbox = CheckBox(size_hint=(0.1, 0.1), pos_hint={'x': 0.45, 'y': 0.3})
        layout.add_widget(checkbox)
        label = Label(text="Don't show this again",
                      size_hint=(0.3, 0.1),
                      pos_hint={'x': 0.55, 'y': 0.3})
        layout.add_widget(label)
        
        self.add_widget(layout)
    
    def open_documentation(self, instance):
        # Open your GitHub documentation page.
        import webbrowser
        webbrowser.open("https://github.com/Nasamid")
    
    def go_to_tutorial(self, instance):
        self.manager.current = 'tutorial'
    
    def go_to_main(self, instance):
        self.manager.current = 'main'

# ----------------------- Tutorial Screen ----------------------- #
class TutorialScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = FloatLayout()
        
        # "Skip" text at top-right.
        skip = Button(text="Skip",
                      size_hint=(0.2, 0.1),
                      pos_hint={'right': 1, 'top': 1})
        skip.bind(on_release=self.go_to_main)
        layout.add_widget(skip)
        
        # Carousel for the 6 tutorial images.
        self.carousel = Carousel(direction='right', loop=False)
        for i in range(1, 7):
            # Ensure tutorial images (tutorial1.jpg ... tutorial6.jpg) exist.
            img = Image(source=f"tutorial{i}.jpg", allow_stretch=True)
            # Bind touch to progress to next slide.
            img.bind(on_touch_down=self.next_slide)
            self.carousel.add_widget(img)
        layout.add_widget(self.carousel)
        self.add_widget(layout)
    
    def next_slide(self, instance, touch):
        if instance.collide_point(*touch.pos):
            if self.carousel.index < len(self.carousel.slides) - 1:
                self.carousel.load_next(mode='next')
            else:
                # After the last image, go to Main Page.
                self.manager.current = 'main'
    
    def go_to_main(self, instance):
        self.manager.current = 'main'

# ----------------------- Main Screen ----------------------- #
class MainScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.update_event = None
        
        # Use a FloatLayout to overlay multiple widgets.
        layout = FloatLayout()
        
        # Camera live feed: an Image widget that will be updated.
        self.camera_preview = Image(size_hint=(1, 1), pos_hint={'x': 0, 'y': 0})
        layout.add_widget(self.camera_preview)
        
        # Zoom bar on the left.
        self.zoom_slider = Slider(min=1, max=3, value=1,
                                  orientation='vertical',
                                  size_hint=(0.1, 0.5),
                                  pos_hint={'x': 0, 'center_y': 0.5})
        self.zoom_slider.bind(value=self.on_zoom_change)
        layout.add_widget(self.zoom_slider)
        
        # Image Gallery Button (leads to Previous Results page) at lower right.
        gallery_button = Button(text="Gallery",
                                size_hint=(0.2, 0.1),
                                pos_hint={'right': 1, 'y': 0})
        gallery_button.bind(on_release=self.go_to_gallery)
        layout.add_widget(gallery_button)
        
        # Question mark button at top-right (goes to Home page).
        home_button = Button(text="?",
                             size_hint=(0.1, 0.1),
                             pos_hint={'right': 1, 'top': 1})
        home_button.bind(on_release=self.go_home)
        layout.add_widget(home_button)
        
        # Red Capture Button on the right side.
        capture_button = Button(text="Capture",
                                background_color=(1, 0, 0, 1),
                                size_hint=(0.2, 0.1),
                                pos_hint={'right': 1, 'center_y': 0.5})
        capture_button.bind(on_release=self.capture_image)
        layout.add_widget(capture_button)
        
        self.add_widget(layout)
    
    def on_enter(self):
        # Start updating the live feed at ~30 FPS.
        self.update_event = Clock.schedule_interval(self.update_camera, 1.0/30.0)
    
    def on_leave(self):
        if self.update_event:
            self.update_event.cancel()
            self.update_event = None
    
    def update_camera(self, dt):
        try:
            # Capture a frame from the camera preview.
            # (Assumes that "capture_array" with stream "main" returns a NumPy array.)
            frame = self.manager.app.camera.capture_array("main")
            buf = frame.tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
            texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
            texture.flip_vertical()
            self.camera_preview.texture = texture
        except Exception as e:
            print("Camera update error:", e)
    
    def on_zoom_change(self, instance, value):
        # Adjust the camera zoom.
        # (Note: Replace with the actual API call for your camera if needed.)
        try:
            self.manager.app.camera.set_zoom(value)
        except Exception as e:
            print("Zoom update error:", e)
    
    def capture_image(self, instance):
        app = self.manager.app
        filename = f"captured_image_{app.image_counter}.jpg"
        image_path = os.path.join(os.getcwd(), filename)
        app.camera.capture_file(image_path)
        app.image_counter += 1
        
        # Run classification using the TFLite model.
        diagnosis = app.classify_image(image_path)
        # Save the result for display in the gallery.
        app.results.append({'image': image_path, 'diagnosis': diagnosis})
    
    def go_to_gallery(self, instance):
        self.manager.current = 'results'
    
    def go_home(self, instance):
        self.manager.current = 'home'

# ----------------------- Previous Results Screen ----------------------- #
class PreviousResultsScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = FloatLayout()
        # Carousel for swipable result slides.
        self.carousel = Carousel(direction='right', loop=False, size_hint=(1, 1))
        layout.add_widget(self.carousel)
        self.add_widget(layout)
    
    def on_pre_enter(self):
        # Refresh the carousel with current results.
        self.carousel.clear_widgets()
        for result in self.manager.app.results:
            slide = ResultSlide(result)
            self.carousel.add_widget(slide)

class ResultSlide(BoxLayout):
    def __init__(self, result, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        # Display the captured image.
        img = Image(source=result['image'], allow_stretch=True, size_hint=(1, 0.7))
        self.add_widget(img)
        # Display the diagnosis.
        diag_label = Label(text=f"Diagnosis: {result['diagnosis']}",
                           size_hint=(1, 0.1))
        self.add_widget(diag_label)
        # Chart button: when clicked, animate a flip to show recommendations.
        chart_button = Button(text="Chart", size_hint=(0.2, 0.1))
        chart_button.bind(on_release=self.flip_view)
        self.add_widget(chart_button)
    
    def flip_view(self, instance):
        # A simple flip animation; replace with more complex animation if desired.
        anim = Animation(rotation_y=90, duration=0.5) + Animation(rotation_y=0, duration=0.5)
        anim.start(self)
        # After the animation, replace content with recommendations (placeholder).
        self.clear_widgets()
        rec_label = Label(text="Recommendations:\n- Use proper fertilizer\n- Maintain humidity\nAccuracy: 90%",
                           halign="center")
        self.add_widget(rec_label)

# ----------------------- Run the Application ----------------------- #
if __name__ == '__main__':
    RiceLeafApp().run()
