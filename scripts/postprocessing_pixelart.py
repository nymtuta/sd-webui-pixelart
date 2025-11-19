from modules import scripts_postprocessing

from sd_webui_pixelart.shared_ui import create_pixelart_ui_with_keys, process_image, extract_process_params

class ScriptPostprocessingUpscale(scripts_postprocessing.ScriptPostprocessing):
    name = "Pixel art"
    order = 20005
    model = None

    def ui(self):
        components, keys = create_pixelart_ui_with_keys()
        # Store keys for use in process
        self._component_keys = keys
        return components

    def process(self, pp: scripts_postprocessing.PostprocessedImage, **kwargs):
        # Postprocessing framework passes components as keyword arguments
        if not kwargs.get('enabled'):
            return

        # Automatically extract parameters that process_image() needs from all kwargs
        process_params = extract_process_params(**kwargs)

        pp.image = process_image(pp.image, **process_params)
