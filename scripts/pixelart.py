import modules.scripts as scripts
from modules import images
from modules.shared import opts

from sd_webui_pixelart.shared_ui import create_pixelart_ui_with_keys, process_image, extract_process_params

class Script(scripts.Script):
    def title(self):
        return "Pixel art"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        components, keys = create_pixelart_ui_with_keys()
        # Store keys for use in postprocess
        self._component_keys = keys
        return list(components.values())

    def postprocess(self, p, processed, *args):
        # Map positional arguments to named parameters using the component key order
        kwargs = dict(zip(self._component_keys, args))
        
        if not kwargs.get('enabled'):
            return

        # Automatically extract parameters that process_image() needs from all kwargs
        process_params = extract_process_params(**kwargs)

        for i in range(len(processed.images)):
            pixel_image = process_image(processed.images[i], **process_params)
            processed.images.append(pixel_image)

            images.save_image(pixel_image, p.outpath_samples, "pixel", processed.seed + i, processed.prompt, opts.samples_format, info= processed.info, p=p)

        return processed
