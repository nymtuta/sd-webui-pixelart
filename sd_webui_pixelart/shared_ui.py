"""Shared UI components and processing logic for pixelart scripts."""

import inspect
import gradio as gr
from PIL import features, Image

from modules import ui_components

from sd_webui_pixelart.utils import (
    DITHER_METHODS, QUANTIZATION_METHODS, downscale_image, limit_colors, 
    resize_image, convert_to_grayscale, convert_to_black_and_white, 
    parse_color_text, create_palette_from_colors, read_palette_file
)

def _colors_to_hex_text(colors):
    return ", ".join([f"#{r:02X}{g:02X}{b:02X}" for r, g, b in colors])

def _merge_and_dedup_colors(existing_text, new_colors):
    """Merge existing text colors with new_colors (list of RGB tuples), remove duplicates preserving order."""
    existing = parse_color_text(existing_text) or []
    combined = existing + (new_colors or [])
    seen = set()
    result = []
    for c in combined:
        tup = (int(c[0]), int(c[1]), int(c[2]))
        if tup not in seen:
            seen.add(tup)
            result.append(tup)
    return result

def on_palette_upload(file_path, existing_text=""):
    """
    Unified callback for palette file upload. Handles both image and text files.
    Detects file type and extracts colors accordingly.
    """
    if not file_path:
        return existing_text, gr.update(value=None), gr.update(value="", visible=False)
    
    # Try to open as image first
    try:
        img = Image.open(file_path)
        # Successfully opened as image - extract colors
        img = img.convert('RGB')
        colors_with_counts = img.getcolors(maxcolors=img.width * img.height)
        if colors_with_counts:
            # sort by count desc to prioritize dominant colors
            colors_sorted = [c for _, c in sorted(colors_with_counts, key=lambda x: -x[0])]
        else:
            # fallback: sample up to 1024 pixels and dedupe
            sample = set()
            w, h = img.size
            max_sample = min(w * h, 1024)
            for i in range(max_sample):
                px = img.getpixel((i % w, i // w))
                sample.add((px[0], px[1], px[2]))
            colors_sorted = list(sample)
        
        merged = _merge_and_dedup_colors(existing_text, colors_sorted)
        color_text = _colors_to_hex_text(merged)
        return color_text, gr.update(value=None), gr.update(value="", visible=False)
    except Exception as img_error:
        # Not an image, try as text file
        colors, error = read_palette_file(file_path)
        if error:
            # Neither valid image nor valid text file
            return existing_text, gr.update(), gr.update(value=f"<div style='color:red'>File is neither a valid image nor a valid text palette file.<br>Image error: {img_error}<br>Text error: {error}</div>", visible=True)
        
        # Successfully parsed as text file
        merged = _merge_and_dedup_colors(existing_text, colors)
        color_text = _colors_to_hex_text(merged)
        return color_text, gr.update(value=None), gr.update(value="", visible=False)

def create_pixelart_ui(open_accordion=False, label="Pixel art"):
    """
    Create the complete pixel art UI.
    Returns a dict of all components including 'enabled'.
    
    Args:
        open_accordion: Whether accordion starts open
        label: Label for the accordion
    """
    quantization_methods = ['Median cut', 'Maximum coverage', 'Fast octree']
    dither_methods = ['None', 'Floyd-Steinberg']

    if features.check_feature("libimagequant"):
        quantization_methods.insert(0, "libimagequant")

    components = {}

    with ui_components.InputAccordion(open_accordion, label=label) as enabled:
        components['enabled'] = enabled
        
        with gr.Row():
            components['downscale'] = gr.Slider(label="Downscale", minimum=1, maximum=64, step=1, value=4)
            components['need_rescale'] = gr.Checkbox(label="Rescale to original size", value=False)
        
        with gr.Tabs():
            with gr.TabItem("Color"):
                components['enable_color_limit'] = gr.Checkbox(label="Enable", value=False)
                components['number_of_colors'] = gr.Slider(label="Palette Size", minimum=1, maximum=256, step=1, value=16)
                components['quantization_method'] = gr.Radio(choices=quantization_methods, value=quantization_methods[0], label='Colors quantization method')
                components['dither_method'] = gr.Radio(choices=dither_methods, value=dither_methods[0], label='Colors dither method')
                components['use_k_means'] = gr.Checkbox(label="Enable k-means for color quantization", value=True)
            with gr.TabItem("Grayscale"):
                components['is_grayscale'] = gr.Checkbox(label="Enable", value=False)
                components['number_of_shades'] = gr.Slider(label="Palette Size", minimum=1, maximum=256, step=1, value=16)
                components['quantization_method_grayscale'] = gr.Radio(choices=quantization_methods, value=quantization_methods[0], label='Colors quantization method')
                components['dither_method_grayscale'] = gr.Radio(choices=dither_methods, value=dither_methods[0], label='Colors dither method')
                components['use_k_means_grayscale'] = gr.Checkbox(label="Enable k-means for color quantization", value=True)
            with gr.TabItem("Black and white"):
                with gr.Row():
                    components['is_black_and_white'] = gr.Checkbox(label="Enable", value=False)
                    components['is_inversed_black_and_white'] = gr.Checkbox(label="Inverse", value=False)
                with gr.Row():
                    components['black_and_white_threshold'] = gr.Slider(label="Threshold", minimum=1, maximum=256, step=1, value=128)
            with gr.TabItem("Custom color palette"):
                components['use_color_palette'] = gr.Checkbox(label="Enable", value=False)
                with gr.Column():
                    components['palette_text'] = gr.Textbox(
                        placeholder="Hex: #FF0000, #00FF00, #0000FF\nRGB: 255,0,0 or (255,0,0)\nSeparate by comma or newline",
                        lines=4
                    )
                    # Error message display (red text, positioned between textbox and file input)
                    components['palette_message'] = gr.HTML(value="", visible=False)
                    with gr.Row():
                        components['palette_upload'] = gr.File(
                            label="Add colors from image or text file",
                            file_count="single",
                            type="filepath",
                            scale=1
                        )
                    # Connect file upload to automatically append to text and clear file input
                    components['palette_upload'].change(
                        fn=on_palette_upload,
                        inputs=[components['palette_upload'], components['palette_text']],
                        outputs=[components['palette_text'], components['palette_upload'], components['palette_message']]
                    )
                components['dither_method_palette'] = gr.Radio(choices=dither_methods, value=dither_methods[0], label='Colors dither method')

    return components

def create_pixelart_ui_with_keys(open_accordion=False, label="Pixel art"):
    """
    Create the complete pixel art UI and return both components dict and ordered keys list.
    
    Args:
        open_accordion: Whether accordion starts open
        label: Label for the accordion
        
    Returns: (components_dict, keys_list)
    """
    components = create_pixelart_ui(open_accordion=open_accordion, label=label)
    return components, list(components.keys())

def extract_process_params(**all_params):
    """
    Automatically extract only the parameters that process_image() accepts.
    Uses introspection to match parameters, so no manual updates needed when
    process_image() signature changes.
    """
    # Get the parameter names that process_image accepts (excluding 'original_image')
    sig = inspect.signature(process_image)
    valid_params = set(sig.parameters.keys()) - {'original_image'}
    
    return {k: v for k, v in all_params.items() if k in valid_params}

def process_image(
    original_image,
    downscale,
    need_rescale,
    enable_color_limit,
    number_of_colors,
    quantization_method,
    dither_method,
    use_k_means,
    is_grayscale,
    number_of_shades,
    quantization_method_grayscale,
    dither_method_grayscale,
    use_k_means_grayscale,
    use_color_palette,
    palette_text,
    dither_method_palette,
    is_black_and_white,
    is_inversed_black_and_white,
    black_and_white_threshold
):
    dither = DITHER_METHODS[dither_method]
    quantize = QUANTIZATION_METHODS[quantization_method]
    dither_grayscale = DITHER_METHODS[dither_method_grayscale]
    quantize_grayscale = QUANTIZATION_METHODS[quantization_method_grayscale]
    dither_palette = DITHER_METHODS[dither_method_palette]

    original_width, original_height = original_image.size

    if original_image.mode != "RGB":
        new_image = original_image.convert("RGB")
    else:
        new_image = original_image

    new_image = downscale_image(new_image, downscale)

    if use_color_palette:
        palette_to_use = None

        # Try to get colors from text input
        if palette_text and palette_text.strip():
            palette_colors_list = parse_color_text(palette_text)

            # If we have colors from text, create palette
            if palette_colors_list:
                palette_to_use = create_palette_from_colors(palette_colors_list)

        # Apply the palette if we have one
        if palette_to_use is not None:
            new_image = limit_colors(
                image=new_image,
                palette=palette_to_use,
                dither=dither_palette
            )

    if is_black_and_white:
        new_image = convert_to_black_and_white(new_image, black_and_white_threshold, is_inversed_black_and_white)

    if is_grayscale:
        new_image = convert_to_grayscale(new_image)
        new_image = limit_colors(
            image=new_image,
            limit=int(number_of_shades),
            quantize=quantize_grayscale,
            dither=dither_grayscale,
            use_k_means=use_k_means_grayscale
        )

    if enable_color_limit:
        new_image = limit_colors(
            image=new_image,
            limit=int(number_of_colors),
            quantize=quantize,
            dither=dither,
            use_k_means=use_k_means
        )

    if need_rescale:
        new_image = resize_image(new_image, (original_width, original_height))

    return new_image.convert('RGBA')
