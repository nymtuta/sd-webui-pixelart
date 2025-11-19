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
    Active tab determines which processing is applied (no enable checkboxes needed).
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
        
        gr.Markdown("**Color palette clamping method:**")
        
        # State to track which tab is selected (Tabs component itself can't be an input)
        components['active_tab'] = gr.State(value=0)
        
        with gr.Tabs() as tabs:
            
            with gr.TabItem("None", id="none"):
                gr.Markdown("No color palette clamping will be applied. Only downscaling.")
            
            with gr.TabItem("From current image", id="color"):
                components['number_of_colors'] = gr.Slider(label="Palette Size", minimum=1, maximum=256, step=1, value=16, interactive=True)
                components['quantization_method'] = gr.Radio(choices=quantization_methods, value=quantization_methods[0], label='Colors quantization method', interactive=True)
                components['dither_method'] = gr.Radio(choices=dither_methods, value=dither_methods[0], label='Colors dither method', interactive=True)
                components['use_k_means'] = gr.Checkbox(label="Enable k-means for color quantization", value=True, interactive=True)
            
            with gr.TabItem("Grayscale", id="grayscale"):
                components['number_of_shades'] = gr.Slider(label="Number of shades", minimum=1, maximum=256, step=1, value=16, interactive=True)
                components['quantization_method_grayscale'] = gr.Radio(choices=quantization_methods, value=quantization_methods[0], label='Colors quantization method', interactive=True)
                components['dither_method_grayscale'] = gr.Radio(choices=dither_methods, value=dither_methods[0], label='Colors dither method', interactive=True)
                components['use_k_means_grayscale'] = gr.Checkbox(label="Enable k-means for color quantization", value=True, interactive=True)
            
            with gr.TabItem("Black and white", id="black_and_white"):
                components['is_inversed_black_and_white'] = gr.Checkbox(label="Inverse", value=False, interactive=True)
                components['black_and_white_threshold'] = gr.Slider(label="Threshold", minimum=1, maximum=256, step=1, value=128, interactive=True)
            
            with gr.TabItem("Custom", id="custom_palette"):
                components['palette_text'] = gr.Textbox(
                    placeholder="Hex: #FF0000, #00FF00, #0000FF\nRGB: 255,0,0 or (255,0,0)\nSeparate by comma or newline",
                    lines=4,
                    interactive=True
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
                components['dither_method_palette'] = gr.Radio(choices=dither_methods, value=dither_methods[0], label='Colors dither method', interactive=True)
        
        # Update state when tab changes - use SelectData to get the selected tab index
        def update_tab_state(evt: gr.SelectData):
            return evt.index
        
        tabs.select(fn=update_tab_state, inputs=[], outputs=[components['active_tab']])

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
    active_tab,
    number_of_colors,
    quantization_method,
    dither_method,
    use_k_means,
    number_of_shades,
    quantization_method_grayscale,
    dither_method_grayscale,
    use_k_means_grayscale,
    palette_text,
    dither_method_palette,
    is_inversed_black_and_white,
    black_and_white_threshold
):
    # Determine which processing to apply based on active tab
    # active_tab is a Gradio Tabs component - use its selected property or default to first tab
    # For now, we'll use a simple approach: check which tab's ID matches
    
    original_width, original_height = original_image.size

    if original_image.mode != "RGB":
        new_image = original_image.convert("RGB")
    else:
        new_image = original_image

    new_image = downscale_image(new_image, downscale)

    # Apply processing based on active tab
    # active_tab can be tab index (int) or tab ID (string)
    # Tabs: None (0/none), From current image (1/color), Grayscale (2/grayscale), 
    #       Black and white (3/black_and_white), Custom (4/custom_palette)
    
    if active_tab == 0 or active_tab == "none":
        # No color palette clamping - just return the downscaled image
        pass
    
    elif active_tab == 1 or active_tab == "color":  # From current image tab
        dither = DITHER_METHODS[dither_method]
        quantize = QUANTIZATION_METHODS[quantization_method]
        new_image = limit_colors(
            image=new_image,
            limit=int(number_of_colors),
            quantize=quantize,
            dither=dither,
            use_k_means=use_k_means
        )
    
    elif active_tab == 2 or active_tab == "grayscale":  # Grayscale tab
        dither_grayscale = DITHER_METHODS[dither_method_grayscale]
        quantize_grayscale = QUANTIZATION_METHODS[quantization_method_grayscale]
        new_image = convert_to_grayscale(new_image)
        new_image = limit_colors(
            image=new_image,
            limit=int(number_of_shades),
            quantize=quantize_grayscale,
            dither=dither_grayscale,
            use_k_means=use_k_means_grayscale
        )
    
    elif active_tab == 3 or active_tab == "black_and_white":  # Black and white tab
        new_image = convert_to_black_and_white(new_image, black_and_white_threshold, is_inversed_black_and_white)
    
    elif active_tab == 4 or active_tab == "custom_palette":  # Custom palette tab
        dither_palette = DITHER_METHODS[dither_method_palette]
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

    if need_rescale:
        new_image = resize_image(new_image, (original_width, original_height))

    return new_image.convert('RGBA')
