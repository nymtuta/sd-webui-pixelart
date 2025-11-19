"""Shared UI components and processing logic for pixelart scripts."""

import inspect
import math
import gradio as gr
from PIL import features, Image

from modules import ui_components

from sd_webui_pixelart.utils import (
    DITHER_METHODS, QUANTIZATION_METHODS, downscale_image, limit_colors, 
    resize_image, convert_to_grayscale, convert_to_black_and_white, 
    parse_color_text, create_palette_from_colors, read_palette_file
)

# Get available resampling methods from PIL
RESAMPLE_METHODS = [name for name in dir(Image.Resampling) if not name.startswith('_')]

# Track if global CSS has been injected
_css_injected = False

def _inject_global_css():
    """Inject global CSS for palette UI styling. Only injects once."""
    global _css_injected
    if not _css_injected:
        _css_injected = True
        return gr.HTML(
            "<style>#palette_buttons_group{--layout-gap:16px;--button-small-radius:8px;}#threshold_group{background:var(--button-secondary-background-fill);gap:0;border-radius:var(--button-small-radius);flex-wrap:nowrap;}.fit_content{min-width:fit-content !important;width:auto !important;flex-grow:0 !important}.small_border{border:1px solid #fff3 !important;}</style>",
            visible=False
        )
    return None

def _get_palette_accordion_label(color_count):
    """Generate the accordion label with color count. Single source of truth for the label format."""
    if color_count > 256:
        return f'<span title="Only the first 256 colors will be used for generation">Current Palette (<strong style="color: red;">{color_count}/256</strong>)</span>'
    return f'<span>Current Palette ({color_count})</span>'

def _colors_to_hex_text(colors):
    return ", ".join([f"#{r:02X}{g:02X}{b:02X}" for r, g, b in colors])

def _create_color_chips_html(colors):
    """Create HTML for displaying color chips with hex codes."""
    if not colors:
        return ""
    
    chips = []
    for r, g, b in colors:
        hex_code = f"#{r:02X}{g:02X}{b:02X}"
        # Use white text for dark colors, black text for light colors
        text_color = "#FFFFFF" if (r * 0.299 + g * 0.587 + b * 0.114) < 128 else "#000000"
        # Lower opacity for black
        opacity = "0.7" if (r, g, b) == (0, 0, 0) else "1.0"
        
        chip_html = f'''
        <div class="color-chip" 
            style="display: inline-block; position: relative; padding: 8px 12px; background-color: {hex_code}; border: 2px solid {hex_code}99; border-radius: 4px; user-select: none;">
            <span style="color: {text_color}; font-size: 12px; font-family: monospace; font-weight: bold; opacity: {opacity};">{hex_code}</span>
        </div>
        '''
        chips.append(chip_html)
    
    return f'''
    <div style="display: flex; flex-wrap: wrap; gap: 8px; padding: 8px; border-radius: 4px; min-height: 50px; max-height: 400px; overflow-y: auto;">{"".join(chips)}</div>
    '''

def _ensure_black_in_palette(colors):
    """Ensure black (0,0,0) is always present in the palette."""
    black = (0, 0, 0)
    if black not in colors:
        colors.insert(0, black)
    return colors

def _update_palette_display(palette_text, palette_input_text):
    """Update palette display from text input or file upload."""
    # Parse colors from the stored palette_text
    colors = parse_color_text(palette_text) if palette_text else []
    
    # If there's new input text, parse and merge it
    if palette_input_text and palette_input_text.strip():
        new_colors = parse_color_text(palette_input_text)
        if new_colors:
            colors = _merge_and_dedup_colors(palette_text, new_colors)
    
    # Ensure black is always present
    colors = _ensure_black_in_palette(colors)
    
    # Generate HTML display and update stored text
    html = _create_color_chips_html(colors)
    text = _colors_to_hex_text(colors)
    accordion_label = _get_palette_accordion_label(len(colors))
    
    return text, html, "", accordion_label  # Return updated palette_text, HTML display, clear input, and accordion label

def _clear_palette():
    """Clear palette, keeping only black."""
    black = (0, 0, 0)
    text = _colors_to_hex_text([black])
    html = _create_color_chips_html([black])
    accordion_label = _get_palette_accordion_label(1)
    return text, html, accordion_label

def _deduplicate_similar_colors(palette_text, threshold=30):
    """Remove similar colors, keeping the best representative from each group.
    
    Args:
        palette_text: Comma-separated hex colors
        threshold: Maximum color distance to consider colors similar (0-441, default 30)
    
    Returns:
        Updated palette_text and HTML display
    """
    colors = parse_color_text(palette_text) if palette_text else []
    if len(colors) <= 1:
        return palette_text, _create_color_chips_html(colors), _get_palette_accordion_label(len(colors))
    
    def color_distance(c1, c2):
        """Calculate Euclidean distance between two RGB colors."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))
    
    # Group similar colors
    groups = []
    used = set()
    
    for i, color in enumerate(colors):
        if i in used:
            continue
        group = [color]
        used.add(i)
        
        for j, other_color in enumerate(colors):
            if j in used:
                continue
            if color_distance(color, other_color) <= threshold:
                group.append(other_color)
                used.add(j)
        
        groups.append(group)
    
    # For each group, pick the "best" color (average)
    result = []
    for group in groups:
        if len(group) == 1:
            result.append(group[0])
        else:
            # Use average of the group
            avg_r = sum(c[0] for c in group) // len(group)
            avg_g = sum(c[1] for c in group) // len(group)
            avg_b = sum(c[2] for c in group) // len(group)
            result.append((avg_r, avg_g, avg_b))
    
    # Ensure black is present
    result = _ensure_black_in_palette(result)
    
    html = _create_color_chips_html(result)
    text = _colors_to_hex_text(result)
    return text, html, _get_palette_accordion_label(len(result))

def _merge_and_dedup_colors(existing_text, new_colors):
    """Merge existing text colors with new_colors (list of RGB tuples), remove duplicates preserving order."""
    existing = parse_color_text(existing_text) or []
    combined = existing + (new_colors or [])
    seen = set()
    result = []
    for c in combined:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result

def on_palette_upload(file_path, existing_text=""):
    """
    Unified callback for palette file upload. Handles both image and text files.
    Detects file type and extracts colors accordingly.
    """
    if not file_path:
        return existing_text, gr.update(value=None), gr.update(value="", visible=False), gr.update(), gr.update()
    
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
        merged = _ensure_black_in_palette(merged)
        
        return _colors_to_hex_text(merged), gr.update(value=None), gr.update(value="", visible=False), _create_color_chips_html(merged), _get_palette_accordion_label(len(merged))
    except Exception as img_error:
        # Not an image, try as text file
        colors, error = read_palette_file(file_path)
        if error:
            # Neither valid image nor valid text file
            return existing_text, gr.update(), gr.update(value=f"<div style='color:red'>File is neither a valid image nor a valid text palette file.<br>Image error: {img_error}<br>Text error: {error}</div>", visible=True), gr.update(), gr.update()
        
        # Successfully parsed as text file
        merged = _merge_and_dedup_colors(existing_text, colors)
        merged = _ensure_black_in_palette(merged)
        
        color_text = _colors_to_hex_text(merged)
        html = _create_color_chips_html(merged)
        return color_text, gr.update(value=None), gr.update(value="", visible=False), html, _get_palette_accordion_label(len(merged))

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
            components['downscale_method'] = gr.Dropdown(
                choices=[(name.title(), name) for name in RESAMPLE_METHODS],
                value="NEAREST",
                label="Downscale method"
            )
        with gr.Row():
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
                # Hidden storage for the actual palette text (passed to process_image)
                components['palette_text'] = gr.Textbox(visible=False, value="#000000")
                
                # Visual color palette display with dynamic title
                palette_title = gr.HTML(f"{_get_palette_accordion_label(1)}")
                
                with gr.Accordion("Palette", open=True):
                    components['palette_display'] = gr.HTML(
                        value=_create_color_chips_html([(0, 0, 0)]),
                        elem_id="palette_display"
                    )
                    
                    # Palette action buttons (under the display)
                    with gr.Row(elem_id="palette_buttons_group"):
                        _inject_global_css()  # Inject CSS once
                        clear_palette_btn = gr.Button("Clear", size="sm", scale=0, min_width=60)
                        copy_hex_btn = gr.Button("Copy Hex", size="sm", scale=0, min_width=80)
                        with gr.Row(elem_id="threshold_group", elem_classes=["fit_content"]):
                            dedupe_palette_btn = gr.Button("Remove Similar", size="sm", scale=0, min_width=120, elem_classes=["small_border"])
                            gr.HTML(
                                "<span style='line-height:40px;padding:0 8px;font-size:14px;'>Threshold:</span>",
                                elem_classes=["fit_content"]
                            )
                            dedupe_threshold = gr.Number(
                                value=10,
                                minimum=0,
                                maximum=441,
                                step=1,
                                show_label=False,
                                container=False,
                                scale=0,
                                min_width=60
                            )
                
                # Input field for adding colors
                palette_input = gr.Textbox(
                    placeholder="Add colors: #FF0000, #00FF00, #0000FF or RGB: 255,0,0",
                    lines=2,
                    interactive=True,
                    label="Add colors"
                )
                
                # Error message display
                components['palette_message'] = gr.HTML(value="", visible=False)
                
                # File upload
                with gr.Row():
                    components['palette_upload'] = gr.File(
                        label="Or upload image/palette file",
                        file_count="single",
                        type="filepath",
                        scale=1
                    )
                
                # Connect input changes to update display (blur event)
                palette_input.blur(
                    fn=_update_palette_display,
                    inputs=[components['palette_text'], palette_input],
                    outputs=[components['palette_text'], components['palette_display'], palette_input, palette_title]
                )
                
                # Connect clear button
                clear_palette_btn.click(
                    fn=_clear_palette,
                    inputs=[],
                    outputs=[components['palette_text'], components['palette_display'], palette_title]
                )
                
                # Connect copy hex button to copy to clipboard
                copy_hex_btn.click(
                    fn=lambda x: x,
                    inputs=[components['palette_text']],
                    outputs=[],
                    _js="(palette_text) => { navigator.clipboard.writeText(palette_text); return palette_text; }"
                )
                
                # Connect deduplicate button
                dedupe_palette_btn.click(
                    fn=_deduplicate_similar_colors,
                    inputs=[components['palette_text'], dedupe_threshold],
                    outputs=[components['palette_text'], components['palette_display'], palette_title]
                )
                
                # Connect file upload to automatically append to palette and clear file input
                components['palette_upload'].change(
                    fn=on_palette_upload,
                    inputs=[components['palette_upload'], components['palette_text']],
                    outputs=[components['palette_text'], components['palette_upload'], components['palette_message'], components['palette_display'], palette_title]
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
    downscale_method,
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

    new_image = downscale_image(new_image, downscale, downscale_method)

    # Apply processing based on active tab
    # active_tab can be tab index (int) or tab ID (string)
    # Tabs: None (0/none), From current image (1/color), Grayscale (2/grayscale), 
    #       Black and white (3/black_and_white), Custom (4/custom_palette)
    
    if active_tab == 1 or active_tab == "color":  # From current image tab
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
