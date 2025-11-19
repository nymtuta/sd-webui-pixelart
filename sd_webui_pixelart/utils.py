from PIL import Image

# https://pillow.readthedocs.io/en/stable/reference/Image.html#dither-modes
DITHER_METHODS = {
    "None": Image.Dither.NONE,
    "Floyd-Steinberg": Image.Dither.FLOYDSTEINBERG
}

#https://pillow.readthedocs.io/en/stable/reference/Image.html#quantization-methods
QUANTIZATION_METHODS = {
    "Median cut": Image.Quantize.MEDIANCUT,
    "Maximum coverage": Image.Quantize.MAXCOVERAGE,
    "Fast octree": Image.Quantize.FASTOCTREE,
    "libimagequant": Image.Quantize.LIBIMAGEQUANT
}


def parse_color_text(text: str) -> list:
    """
    Parse a text string of colors and return a list of RGB tuples.
    Supports formats like:
    - Hex: #FF0000, #F00, FF0000, F00
    - RGB: (255,0,0), 255,0,0, 255 0 0
    
    Multiple colors can be separated by commas or newlines.
    Returns a list of (R, G, B) tuples, or None if parsing fails.
    """
    if not text or not text.strip():
        return None
    
    colors = []
    entries = [e.strip() for e in text.replace('\n', ',').split(',') if e.strip()]
    
    for entry in entries:
        try:
            # Try hex format (#RRGGBB or #RGB or RRGGBB or RGB)
            if entry.startswith('#'):
                hex_str = entry[1:]
            else:
                hex_str = entry
            
            # Only treat as hex if it looks like a valid hex color (not mixed alphanumeric gibberish)
            if len(hex_str) == 6 and all(c in '0123456789ABCDEFabcdef' for c in hex_str):
                r = int(hex_str[0:2], 16)
                g = int(hex_str[2:4], 16)
                b = int(hex_str[4:6], 16)
                colors.append((r, g, b))
            elif len(hex_str) == 3 and all(c in '0123456789ABCDEFabcdef' for c in hex_str):
                r = int(hex_str[0] * 2, 16)
                g = int(hex_str[1] * 2, 16)
                b = int(hex_str[2] * 2, 16)
                colors.append((r, g, b))
            else:
                # Try RGB format (r,g,b) or r g b
                parts = [p.strip() for p in entry.replace('(', '').replace(')', '').split(None if ' ' in entry else ',')]
                if len(parts) == 3:
                    r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
                    if 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255:
                        colors.append((r, g, b))
        except (ValueError, IndexError):
            pass
    
    return colors if colors else None


def read_palette_file(file_path: str) -> tuple:
    """
    Read a palette file and return (colors_list, error_message).
    Accepts any file type and validates the content rigorously.
    
    Returns:
        (list of (R,G,B) tuples, None) if successful
        (None, error_message_str) if validation fails
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        return None, "File is not a valid text file (appears to be binary)"
    except Exception as e:
        return None, f"Could not read file: {str(e)}"
    
    if not content.strip():
        return None, "File is empty"
    
    # Count valid color entries to ensure meaningful content
    colors = parse_color_text(content)
    if not colors:
        return None, "No valid colors found in file. Use formats like: #FF0000, 255,0,0, or (255,0,0)"
    
    # Additional validation: check that the file contains mostly color-like content
    # Count lines and color entries to ensure file is plausibly a color palette
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    non_empty_lines = len(lines)
    
    # If very few colors compared to lines, might be wrong file format
    # But be lenient: at least 50% of lines should parse as colors OR have at least 3 colors
    if non_empty_lines > 5 and len(colors) < non_empty_lines * 0.3:
        return None, f"File appears to contain mostly non-color data ({len(colors)} colors found in {non_empty_lines} lines)"
    
    return colors, None


def create_palette_from_colors(rgb_colors: list, size: int = None) -> Image.Image:
    """
    Create a PIL Image palette from a list of RGB tuples.
    Pads with black if fewer colors than size.
    
    Args:
        rgb_colors: List of (R, G, B) tuples
        size: Total palette size (default: auto-deduce from rgb_colors, max 256)
    
    Returns:
        A palette Image (mode 'P')
    """
    if not rgb_colors:
        return None
    
    # Auto-deduce size: use actual color count, capped at 256
    max_colors = 256
    if size is None:
        size = min(len(rgb_colors), max_colors)
    else:
        size = min(size, max_colors)

    # Create palette data: flatten RGB tuples for up to `size` entries
    palette_data = []
    for color in rgb_colors[:size]:
        palette_data.extend(color)

    # Pad with black (0, 0, 0) to reach full 256 * 3 entries required by putpalette
    while len(palette_data) < max_colors * 3:
        palette_data.extend([0, 0, 0])

    # Create a palette (mode 'P') image and apply the palette data
    palette_img = Image.new('P', (1, 1))
    palette_img.putpalette(palette_data)
    return palette_img


def downscale_image(image: Image, scale: int) -> Image:
    width, height = image.size
    downscaled_image = image.resize((int(width / scale), int(height / scale)), Image.NEAREST)
    return downscaled_image


def resize_image(image: Image, size) -> Image:
    width, height = size
    resized_image = image.resize((width, height), Image.NEAREST)
    return resized_image


def limit_colors(
        image,
        limit: int=16,
        palette=None,
        palette_colors: int=256,
        quantize: Image.Quantize=Image.Quantize.MEDIANCUT,
        dither: Image.Dither=Image.Dither.NONE,
        use_k_means: bool=False
    ):
    # Ensure image is in a mode that supports quantize (RGB or L)
    if image.mode not in ('RGB', 'L'):
        if image.mode == 'RGBA':
            # Convert RGBA to RGB by compositing on white background
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3] if len(image.split()) > 3 else None)
            image = background
        else:
            # For other modes (P, 1, etc.), convert to RGB
            image = image.convert('RGB')
    
    if use_k_means:
        k_means_value = limit
    else:
        k_means_value = 0

    if palette:
        # Accept palette as either a PIL Image (mode 'P' preferred) or a list of (R,G,B) tuples
        if isinstance(palette, Image.Image):
            # If palette is already a 'P' image created via putpalette, use it directly
            if palette.mode == 'P':
                color_palette = palette
            else:
                # If it's an RGB image, extract unique colors and build a proper 'P' palette
                cols = []
                try:
                    cols = [c[1] for c in palette.getcolors()] if palette.getcolors() else []
                except Exception:
                    cols = []

                if not cols:
                    # Fallback: sample the image's pixels up to palette_colors
                    cols = list({palette.getpixel((x % palette.width, x // palette.width)) for x in range(min(palette.width * palette.height, palette_colors))})

                # Ensure colors are RGB tuples
                rgb_list = []
                for c in cols:
                    if isinstance(c, int):
                        # getpixel may return an index for 'P' images; convert via convert
                        c = palette.convert('RGB').getpixel((0, 0))
                    if isinstance(c, tuple) and len(c) >= 3:
                        rgb_list.append((c[0], c[1], c[2]))

                if rgb_list:
                    color_palette = create_palette_from_colors(rgb_list, size=min(len(rgb_list), palette_colors))
                else:
                    # As a last resort, quantize the palette image itself to build a palette
                    color_palette = palette.quantize(colors=palette_colors)
        else:
            # Treat palette as an iterable of RGB tuples
            try:
                rgb_list = list(palette)
                color_palette = create_palette_from_colors(rgb_list, size=min(len(rgb_list), palette_colors))
            except Exception:
                # Fallback to quantizing the image to requested number of colors
                color_palette = image.quantize(colors=limit, kmeans=0)
    else:
        # we need to get palette from image, because
        # dither in quantize doesn't work without it
        # https://pillow.readthedocs.io/en/stable/_modules/PIL/Image.html#Image.quantize
        color_palette = image.quantize(colors=limit, kmeans=k_means_value, method=quantize, dither=Image.Dither.NONE)

    # Ensure image is in correct mode for quantize with palette
    if image.mode not in ('RGB', 'L'):
        image = image.convert('RGB')
    
    new_image = image.quantize(palette=color_palette, dither=dither)

    return new_image


def convert_to_grayscale(image):
    new_image = image.convert("L")
    return new_image.convert("RGB")


def convert_to_black_and_white(image: Image, threshold: int=128, is_inversed: bool=False):
    if is_inversed:
        apply_threshold = lambda x : 255 if x < threshold else 0
    else:
        apply_threshold = lambda x : 255 if x > threshold else 0

    black_and_white_image = image.convert('L', dither=Image.Dither.NONE).point(apply_threshold, mode='1')
    return black_and_white_image.convert("RGB")
