import random
import colorsys
import numpy as np
from typing import List

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def get_nuisance_colors(seed: int, num_colors: int = 4) -> List[str]:
    rng = random.Random(seed)
    palette = [
        "#E63946", "#457B9D", "#2A9D8F", "#E76F51",
        "#F4A261", "#9B5DE5", "#F15BB5", "#00BBF9",
        "#00F5D4", "#FF006E", "#8338EC", "#3A86FF",
        "#D4A373", "#CCD5AE", "#E9EDC9", "#FAEDCD"
    ]

    if num_colors <= len(palette):
        return rng.sample(palette, num_colors)

    # For larger num_colors (e.g., L=16+), dynamically generate distinct HSL colors
    generated = list(palette)
    needed = num_colors - len(palette)
    for i in range(needed):
        hue = (i / needed + rng.uniform(-0.02, 0.02)) % 1.0
        r, g, b = colorsys.hls_to_rgb(hue, 0.5, 0.8)
        hex_col = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
        generated.append(hex_col)

    rng.shuffle(generated)
    return generated[:num_colors]
