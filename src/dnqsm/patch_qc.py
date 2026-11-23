"""Render comparable center slices of the generated susceptibility and fields."""

from functools import lru_cache

import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont

PANELS = (
    "Susceptibility",
    "Brain mask",
    "Exterior susceptibility",
    "Local field",
    "Background field",
    "Total field",
)
AXES = ("D", "H", "W")
TILE = 160
DISPLAY_LIMIT = 0.1


@lru_cache(maxsize=1)
def color_table():
    return colormaps["gray"](np.linspace(0, 1, 256), bytes=True)[:, 0]


def slice_image(values):
    indices = np.rint(np.clip((values / DISPLAY_LIMIT + 1) * 127.5, 0, 255)).astype(
        np.uint8
    )
    return Image.fromarray(color_table()[indices]).resize(
        (TILE, TILE), Image.Resampling.NEAREST
    )


def render_center_slices(path, title, volumes):
    image = Image.new("L", (1240, 618), 255)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=14)
    heading = ImageFont.load_default(size=17)
    ink = 28
    draw.text((16, 12), title, fill=ink, font=heading)
    for column, (label, volume) in enumerate(zip(PANELS, volumes, strict=True)):
        x = 66 + 194 * column
        draw.text((x, 46), label, fill=ink, font=font)
        for axis, axis_name in enumerate(AXES):
            y = 69 + axis * 171
            index = volume.shape[axis] // 2
            if column == 0:
                draw.text((14, y + 68), f"{axis_name}={index}", fill=ink, font=font)
            values = np.take(volume, index, axis=axis)
            image.paste(slice_image(values), (x, y))
        bar = Image.fromarray(color_table()[None, :])
        image.paste(bar.resize((TILE, 9)), (x, 595))
        units = "" if label == "Brain mask" else " ppm"
        low, high = f"-{DISPLAY_LIMIT:g}", f"+{DISPLAY_LIMIT:g}{units}"
        draw.text((x, 575), low, fill=ink, font=font)
        draw.text((x + TILE, 575), high, anchor="ra", fill=ink, font=font)
    image.save(path, compress_level=3)
