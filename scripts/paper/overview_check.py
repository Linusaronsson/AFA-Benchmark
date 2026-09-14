"""
Render and check the Figure 1 prototypes.

Trims the page down to the drawing, then produces the two views a printed
figure has to survive: greyscale, and a colour-vision simulation. Colour is
never the only channel in these prototypes, so both views should stay readable.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

BUILD = Path("extra/output/paper/figs/overview/build")

# Projections onto the confusion line of each dichromacy, applied in linear
# RGB. Enough to answer "does this survive", not a clinical simulation.
CVD = {
    "deutan": np.array(
        [[0.625, 0.375, 0.0], [0.70, 0.30, 0.0], [0.0, 0.30, 0.70]]
    ),
    "protan": np.array(
        [[0.567, 0.433, 0.0], [0.558, 0.442, 0.0], [0.0, 0.242, 0.758]]
    ),
}


def _srgb_to_linear(a: np.ndarray) -> np.ndarray:
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(a: np.ndarray) -> np.ndarray:
    return np.where(a <= 0.0031308, a * 12.92, 1.055 * a ** (1 / 2.4) - 0.055)


def simulate(image: Image.Image, kind: str) -> Image.Image:
    arr = _srgb_to_linear(
        np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    )
    out = _linear_to_srgb(np.clip(arr @ CVD[kind].T, 0.0, 1.0))
    return Image.fromarray((out * 255).round().astype(np.uint8))


def trim(image: Image.Image, pad: int = 12) -> Image.Image:
    arr = np.asarray(image.convert("L"))
    ink = np.argwhere(arr < 245)
    if ink.size == 0:
        return image
    (top, left), (bottom, right) = ink.min(0), ink.max(0)
    return image.crop(
        (
            max(int(left) - pad, 0),
            max(int(top) - pad, 0),
            min(int(right) + pad + 1, image.width),
            min(int(bottom) + pad + 1, image.height),
        )
    )


def process(variant: str, dpi: int) -> None:
    pdf = BUILD / f"{variant}-figure.pdf"
    subprocess.run(
        [
            "pdftoppm",
            "-r",
            str(dpi),
            "-singlefile",
            "-png",
            str(pdf),
            str(BUILD / f"{variant}-raw"),
        ],
        check=True,
    )
    image = trim(Image.open(BUILD / f"{variant}-raw.png"))
    image.save(BUILD / f"{variant}.png")
    (BUILD / f"{variant}-raw.png").unlink()
    image.convert("L").save(BUILD / f"{variant}-grey.png")
    for kind in CVD:
        simulate(image, kind).save(BUILD / f"{variant}-{kind}.png")
    print(
        f"{variant}: {image.width}x{image.height}px, {image.width / dpi:.2f}in wide"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variants", nargs="+")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    for variant in args.variants:
        process(variant, args.dpi)


if __name__ == "__main__":
    main()
