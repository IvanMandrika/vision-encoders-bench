import io
import re
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import cairosvg
import numpy as np
from PIL import Image

from veb.common import DATA

SVG_NS = "http://www.w3.org/2000/svg"
TWEMOJI_REPO = "https://github.com/jdecked/twemoji"
TWEMOJI_TAG = "v17.0.3"
TWEMOJI_DIR = DATA / "twemoji"
DRAWABLE = ("path", "circle", "ellipse", "rect", "polygon")
UNSUPPORTED_ATTRS = ("opacity", "fill-opacity", "stroke", "clip-path", "style")
HEX_COLOUR = re.compile(r"#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})")
VIEWBOX = "0 0 36 36"

ET.register_namespace("", SVG_NS)


@dataclass(frozen=True)
class Icon:
    name: str
    svg: str


def local_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def drawables(root: ET.Element) -> list[ET.Element]:
    return [child for child in root if local_name(child) in DRAWABLE]


def parse(svg: str) -> ET.Element:
    return ET.fromstring(svg)


def serialise(root: ET.Element) -> str:
    return ET.tostring(root, encoding="unicode")


def is_usable(root: ET.Element, min_elements: int = 4, max_elements: int = 40) -> bool:
    children = list(root)
    if root.get("viewBox") != VIEWBOX or len(children) != len(drawables(root)):
        return False
    if not min_elements <= len(children) <= max_elements:
        return False
    for child in children:
        if len(child) or any(attr in child.attrib for attr in UNSUPPORTED_ATTRS):
            return False
        if not HEX_COLOUR.fullmatch(child.get("fill", "#000")):
            return False
    return True


def ensure_twemoji(target: Path = TWEMOJI_DIR) -> Path:
    svg_dir = target / "assets" / "svg"
    if not svg_dir.is_dir():
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", "--branch", TWEMOJI_TAG, "--filter=blob:none", "--sparse", TWEMOJI_REPO, str(target)], check=True)
        subprocess.run(["git", "-C", str(target), "sparse-checkout", "set", "assets/svg"], check=True)
    return svg_dir


def usable_icons(svg_dir: Path) -> list[Icon]:
    icons = []
    for path in sorted(svg_dir.glob("*.svg")):
        svg = path.read_text(encoding="utf-8")
        try:
            root = parse(svg)
        except ET.ParseError:
            continue
        if is_usable(root):
            icons.append(Icon(path.stem, serialise(root)))
    return icons


def load_icons(n: int = 400, seed: int = 0, svg_dir: Path | None = None) -> list[Icon]:
    pool = usable_icons(svg_dir or ensure_twemoji())
    if len(pool) < n:
        raise ValueError(f"only {len(pool)} usable icons, {n} requested")
    chosen = np.random.default_rng(seed).choice(len(pool), size=n, replace=False)
    return [pool[i] for i in sorted(chosen)]


def render(svg: str, size: int = 224, supersample: int = 1) -> np.ndarray:
    side = size * supersample
    png = cairosvg.svg2png(bytestring=svg.encode("utf-8"), output_width=side, output_height=side, background_color="white")
    image = Image.open(io.BytesIO(png)).convert("RGB")
    if supersample > 1:
        image = image.resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.uint8)
