import colorsys
import copy
import xml.etree.ElementTree as ET
from collections.abc import Callable

import numpy as np

from veb.svg_data import SVG_NS

SEVERITIES = (1, 2, 3, 4, 5)
SEVERITY = {
    "remove": {"fraction": (0.1, 0.2, 0.3, 0.45, 0.6)},
    "shift": {"subset": 0.5, "offset_fraction": (0.03, 0.06, 0.11, 0.17, 0.25)},
    "rotate": {"subset": 0.5, "angle_deg": (5, 10, 20, 40, 80)},
    "recolor": {"subset": 0.5, "hue_deg": (15, 30, 60, 120, 180), "lightness": (0.03, 0.06, 0.1, 0.15, 0.2)},
    "duplicate": {"fraction": (0.1, 0.2, 0.3, 0.45, 0.6), "offset_fraction": 0.08},
    "reorder": {"spread": (0.2, 0.4, 0.8, 1.6, None)},
}

Corruption = Callable[[ET.Element, int, np.random.Generator], ET.Element]


def _level(name: str, key: str, severity: int):
    return SEVERITY[name][key][severity - 1]


def _viewbox(root: ET.Element) -> tuple[float, float, float, float]:
    x, y, w, h = (float(v) for v in root.get("viewBox").replace(",", " ").split())
    return x, y, w, h


def _count(n: int, fraction: float) -> int:
    return max(1, round(fraction * n))


def _with_children(root: ET.Element, children: list[ET.Element]) -> ET.Element:
    out = ET.Element(root.tag, dict(root.attrib))
    out.text, out.tail = root.text, root.tail
    out.extend(children)
    return out


def _prepend_transform(element: ET.Element, transform: str) -> None:
    element.set("transform", f"{transform} {element.get('transform', '')}".strip())


def _translated(element: ET.Element, distance: float, direction: float) -> ET.Element:
    out = copy.deepcopy(element)
    _prepend_transform(out, f"translate({distance * np.cos(direction):.3f} {distance * np.sin(direction):.3f})")
    return out


def _parse_hex(colour: str) -> tuple[float, float, float]:
    digits = colour.lstrip("#")
    if len(digits) == 3:
        digits = "".join(c * 2 for c in digits)
    return tuple(int(digits[i : i + 2], 16) / 255 for i in (0, 2, 4))


def _shift_colour(colour: str, hue_deg: float, lightness: float) -> str:
    h, l, s = colorsys.rgb_to_hls(*_parse_hex(colour))
    l = l + lightness if l < 0.5 else l - lightness
    rgb = colorsys.hls_to_rgb((h + hue_deg / 360) % 1, l, s)
    return "#" + "".join(f"{round(c * 255):02X}" for c in rgb)


def remove(root: ET.Element, severity: int, rng: np.random.Generator) -> ET.Element:
    children = list(root)
    order = rng.permutation(len(children))
    k = max(1, min(len(children) - 1, _count(len(children), _level("remove", "fraction", severity))))
    dropped = set(order[:k].tolist())
    return _with_children(root, [copy.deepcopy(c) for i, c in enumerate(children) if i not in dropped])


def shift(root: ET.Element, severity: int, rng: np.random.Generator) -> ET.Element:
    children = [copy.deepcopy(c) for c in root]
    order = rng.permutation(len(children))
    directions = rng.uniform(0, 2 * np.pi, size=len(children))
    distance = _level("shift", "offset_fraction", severity) * _viewbox(root)[2]
    for i in order[: _count(len(children), SEVERITY["shift"]["subset"])]:
        children[i] = _translated(children[i], distance, directions[i])
    return _with_children(root, children)


def rotate(root: ET.Element, severity: int, rng: np.random.Generator) -> ET.Element:
    children = [copy.deepcopy(c) for c in root]
    order = rng.permutation(len(children))
    signs = rng.choice((-1, 1), size=len(children))
    x, y, w, h = _viewbox(root)
    angle = _level("rotate", "angle_deg", severity)
    for i in order[: _count(len(children), SEVERITY["rotate"]["subset"])]:
        _prepend_transform(children[i], f"rotate({signs[i] * angle} {x + w / 2:g} {y + h / 2:g})")
    return _with_children(root, children)


def recolor(root: ET.Element, severity: int, rng: np.random.Generator) -> ET.Element:
    children = [copy.deepcopy(c) for c in root]
    order = rng.permutation(len(children))
    signs = rng.choice((-1, 1), size=len(children))
    hue, lightness = _level("recolor", "hue_deg", severity), _level("recolor", "lightness", severity)
    for i in order[: _count(len(children), SEVERITY["recolor"]["subset"])]:
        children[i].set("fill", _shift_colour(children[i].get("fill", "#000000"), signs[i] * hue, lightness))
    return _with_children(root, children)


def duplicate(root: ET.Element, severity: int, rng: np.random.Generator) -> ET.Element:
    children = list(root)
    order = rng.permutation(len(children))
    directions = rng.uniform(0, 2 * np.pi, size=len(children))
    copied = set(order[: _count(len(children), _level("duplicate", "fraction", severity))].tolist())
    distance = SEVERITY["duplicate"]["offset_fraction"] * _viewbox(root)[2]
    out = []
    for i, child in enumerate(children):
        out.append(copy.deepcopy(child))
        if i in copied:
            out.append(_translated(child, distance, directions[i]))
    return _with_children(root, out)


def reorder(root: ET.Element, severity: int, rng: np.random.Generator) -> ET.Element:
    children = [copy.deepcopy(c) for c in root]
    n = len(children)
    noise = rng.uniform(-1, 1, size=n)
    fallback = int(rng.integers(max(1, n - 1)))
    spread = _level("reorder", "spread", severity)
    keys = noise if spread is None else np.arange(n) + spread * n * noise
    order = np.argsort(keys, kind="stable").tolist()
    if order == list(range(n)) and n > 1:
        order[fallback], order[fallback + 1] = order[fallback + 1], order[fallback]
    return _with_children(root, [children[i] for i in order])


def redundant(root: ET.Element, severity: int = 1, rng: np.random.Generator | None = None) -> ET.Element:
    wrapped = []
    for child in root:
        outer = ET.Element(f"{{{SVG_NS}}}g")
        ET.SubElement(outer, f"{{{SVG_NS}}}g").append(copy.deepcopy(child))
        wrapped.append(outer)
    unused = ET.Element(f"{{{SVG_NS}}}defs")
    unused.extend(copy.deepcopy(child) for child in root)
    return _with_children(root, [*wrapped, unused])


CORRUPTIONS: dict[str, Corruption] = {
    "remove": remove,
    "shift": shift,
    "rotate": rotate,
    "recolor": recolor,
    "duplicate": duplicate,
    "reorder": reorder,
}
CONTROL = "redundant"
