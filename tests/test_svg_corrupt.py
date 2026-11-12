import xml.etree.ElementTree as ET

import numpy as np
import pytest

from veb.svg_corrupt import CORRUPTIONS, SEVERITIES, SEVERITY, redundant
from veb.svg_data import drawables, is_usable, parse, render, serialise

SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 36">'
    '<circle fill="#FFCC4D" cx="18" cy="18" r="17"/>'
    '<rect fill="#3B88C3" x="4" y="5" width="12" height="9"/>'
    '<ellipse fill="#DD2E44" cx="24" cy="12" rx="5" ry="3"/>'
    '<path fill="#664500" d="M8 22c3 5 17 5 20 0l-2 8H10z"/>'
    '<polygon fill="#77B255" points="18,3 22,10 14,10"/>'
    '<circle fill="#FFF" cx="12" cy="26" r="2.5"/>'
    '<path fill="#292F33" d="M20 20h9v4h-9z"/>'
    '<rect fill="#9266CC" x="15" y="14" width="7" height="6"/>'
    '<circle fill="#F4900C" cx="28" cy="28" r="4"/>'
    '<path fill="#55ACEE" d="M2 30l6-4 3 6z"/>'
    "</svg>"
)
N = 10


def apply(name: str, severity: int, seed: int = 0) -> ET.Element:
    return CORRUPTIONS[name](parse(SVG), severity, np.random.default_rng(seed))


def fills(root: ET.Element) -> list[str]:
    return [child.get("fill") for child in root]


def geometry(root: ET.Element) -> list[tuple]:
    return [tuple(sorted((k, v) for k, v in child.attrib.items() if k != "transform")) for child in root]


def test_fixture_is_usable():
    assert is_usable(parse(SVG))


@pytest.mark.parametrize("name", list(CORRUPTIONS))
@pytest.mark.parametrize("severity", SEVERITIES)
def test_output_is_valid_svg_that_renders_differently(name, severity):
    text = serialise(apply(name, severity))
    reparsed = ET.fromstring(text)
    assert reparsed.get("viewBox") == "0 0 36 36"
    assert text.startswith('<svg xmlns="http://www.w3.org/2000/svg"')
    image = render(text, size=64)
    assert image.shape == (64, 64, 3) and image.dtype == np.uint8
    if name != "reorder":
        assert np.any(image != render(SVG, size=64))


@pytest.mark.parametrize("name", list(CORRUPTIONS))
def test_deterministic_given_seed_and_input_untouched(name):
    root = parse(SVG)
    before = serialise(root)
    first = serialise(CORRUPTIONS[name](root, 3, np.random.default_rng(7)))
    second = serialise(CORRUPTIONS[name](root, 3, np.random.default_rng(7)))
    assert first == second
    assert serialise(root) == before
    assert first != before


@pytest.mark.parametrize("severity", SEVERITIES)
def test_remove_drops_promised_fraction(severity):
    expected = max(1, round(SEVERITY["remove"]["fraction"][severity - 1] * N))
    assert len(drawables(apply("remove", severity))) == N - expected


def test_remove_keeps_one_element_and_drops_one():
    two = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 36"><rect width="3" height="3"/><circle r="2"/></svg>'
    for severity in SEVERITIES:
        assert len(CORRUPTIONS["remove"](parse(two), severity, np.random.default_rng(0))) == 1


def test_remove_is_nested_across_severities():
    kept = [set(geometry(apply("remove", severity))) for severity in SEVERITIES]
    assert all(later <= earlier for earlier, later in zip(kept, kept[1:]))


@pytest.mark.parametrize("severity", SEVERITIES)
def test_duplicate_adds_promised_copies(severity):
    expected = max(1, round(SEVERITY["duplicate"]["fraction"][severity - 1] * N))
    out = apply("duplicate", severity)
    assert len(drawables(out)) == N + expected
    assert sum("translate" in child.get("transform", "") for child in out) == expected


@pytest.mark.parametrize("name", ["shift", "rotate", "recolor", "reorder"])
@pytest.mark.parametrize("severity", SEVERITIES)
def test_count_preserving_corruptions(name, severity):
    assert len(drawables(apply(name, severity))) == N


@pytest.mark.parametrize("name,keyword", [("shift", "translate"), ("rotate", "rotate")])
def test_transform_applied_to_half_the_elements(name, keyword):
    out = apply(name, 2)
    assert sum(keyword in child.get("transform", "") for child in out) == round(SEVERITY[name]["subset"] * N)
    assert geometry(out) == geometry(parse(SVG))


def test_rotate_uses_canvas_centre():
    transforms = [child.get("transform") for child in apply("rotate", 4) if child.get("transform")]
    assert all(t in ("rotate(40 18 18)", "rotate(-40 18 18)") for t in transforms)


def test_recolor_changes_only_fills_of_half_the_elements():
    out = apply("recolor", 3)
    changed = sum(a != b for a, b in zip(fills(out), fills(parse(SVG))))
    assert changed == round(SEVERITY["recolor"]["subset"] * N)
    assert all(len(f) == 7 and f.startswith("#") for f in fills(out) if f != "#FFF")
    strip = lambda root: [tuple(sorted((k, v) for k, v in c.attrib.items() if k != "fill")) for c in root]
    assert strip(out) == strip(parse(SVG))


@pytest.mark.parametrize("severity", SEVERITIES)
def test_reorder_is_a_non_identity_permutation(severity):
    out = apply("reorder", severity)
    assert sorted(geometry(out)) == sorted(geometry(parse(SVG)))
    assert geometry(out) != geometry(parse(SVG))


def test_reorder_displacement_grows():
    original = geometry(parse(SVG))

    def displacement(severity: int) -> float:
        totals = []
        for seed in range(30):
            out = geometry(apply("reorder", severity, seed))
            totals.append(np.mean([abs(i - original.index(g)) for i, g in enumerate(out)]))
        return float(np.mean(totals))

    values = [displacement(severity) for severity in SEVERITIES]
    assert values == sorted(values) and values[0] < values[-1]


def test_redundant_doubles_code_and_keeps_pixels_identical():
    text = serialise(redundant(parse(SVG)))
    leaves = [e for e in ET.fromstring(text).iter() if len(e) == 0]
    assert len(leaves) == 2 * N
    assert 1.8 < len(text) / len(SVG) < 2.6
    original = [c.attrib for c in parse(SVG)]
    assert [leaf.attrib for leaf in leaves[:N]] == [leaf.attrib for leaf in leaves[N:]] == original
    assert np.array_equal(render(text), render(SVG))
