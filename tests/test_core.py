"""Tests for each project milestone."""
import pytest

from app.core import milestone_1


def test_milestone_1(tmp_path):
    source = tmp_path / "raw.txt"
    output = tmp_path / "prepared" / "corpus.txt"

    source.write_text(
        "Gutenberg header\n"
        "Contents\n"
        "  CHAPTER I\n"
        "  Botany 1\n\n"
        "CHAPTER I\n\n"
        "Botany of the Tomato\n\n"
        "The tomato’s roots\n"
        "need space.\n\n"
        "[Illustration: A tomato plant.]\n"
        "[Pg 2]\n"
        "Keep paragraphs separate.\n\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
        "License footer",
        encoding="utf-8",
    )

    text = milestone_1(source, output)

    assert text.startswith("CHAPTER I\n\nBotany")
    assert "The tomato's roots need space." in text
    assert "\n\nKeep paragraphs separate." in text
    assert "Contents" not in text
    assert "Gutenberg header" not in text
    assert "License footer" not in text
    assert "[Illustration:" not in text
    assert "[Pg" not in text
    assert output.read_text(encoding="utf-8") == text


@pytest.mark.skip(reason="start milestone 2")
def test_milestone_2():
    raise NotImplementedError


@pytest.mark.skip(reason="start milestone 3")
def test_milestone_3():
    raise NotImplementedError


@pytest.mark.skip(reason="start milestone 4")
def test_milestone_4():
    raise NotImplementedError


@pytest.mark.skip(reason="start milestone 5")
def test_milestone_5():
    raise NotImplementedError


@pytest.mark.skip(reason="start milestone 6")
def test_milestone_6():
    raise NotImplementedError