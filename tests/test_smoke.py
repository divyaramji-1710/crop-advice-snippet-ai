"""Basic checks for the actual crop-text application."""
from app.cli import main


def test_hello_command(capsys):
    main(["hello"])
    assert "is ready" in capsys.readouterr().out


def test_generation_help(capsys):
    import pytest

    with pytest.raises(SystemExit) as error:
        main(["generate", "--help"])

    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "--checkpoint" in output
    assert "--temperature" in output


def test_tokenizer_round_trip():
    from app.tokenizer import BytePairTokenizer

    tokenizer = BytePairTokenizer().fit(
        "Tomato plants grow in soil. " * 5,
        num_merges=4,
    )
    text = "Tomato plants"
    assert tokenizer.decode(tokenizer.encode(text)) == text