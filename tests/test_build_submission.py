"""The submission build: what must never reach a reviewer, and what must survive intact."""

import re

import pytest

from admetood import build_submission as bs


def test_the_preprint_abstract_is_dropped():
    text = "Keep this.\n\n<details>\n<summary>x</summary>\nLong abstract.\n</details>\n\nAnd this."
    out = bs.strip_working_blocks(text)
    assert "Long abstract" not in out
    assert "Keep this." in out and "And this." in out


def test_the_revision_notes_are_dropped_with_everything_under_them():
    text = "## 8. Reproducibility\n\nBody.\n\n## Notes for revision\n\n- secret\n- more\n"
    out = bs.strip_working_blocks(text)
    assert "secret" not in out and "more" not in out
    assert "Body." in out


def test_underscores_in_paths_are_escaped_once_not_twice():
    """Escaping a code span in place and then escaping the whole string breaks the file."""
    out = bs.inline("see `results/interval_audit.json` for this")
    assert r"interval\_audit" in out
    assert r"\\_" not in out


def test_citations_become_cite_commands():
    assert r"\cite{ref4}" in bs.inline("an audit [4] found")


def test_escaped_pipes_do_not_invent_table_columns():
    """A cell reading "mean |gap| (pp)" splits into three cells if pipes are naive."""
    rows = [r"| Split | mean \|gap\| (pp) |", "| --- | ---: |", "| random | 12.6 |"]
    out = bs.convert_table(rows, "Caption", "tab:1")
    assert "{lr}" in out, "two columns, second right-aligned"
    # Two columns means exactly one separator per emitted row. A naive split on every
    # pipe would make the header three cells wide and LaTeX would abort on the mismatch.
    body = [line for line in out.split("\n") if line.endswith(r"\\")]
    assert len(body) == 2, "header and one data row"
    assert all(line.count("&") == 1 for line in body), body
    assert r"\textbar{}gap\textbar{}" in out, "the literal bars survive as text"


def test_alignment_comes_from_the_separator_row():
    rows = ["| a | b | c |", "| --- | ---: | --- |", "| 1 | 2 | 3 |"]
    assert "{lrl}" in bs.convert_table(rows, "c", "t")


def test_a_wide_table_is_allowed_to_shrink_but_a_narrow_one_is_not_forced_to():
    out = bs.convert_table(["| a |", "| --- |", "| 1 |"], "c", "t")
    assert r"\resizebox" in out and r"\ifdim\width>\linewidth" in out


def test_a_figure_paragraph_becomes_a_float_with_a_usable_caption():
    text = "**Figure 1** (`figures/f1.png`): real-split coverage by endpoint, one bar each."
    out = bs.paragraph_or_figure(text)
    assert r"\begin{figure}" in out
    assert "figures/f1.png" in out
    assert "Real-split coverage by endpoint" in out, "caption must open with a capital"
    assert "Figure 1" not in out, "the number comes from LaTeX, not the prose"


def test_a_figure_paragraph_keeps_prose_that_follows_the_caption_sentence():
    text = ("**Figure 5** (`figures/f5.png`): selective-prediction curves. A follow-up "
            "should set endpoint-specific targets.")
    out = bs.paragraph_or_figure(text)
    assert r"\begin{figure}" in out
    assert "A follow-up should set endpoint-specific targets." in out


def test_an_ordinary_paragraph_is_not_mistaken_for_a_figure():
    out = bs.paragraph_or_figure("Figure 1 is discussed below, with **emphasis**.")
    assert r"\begin{figure}" not in out
    assert r"\textbf{emphasis}" in out


def test_every_reference_is_numbered_and_in_acs_order():
    refs = bs.references()
    assert refs.count(r"\bibitem") == 13
    for number in range(1, 14):
        assert f"bibitem{{ref{number}}}" in refs
    # ACS puts the journal in italic and the year in bold.
    assert r"\emph{J. Chem. Inf. Model.} \textbf{2023}" in refs
    assert r"\emph{Nat. Commun.} \textbf{2023}" in refs


def test_preamble_declares_both_authors_and_one_corresponding_address():
    assert bs.PREAMBLE.count(r"\author{") == 2
    assert bs.PREAMBLE.count(r"\email{") == 1
    assert "linh.ntm@vlu.edu.vn" in bs.PREAMBLE
    assert r"\alsoaffiliation{}" not in bs.PREAMBLE, "an empty one prints a stray marker"


def test_section_numbers_survive_because_the_text_cross_references_them():
    # Starred forms: the numbers are already in the heading text, and a class that
    # numbers sections itself would otherwise print each number twice.
    out = bs.convert_body("## 6. Simulated splits\n\n### 6.2 Why\n\nSee Section 6.2.")
    assert r"\section*{6. Simulated splits}" in out
    assert r"\subsection*{6.2 Why}" in out


@pytest.mark.parametrize("raw,expected", [("±", r"$\pm$"), ("×", r"$\times$"),
                                          ("—", "---"), ("–", "--")])
def test_characters_pdflatex_cannot_read_are_translated(raw, expected):
    assert expected in bs.escape(f"a {raw} b")


def test_no_literal_non_ascii_survives_conversion():
    """pdflatex reads bytes; a stray en dash or sigma stops the build."""
    sample = "Coverage 0.587–0.947 with ±1.645σ and 40 × 8 comparisons ≥ nominal."
    out = bs.inline(sample)
    assert all(ord(c) < 128 for c in out), [c for c in out if ord(c) > 127]


def test_citations_above_nine_are_not_left_as_literal_text():
    """A single-digit pattern silently drops [10] and up once the list passes nine."""
    out = bs.inline("see [7] and [10] and [13]")
    assert r"\cite{ref7}" in out
    assert r"\cite{ref10}" in out
    assert r"\cite{ref13}" in out
    assert "[10]" not in out and "[13]" not in out


def test_structural_bold_survives_but_is_not_required_elsewhere():
    """Table and Figure markers drive the float conversion; prose bold is incidental."""
    rows = ["| a | b |", "| --- | ---: |", "| 1 | 2 |"]
    body = bs.convert_body("**Table 3. A caption.**\n\n" + "\n".join(rows))
    assert r"\caption{A caption.}" in body
    assert r"\begin{table}" in body
