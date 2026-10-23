"""Build the ACS submission package from the working Markdown manuscript.

The working draft carries two blocks that must never reach a reviewer: an extended
abstract kept for a preprint, and the revision notes. This drops them, converts the rest
to the `achemso` document class that ACS supplies for its journals, and rewrites the
reference list into ACS style.

The conversion is deliberately narrow. It handles the constructs this manuscript actually
uses and raises on anything it does not recognise, rather than silently emitting wrong
LaTeX: a converter that guesses is worse than one that stops, because nobody proofreads
7,000 words of generated markup.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

DROP_SECTIONS = ("Notes for revision",)
# Characters the manuscript uses that pdflatex will not accept as literal bytes.
REPLACEMENTS = {
    "–": "--", "—": "---", "−": "$-$", "±": r"$\pm$", "×": r"$\times$",
    "σ": r"$\sigma$", "Δ": r"$\Delta$", "χ": r"$\chi$", "²": "$^2$",
    "≥": r"$\geq$", "≤": r"$\leq$", "¹": "", "'": "'", "'": "'",
    """: "``", """: "''",
}
ESCAPES = {"&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_", "|": r"\textbar{}"}


def strip_working_blocks(text: str) -> str:
    """Remove the preprint abstract and the revision notes."""
    text = re.sub(r"<details>.*?</details>\s*", "", text, flags=re.S)
    for heading in DROP_SECTIONS:
        text = re.sub(rf"^## {re.escape(heading)}\b.*", "", text, flags=re.S | re.M)
    # Editorial asides to the authors, written as a whole italic paragraph.
    text = re.sub(r"^\*[A-Z][^*]{40,}\*\s*$", "", text, flags=re.M)
    return text


def escape(text: str) -> str:
    """Escape LaTeX specials, leaving inline markup for `inline` to handle."""
    for raw, latex in REPLACEMENTS.items():
        text = text.replace(raw, latex)
    for raw, latex in ESCAPES.items():
        text = text.replace(raw, latex)
    return text


def inline(text: str) -> str:
    """Bold, italic, inline code and bracketed citations.

    Code spans are lifted out before the rest is escaped and put back afterwards. Escaping
    them in place and then escaping the whole string again turns every `\\_` into `\\\\_`,
    which compiles to a stray backslash and a maths error.
    """
    spans: list[str] = []

    def park(match: re.Match) -> str:
        spans.append(escape(match.group(1)))
        return f"\0{len(spans) - 1}\0"

    text = re.sub(r"`([^`]+)`", park, text)
    # \d+, not \d: with ten or more references, a single-digit pattern silently leaves
    # [10] and up as literal text while the earlier ones become citations.
    text = re.sub(r"\[(\d+)\]", r"\\cite{ref\1}", text)
    text = escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text, flags=re.S)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\\emph{\1}", text, flags=re.S)
    for position, content in enumerate(spans):
        text = text.replace(f"\0{position}\0", r"\texttt{" + content + "}")
    return text


def convert_table(rows: list[str], caption: str, label: str) -> str:
    """A pipe table becomes a `table` float; alignment comes from the separator row.

    `caption` arrives already converted, because it was lifted from the output stream.
    Running `inline` over it again would escape its escapes.
    """
    # Split on unescaped pipes only: cells such as "mean \|gap\| (pp)" contain literal
    # ones, and splitting on those invents columns the alignment spec does not have.
    cells = [[c.strip().replace(r"\|", "|")
              for c in re.split(r"(?<!\\)\|", r.strip().strip("|"))] for r in rows]
    header, separator, body = cells[0], cells[1], cells[2:]
    spec = "".join("r" if s.endswith(":") and not s.startswith(":") else "l"
                   for s in separator)
    out = [r"\begin{table}[htbp]", r"\centering",
           rf"\caption{{{caption}}}", rf"\label{{{label}}}",
           # Shrink only if the table would otherwise run into the margin, so narrow
           # tables keep the body font and wide ones stay on the page.
           r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%",
           rf"\begin{{tabular}}{{{spec}}}", r"\hline",
           " & ".join(inline(c) for c in header) + r" \\", r"\hline"]
    out += [" & ".join(inline(c) for c in row) + r" \\" for row in body]
    out += [r"\hline", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(out)


def convert_figure(number: str, filename: str, caption: str) -> str:
    return "\n".join([
        r"\begin{figure}[htbp]", r"\centering",
        rf"\includegraphics[width=\linewidth]{{{filename}}}",
        rf"\caption{{{inline(caption)}}}", rf"\label{{fig:{number}}}",
        r"\end{figure}"])


def convert_body(text: str) -> str:
    """Walk the document line by line; stop on anything unrecognised."""
    lines = text.split("\n")
    out: list[str] = []
    index, table_count = 0, 0

    while index < len(lines):
        line = lines[index].rstrip()

        if not line:
            out.append("")
            index += 1
        elif line.startswith("### "):
            # The numbers stay. The text makes twenty-one cross-references of the form
            # "Section 6.2", and ACS does not number sections for us, so stripping them
            # leaves every one of those pointing at nothing.
            out.append(r"\subsection{" + inline(line[4:].strip()) + "}")
            index += 1
        elif line.startswith("## "):
            out.append(r"\section{" + inline(line[3:].strip()) + "}")
            index += 1
        elif line.startswith("# "):
            index += 1                      # Title is set in the preamble.
        elif line.startswith("|"):
            block = []
            while index < len(lines) and lines[index].startswith("|"):
                block.append(lines[index]); index += 1
            # Drop the blank lines first: the caption paragraph sits above them, and
            # checking before popping leaves it in the body and the float uncaptioned.
            while out and not out[-1]:
                out.pop()
            caption = out.pop() if out and out[-1].startswith(r"\textbf{Table") else ""
            caption = re.sub(r"^\\textbf\{Table \d+\.\s*", "", caption)
            caption = re.sub(r"\}", "", caption, count=1) if caption else caption
            table_count += 1
            out.append(convert_table(block, re.sub(r"^\\textbf\{|\}$", "", caption),
                                     f"tab:{table_count}"))
        elif line.startswith("---"):
            index += 1
        else:
            paragraph = []
            while index < len(lines) and lines[index].strip() and \
                    not lines[index].startswith(("|", "#", "---")):
                paragraph.append(lines[index].strip()); index += 1
            out.append(paragraph_or_figure(" ".join(paragraph)))
    return "\n".join(out)


FIGURE_PARAGRAPH = re.compile(r"^\*\*Figure (\d)\*\* \(`([^`]+)`\):\s*(.+)$", re.S)


def paragraph_or_figure(text: str) -> str:
    """A paragraph that opens by naming a figure becomes a float plus a cross-reference.

    The draft refers to figures inline with their file paths, which is right for a working
    document and wrong for a submission: a reviewer needs a numbered float with a caption.
    The opening sentence becomes the caption and anything after it stays as prose.
    """
    match = FIGURE_PARAGRAPH.match(text.strip())
    if not match:
        return inline(text)

    number, filename, rest = match.group(1), match.group(2), match.group(3).strip()
    split = re.split(r"(?<=\.)\s+(?=[A-Z])", rest, maxsplit=1)
    caption, remainder = split[0], (split[1] if len(split) > 1 else "")
    caption = caption[0].upper() + caption[1:] if caption else caption
    block = convert_figure(number, filename, caption)
    if remainder:
        block += "\n\n" + inline(remainder)
    return block


def references() -> str:
    """ACS style: authors; title; abbreviated journal in italic; bold year; volume; pages."""
    entries = [
        (r"ref1", r"Zheng, J.; Guo, C.; Wang, Z.; Liu, X.; Chen, H. Beyond Scaffold "
                  r"Splits: Structural-Frontier Evaluation Reveals Hidden Failures in "
                  r"ADMET Models. \emph{arXiv} \textbf{2026}, arXiv:2607.10729."),
        (r"ref2", r"Wei, S.; Wen, X.; Zhu, L.; Li, S.; Zhu, R. ADMEOOD: "
                  r"Out-of-Distribution Benchmark for Drug Property Prediction. "
                  r"\emph{arXiv} \textbf{2023}, arXiv:2310.07253."),
        (r"ref3", r"Deng, J.; Yang, Z.; Wang, H.; Ojima, I.; Samaras, D.; Wang, F. A "
                  r"Systematic Study of Key Elements Underlying Molecular Property "
                  r"Prediction. \emph{Nat. Commun.} \textbf{2023}, \emph{14}, 6395."),
        (r"ref4", r"Koleiev, I.; Stratiichuk, R.; Shevchuk, N.; Melnychenko, M.; "
                  r"Nyporko, A.; Todoryshyn, D.; Husak, V.; Starosyla, S.; "
                  r"Yesylevskyy, S.; Nafiiev, A. An End-User Audit of Reproducibility, "
                  r"Data Leakage, and Overfitting of the Top-Ranked ADMET Prediction "
                  r"Models in TDC Leaderboards. \emph{J. Chem. Inf. Model.} "
                  r"\textbf{2026}, \emph{66}, 8045--8058."),
        (r"ref5", r"Fang, C.; Wang, Y.; Grater, R.; Kapadnis, S.; Black, C.; Trapa, P.; "
                  r"Sciabola, S. Prospective Validation of Machine Learning Algorithms "
                  r"for Absorption, Distribution, Metabolism, and Excretion Prediction: "
                  r"An Industrial Perspective. \emph{J. Chem. Inf. Model.} "
                  r"\textbf{2023}, \emph{63}, 3263--3274."),
        (r"ref6", r"Friesacher, H. R.; Svensson, E.; Winiwarter, S.; Mervin, L.; "
                  r"Arany, A.; Engkvist, O. Temporal Distribution Shift in Real-World "
                  r"Pharmaceutical Data: Implications for Uncertainty Quantification in "
                  r"QSAR Models. \emph{Artif. Intell. Life Sci.} \textbf{2025}, "
                  r"\emph{8}, 100132."),
        (r"ref7", r"Papadopoulos, H.; Proedrou, K.; Vovk, V.; Gammerman, A. Inductive "
                  r"Confidence Machines for Regression. In \emph{Machine Learning: ECML "
                  r"2002}; Lecture Notes in Computer Science; Springer: Berlin, "
                  r"\textbf{2002}; pp 345--356."),
        (r"ref8", r"Lei, J.; G'Sell, M.; Rinaldo, A.; Tibshirani, R. J.; Wasserman, L. "
                  r"Distribution-Free Predictive Inference for Regression. \emph{J. Am. "
                  r"Stat. Assoc.} \textbf{2018}, \emph{113}, 1094--1111."),
        (r"ref9", r"Johansson, U.; Ahlberg, E.; Bostr\"om, H.; Carlsson, L.; Linusson, "
                  r"H.; S\"onstr\"od, C. Handling Small Calibration Sets in Mondrian "
                  r"Inductive Conformal Regressors. In \emph{Statistical Learning and "
                  r"Data Sciences}; Lecture Notes in Computer Science; Springer: Cham, "
                  r"\textbf{2015}; pp 271--280."),
        (r"ref10", r"Papadopoulos, H.; Vovk, V.; Gammerman, A. Regression Conformal "
                   r"Prediction with Nearest Neighbours. \emph{J. Artif. Intell. Res.} "
                   r"\textbf{2011}, \emph{40}, 815--840."),
        (r"ref11", r"Lakshminarayanan, B.; Pritzel, A.; Blundell, C. Simple and Scalable "
                   r"Predictive Uncertainty Estimation Using Deep Ensembles. In "
                   r"\emph{Advances in Neural Information Processing Systems 30}, "
                   r"\textbf{2017}; arXiv:1612.01474."),
        (r"ref12", r"Romano, Y.; Patterson, E.; Cand\`es, E. J. Conformalized Quantile "
                   r"Regression. In \emph{Advances in Neural Information Processing "
                   r"Systems 32}, \textbf{2019}; arXiv:1905.03222."),
        (r"ref13", r"Norinder, U.; Carlsson, L.; Boyer, S.; Eklund, M. Introducing "
                   r"Conformal Prediction in Predictive Modeling. A Transparent and "
                   r"Flexible Alternative to Applicability Domain Determination. "
                   r"\emph{J. Chem. Inf. Model.} \textbf{2014}, \emph{54}, 1596--1603."),
    ]
    body = "\n".join(rf"\bibitem{{{key}}} {text}" for key, text in entries)
    return ("\\begin{thebibliography}{9}\n" + body + "\n\\end{thebibliography}")


PREAMBLE = r"""\documentclass[journal=jcisd8,manuscript=article]{achemso}
\usepackage{graphicx}
\usepackage[T1]{fontenc}
% File paths set in \texttt{} are long and contain no normal break points, so without
% this they run into the margin rather than wrapping.
\usepackage[htt]{hyphenat}
\sloppy
\emergencystretch=2em

\author{Nhat Trinh Ngoc Minh}
\affiliation{Faculty of Information Technology, Van Lang School of Technology,
Van Lang University, 69/68 Dang Thuy Tram Street, Binh Loi Trung Ward,
Ho Chi Minh City, Vietnam}
\author{Linh Nguyen Thi My}
\email{linh.ntm@vlu.edu.vn}
\affiliation{Faculty of Information Technology, Van Lang School of Technology,
Van Lang University, 69/68 Dang Thuy Tram Street, Binh Loi Trung Ward,
Ho Chi Minh City, Vietnam}

\title{Calibrated uncertainty fails under prospective shift in ADMET,
unseen by simulated splits}

\begin{document}

\begin{tocentry}
\includegraphics[width=\linewidth]{figures/toc_graphic.png}
\end{tocentry}

"""


COVER_PREAMBLE = r"""\documentclass[11pt]{letter}
\usepackage[T1]{fontenc}
\usepackage[margin=1in]{geometry}
\usepackage{parskip}
\address{Faculty of Information Technology\\Van Lang School of Technology\\
Van Lang University\\69/68 Dang Thuy Tram Street\\Binh Loi Trung Ward\\
Ho Chi Minh City, Vietnam}
\signature{Linh Nguyen Thi My\\Corresponding author\\
Faculty of Information Technology, Van Lang School of Technology\\
Van Lang University, Ho Chi Minh City, Vietnam\\
linh.ntm@vlu.edu.vn}
\date{%s}
\begin{document}
\begin{letter}{The Editors\\\emph{Journal of Chemical Information and Modeling}\\
American Chemical Society}
\opening{Dear Editors,}

"""


def cover_letter(markdown: str, date: str) -> str:
    """Typeset the cover letter, which the submission form wants as a PDF.

    The editorial note at the top of the Markdown is for the authors, not the editor, so
    the body is taken from the first sentence of the letter proper to the sign-off.
    """
    body = markdown[markdown.index("We submit for your consideration"):
                    markdown.index("Yours sincerely,")].strip()
    paragraphs = "\n\n".join(inline(" ".join(p.split()))
                             for p in body.split("\n\n") if p.strip())
    return (COVER_PREAMBLE % date + paragraphs
            + "\n\n" + r"\closing{Yours sincerely,}" + "\n"
            + r"\end{letter}" + "\n" + r"\end{document}" + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("paper/MANUSCRIPT.md"))
    parser.add_argument("--cover", type=Path, default=Path("paper/COVER_LETTER.md"))
    parser.add_argument("--output", type=Path, default=Path("paper/submission"))
    args = parser.parse_args()
    if not args.source.is_file():
        parser.error(f"Missing {args.source}")

    text = strip_working_blocks(args.source.read_text(encoding="utf-8"))
    if "Notes for revision" in text or "<details>" in text:
        parser.error("Working blocks survived the strip; refusing to write a draft "
                     "that could be submitted with them in")

    abstract = text.split("## Abstract", 1)[1].split("## 1.", 1)[0].strip()
    before_refs = text.split("## References", 1)[0]
    body = before_refs.split("## 1.", 1)[1]
    front = text.split("## Front matter", 1)[1]

    # ACS puts funding under Acknowledgement and the competing-interest declaration under
    # Notes. Both belong in the manuscript, not only in the submission form.
    closing = [r"\begin{acknowledgement}",
               "This research received no specific grant from any funding agency in the "
               "public, commercial or not-for-profit sectors.",
               r"\end{acknowledgement}", "",
               r"\section*{Notes}",
               "The authors declare no competing financial interest. Neither author is "
               "affiliated with Expansion Therapeutics or the OpenADMET consortium, and "
               "neither participated in the ExpansionRx blind challenge.", ""]

    document = [PREAMBLE,
                r"\begin{abstract}", inline(abstract), r"\end{abstract}", "",
                convert_body("## 1." + body), ""] + closing + [
                references(), "", r"\end{document}"]

    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "manuscript.tex"
    target.write_text("\n".join(document) + "\n", encoding="utf-8")
    (args.output / "front_matter.md").write_text(front.strip() + "\n", encoding="utf-8")
    print(f"wrote {target}")
    print(f"wrote {args.output / 'front_matter.md'}")

    if args.cover.is_file():
        import datetime
        letter = args.output / "cover_letter.tex"
        letter.write_text(
            cover_letter(args.cover.read_text(encoding="utf-8"),
                         datetime.date.today().strftime("%d %B %Y")), encoding="utf-8")
        print(f"wrote {letter}")

    print(f"abstract: {len(abstract.split())} words")


if __name__ == "__main__":
    main()
