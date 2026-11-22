"""Build an ISMRM 2027 DN-QSM draft and its verified figure package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dnqsm-article-mpl")

import numpy as np
from PIL import Image


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


def word_count(text, *, conservative=False):
    text = re.sub(r"\[\d+(?:,\d+)*\]", "", text)
    if conservative:
        text = re.sub(r"[-–—/]", " ", text)
    return len(text.split())


def markdown_text(text):
    return re.sub(r"\[(\d+)\]", r"<sup>\1</sup>", text)


def latex_text(text):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
                    "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
                    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    output = []
    for index, part in enumerate(re.split(r"(\$[^$]*\$)", text)):
        if index % 2:
            output.append(part)
        else:
            part = "".join(replacements.get(char, char) for char in part)
            output.append(re.sub(r"\[(\d+)\]", r"\\textsuperscript{\1}", part))
    return "".join(output)


def make_preview(destination, source):
    # The separate, uncaptained program preview emphasizes the model proposal.
    from build_dnqsm_preview import build_preview

    preview_dir = destination.parent/"preview_model"
    build_preview(preview_dir, source)
    shutil.copy2(preview_dir/"dnqsm_model_preview.png", destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("docs/abstracts/ismrm2027_dnqsm_draft1.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/manuscripts/ismrm2027_dnqsm_draft1_20261027"))
    args = parser.parse_args()
    data = json.loads(args.source.read_text())
    draft_label = data.get("draft_label", "First draft")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    figures = output/"figures"
    figures.mkdir(exist_ok=True)
    limits = dict(title_characters=125, synopsis_words=100, impact_words=40, body_words=750,
                  figure_caption_characters=500, figure_count=5)
    synopsis = " ".join(f"{key}: {value}" for key, value in data["synopsis"].items())
    body = "\n\n".join(f"{key}\n{value}" for key, value in data["body"].items())
    counts = dict(
        title_characters=len(data["title"]), synopsis_words=word_count(synopsis),
        impact_words=word_count(data["impact"]), body_words=word_count(body),
        conservative_synopsis_words=word_count(synopsis, conservative=True),
        conservative_impact_words=word_count(data["impact"], conservative=True),
        conservative_body_words=word_count(body, conservative=True),
        figure_caption_characters={str(f["number"]):len(f["caption"]) for f in data["figures"]},
        figure_count=len(data["figures"]), headings_included=True,
        count_note="Whitespace count with superscript reference numbers excluded; conservative counts also split hyphens and slashes. ECHO may count mathematical notation differently.",
    )
    for key in ("title_characters", "synopsis_words", "impact_words", "body_words", "figure_count"):
        assert counts[key] <= limits[key], (key, counts[key])
    assert counts["conservative_synopsis_words"] <= 100
    assert counts["conservative_impact_words"] <= 40
    assert counts["conservative_body_words"] <= 750
    assert all(n <= 500 for n in counts["figure_caption_characters"].values())
    write_json(output/"word_counts.json", dict(counts=counts, limits=limits, all_within_limits=True))

    arch = Path("artifacts/figures/dnqsm_architecture_overview_20261027")
    cosmos = Path("artifacts/figures/cosmos_article_composites_best18_20261027")
    cosmos_raw = Path("artifacts/figures/cosmos_article_best18_20261027")
    graz = Path("artifacts/figures/graz1_article_composites_best18_20261027")
    inputs = {
        "figure1_architecture.png": arch/"dnqsm_architecture_overview.png",
        "figure1_architecture.pdf": arch/"dnqsm_architecture_overview.pdf",
        "figure3_in_vivo.png": graz/"graz1_reconstructions.png",
        "figure3_in_vivo.pdf": graz/"graz1_reconstructions.pdf",
    }
    for name, source in inputs.items():
        shutil.copy2(source, figures/name)
    # Keep both original COSMOS images bitwise intact in one uploadable figure.
    top = Image.open(cosmos/"cosmos_reconstructions.png").convert("RGB")
    bottom = Image.open(cosmos/"cosmos_differences.png").convert("RGB")
    assert top.width == bottom.width
    combined = Image.new("RGB", (top.width, top.height+bottom.height+2), "black")
    combined.paste(top, (0, 0))
    combined.paste(bottom, (0, top.height+2))
    combined.save(figures/"figure2_cosmos.png", dpi=(600, 600))
    # Embed losslessly through Matplotlib rather than Pillow's JPEG PDF path.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure = plt.figure(figsize=(combined.width/600, combined.height/600), dpi=600)
    figure.figimage(np.asarray(combined), origin="upper")
    figure.savefig(figures/"figure2_cosmos.pdf", dpi=600)
    plt.close(figure)
    with Image.open(figures/"figure2_cosmos.png") as saved:
        a = np.asarray(saved)
        assert np.array_equal(a[:top.height], np.asarray(top))
        assert np.array_equal(a[top.height+2:], np.asarray(bottom))
    make_preview(output/"program_preview.png", cosmos_raw)

    published = json.loads((cosmos_raw/"metrics.json").read_text())
    assert all(abs(published[k]["nrmse_percent"]-v)<.005 for k,v in
               {"soft_tkd":44.38, "dnqsm_final":37.36, "iqsm":48.97, "nextqsm":76.96}.items())
    graz_provenance = json.loads((graz/"provenance.json").read_text())
    current_iqsm = Path(graz_provenance["source_files"]["iqsm"]["path"])
    current_hash = hashlib.file_digest(current_iqsm.open("rb"), "sha256").hexdigest()
    assert current_hash == graz_provenance["source_files"]["iqsm"]["sha256"], "Graz iQSM2 changed; regenerate Graz figures first"

    file_names = {1:"figure1_architecture", 2:"figure2_cosmos", 3:"figure3_in_vivo"}
    md = [f"# {data['title']}", "", f"*{draft_label} for author review; author metadata and unresolved provenance are listed at the end.*", "", "## Synopsis", ""]
    md += [f"**{key}:** {value}\n" for key, value in data["synopsis"].items()]
    md += ["## Impact", "", data["impact"], ""]
    for section, value in data["body"].items():
        md += [f"## {section}", "", markdown_text(value), ""]
    md += ["## References", ""]
    md += [f"{i}. {value}" for i, value in enumerate(data["references"], 1)]
    md += ["", "## Figures", ""]
    for figure in data["figures"]:
        number = figure["number"]
        md += [f"### Figure {number}. {figure['name']}", "",
               f"![Figure {number}](figures/{file_names[number]}.png)", "", figure["caption"], ""]
    md += ["## Acknowledgements (separate submission field)", "", data["acknowledgements"], "",
           "## Author notes — outside the abstract body", ""]
    md += [f"- {note}" for note in data["author_notes"]]
    md += ["", "## Submission checks", "",
           f"Title: {counts['title_characters']}/125 characters. Synopsis: {counts['synopsis_words']}/100 words. Impact: {counts['impact_words']}/40 words. Body: {counts['body_words']}/750 words. Three figures; each caption is below 500 characters.", "",
           "The separate program preview is 1200 × 800 pixels (3:2), below 1 MB, with no external caption.", ""]
    (output/"abstract_draft.md").write_text("\n".join(md))
    write_json(output/"submission_fields.json", {key:value for key,value in data.items() if key not in ("author_notes", "draft_label")})
    write_json(output/"source_draft.json", data)
    write_json(output/"author_notes.json", dict(notes=data["author_notes"]))

    tex = [r"\documentclass[11pt]{article}", r"\usepackage[a4paper,margin=20mm]{geometry}",
           r"\usepackage[T1]{fontenc}", r"\usepackage{lmodern}", r"\usepackage{microtype}",
           r"\usepackage{amsmath,graphicx,enumitem}", r"\usepackage[font=small,labelfont=bf]{caption}",
           r"\usepackage[hidelinks]{hyperref}", r"\urlstyle{same}", r"\setlength{\parindent}{0pt}",
           r"\setlength{\parskip}{5pt}", r"\begin{document}",
           r"{\small "+latex_text(draft_label)+r" --- ISMRM 2027 Standard Abstract\par}",
           r"\begin{center}{\Large\bfseries "+latex_text(data["title"])+r"\par}\end{center}",
           r"\textbf{Synopsis}\par"]
    for key, value in data["synopsis"].items():
        tex.append(r"\textbf{"+latex_text(key)+r":} "+latex_text(value)+r"\par")
    tex += [r"\textbf{Impact}\par", latex_text(data["impact"])+r"\par"]
    for section, value in data["body"].items():
        tex += [r"\par\medskip\noindent\textbf{"+section+r"}\par\nobreak", latex_text(value)+r"\par"]
    tex += [r"\medskip\textbf{REFERENCES}", r"\begin{enumerate}[leftmargin=*,nosep,label=\arabic*.]\small"]
    for value in data["references"]:
        prefix, doi = value.rsplit("doi:", 1)
        tex.append(r"\item "+latex_text(prefix)+r"\nolinkurl{doi:"+doi.rstrip(".")+"}.")
    tex += [r"\end{enumerate}", r"\clearpage"]
    for figure in data["figures"]:
        number = figure["number"]
        tex += [r"\begin{figure}[p]\centering",
                r"\includegraphics[width=\linewidth,height=.82\textheight,keepaspectratio]{figures/"+file_names[number]+r".pdf}",
                r"\caption{"+latex_text(figure["caption"])+r"}\end{figure}\clearpage"]
    tex += [r"\section*{Author notes --- not part of the abstract body}",
            r"\textbf{Acknowledgements --- separate submission field}\par",
            latex_text(data["acknowledgements"]),
            r"\par\medskip\textbf{Items to confirm}\par", r"\begin{itemize}[leftmargin=*]"]
    tex += [r"\item "+latex_text(note) for note in data["author_notes"]]
    tex += [r"\end{itemize}", r"\textbf{Submission checks}\par",
            latex_text(f"Title {counts['title_characters']}/125 characters; synopsis {counts['synopsis_words']}/100 words; impact {counts['impact_words']}/40 words; body {counts['body_words']}/750 words. Captions: "+", ".join(str(x) for x in counts["figure_caption_characters"].values())+" characters (limit 500 each)."),
            r"\end{document}"]
    (output/"abstract_draft.tex").write_text("\n".join(tex)+"\n")
    sources = [dict(name=name, source=str(path.resolve()), sha256=hashlib.file_digest(path.open("rb"), "sha256").hexdigest()) for name,path in inputs.items()]
    sources.extend(dict(name=path.name, source=str(path.resolve()), sha256=hashlib.file_digest(path.open("rb"), "sha256").hexdigest()) for path in [cosmos/"cosmos_reconstructions.png", cosmos/"cosmos_differences.png", current_iqsm])
    write_json(output/"evidence.json", dict(
        figure_sources=sources, cosmos_metrics=published,
        graz1_iqsm2_sha256=current_hash,
        current_run="artifacts/runs/dnqsm_original105_softcut01_cosmos_joint30_bs8_20261027_120029",
        training_split=dict(training=95, validation=10), final_finetune_epochs=30, selected_epoch=18,
        native_inference_grids=dict(dnqsm_cosmos=184, nextqsm_cosmos=192, dnqsm_graz1=201, nextqsm_graz1=256),
        preview=dict(width=1200,height=800,caption=None,bytes=(output/"program_preview.png").stat().st_size),
    ))
    (output/"guidelines_and_sources.md").write_text(
        "# Guidelines and source checks\n\n"
        "Checked the [2027 standard abstract instructions](https://www.ismrm.org/27m/call/standard/), "
        "[synopsis/impact guide](https://www.ismrm.org/27m/call/submission-guide/impact-synopsis/), "
        "and [submission workflow](https://www.ismrm.org/27m/call/submission-guide/).\n\n"
        "The draft includes Introduction, Methods, Results, Discussion and Conclusion; a four-part synopsis; an impact statement; "
        "numbered references; three captioned study figures; and an uncaptained 3:2 program preview below 1 MB. "
        "Funding and code/data links are confined to the separate acknowledgements field. Numeric limits and counts are in word_counts.json.\n\n"
        "Literature metadata: [COSMOS](https://doi.org/10.1002/mrm.21828), "
        "[NeXtQSM](https://doi.org/10.1016/j.media.2022.102700), "
        "[iQSM](https://doi.org/10.1016/j.neuroimage.2022.119410). "
        "The [author-posted iQSM preprint](https://arxiv.org/abs/2111.07665) links to the published DOI. "
        "Reference 1 uses a shortened article title. Other bibliographic entries follow the structured draft.\n\n"
        "The provided PDF is for author review. ECHO submission uses separate fields and uploaded figure files; "
        "author notes are not part of the blinded abstract. ISMRM permits AI-assisted writing with author responsibility for accuracy.\n"
    )
    shutil.copy2(__file__, output/Path(__file__).name)
    print(json.dumps(dict(output=str(output.resolve()), counts=counts, all_limits_pass=True)), flush=True)


if __name__ == "__main__":
    main()
