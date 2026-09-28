# Wiki figure staging

Figures are grouped into `model/`, `document/`, `tool/`, and `engineering/`.
The first batch contains three Model figures: workflow, WHS score contributions,
and design checks. The other sections are reserved for later artwork.

SVG is the upload master (editable text); PNG is a 300 dpi preview. Generated
artwork, the preview gallery, and run provenance remain local and are ignored by
Git. Upload reviewed figures to iGEM static hosting before using `WikiFigure` in
the Wiki. Copy captions and alt text from `model/provenance.json`; captions are
deliberately outside the images. The Wiki repository is not modified by this script.

```text
# Install optional plotting dependencies in an authoring environment
python -m pip install -r wiki_figures/requirements.txt

# Generate the figures from a completed interval run, checking its recorded hashes
python wiki_figures/generate_model.py --run workspace/screening/results/whs_interval_02
```

Open `wiki_figures/index.html` to compare the three figures and read their captions.
The contribution chart uses recorded scores, not invented examples or a new
screening run. Its selected source, hashes, versions and code provenance are saved
in `model/provenance.json`. Missing HI values are distinguished from an observed
HI score of zero. A total score is not a probability or evidence of rescue.

The figures use Wiki Blue `#169bb7`, Wiki Green `#059669`, Ink `#1a3348`, white
backgrounds, thin lines and system sans-serif text. Bar patterns distinguish
the scoring terms as well as color. No external fonts or assets are loaded.
