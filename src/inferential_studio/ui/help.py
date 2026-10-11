"""Contextual help: a short guide per stage (Help panel) and the full documents (F1)."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QComboBox, QDialog, QTextBrowser, QVBoxLayout, QWidget

from inferential_studio.infrastructure.paths import resource_path

STAGE_HELP: dict[str, str] = {
    "import": """
<h3>Data Import</h3>
<p>Bring in a <b>process</b> file (historian export: one timestamp column, many tags) and a
<b>lab</b> file (sample time, optional result time and sample id, the measured value).</p>
<ul><li>The wizard detects the timestamp format; ambiguous dates (e.g. 03/04) must be confirmed.</li>
<li>Set the timezone the file was written in. Everything is stored in UTC.</li>
<li>Check the validation report for gaps, duplicates and text cells.</li></ul>
<p>New here? <i>Generate example dataset</i> creates a complete artificial project.</p>""",
    "initial_eda": """
<h3>Initial EDA</h3>
<p>Look at the <b>raw</b> data before changing anything: trends, statistics, correlations,
coverage and a first X–Y screening.</p>
<ul><li>Wheel zooms, middle-drag pans, click pins a time, drag highlights a period.</li>
<li>Shutdowns and regime changes inflate correlations; clean first, then judge.</li></ul>""",
    "cleaning": """
<h3>Cleaning &amp; Exclusions</h3>
<p>Mark data that should not train the model. Nothing is deleted; every edit can be undone.</p>
<ul><li><b>Manual</b>: Select range → Exclude selection (reason, scope).</li>
<li><b>Rules</b>: e.g. <i>Feed_Flow &lt; 50 for at least 10 min</i>.</li>
<li><b>Detectors</b> flag suspicious values for <b>review</b>: accept (bad) or reject (valid).</li>
<li>Real upsets and turndowns are often valid data.</li></ul>""",
    "variables": """
<h3>Variable Selection</h3>
<p>Choose the <b>target</b> (lab variable) and candidate <b>predictors</b>.</p>
<ul><li>Ranking uses the aligned features (Signal Processing settings) on usable lab samples.</li>
<li>High VIF or |r| are warnings: PLS, PCR and Ridge handle correlated inputs.</li>
<li><i>Sample impact</i> shows predictors that cost many samples.</li></ul>""",
    "processing": """
<h3>Signal Processing</h3>
<p>Turn each tag into one value per lab sample.</p>
<ul><li><b>Deadtime</b>: how long X takes to affect Y. The window ends that long before the sample.</li>
<li><b>Window / aggregation</b>: e.g. 30-min mean. Filters are trailing only.</li>
<li><b>Coverage</b> rules leave a sample unusable rather than guessing.</li>
<li>Centred windows or negative delays read the future: retrospective use only.</li>
<li><i>Lag sensitivity</i> uses the training period only.</li></ul>""",
    "model_setup": """
<h3>Model Setup</h3>
<ul><li>Pick MLR, PLS, PCR and/or Ridge; components and alpha are tuned by cross-validation
inside the training period.</li>
<li>The split is chronological: training → validation → final test.</li>
<li>The purge gap keeps neighbouring periods from sharing process data.</li>
<li>Read the problem list before training.</li></ul>""",
    "training": """
<h3>Training &amp; Interpretation</h3>
<ul><li><b>Standardised coefficients</b> compare predictors; the table gives original units.</li>
<li><b>Cross-validation</b> shows how components / alpha were chosen.</li>
<li>PLS: VIP, X weights and X loadings are different things; VIP &gt; 1 is a rule of thumb.</li>
<li>Coefficients describe the model, not physical cause and effect.</li></ul>""",
    "validation": """
<h3>Validation</h3>
<ul><li>All models are compared on the same samples, ranked by validation RMSE.</li>
<li>Check residuals for trends, bias and autocorrelation, and the applicability ranges.</li>
<li><b>Reveal the final test once</b>, after choosing a model.</li>
<li>Good statistics alone do not make a model ready for online use.</li></ul>""",
    "export": """
<h3>Export &amp; Report</h3>
<ul><li><b>Model card (HTML)</b>: everything an engineer needs to review the model; print it to
PDF from a browser.</li>
<li><b>Predictions</b>: per lab sample, Excel or CSV. <b>Feature matrix</b>: for audit.</li>
<li><b>Model file</b> (.imsmodel): plain JSON, safe to share; apply it to new data below.</li>
<li>The equation is valid only with the same feature processing.</li>
<li><b>File → Save Portable Copy</b> puts the data inside the project file.</li></ul>""",
}


class HelpPanel(QTextBrowser):
    """Short guidance for the stage on screen (lives in the right-hand dock)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setOpenExternalLinks(False)
        self.setFrameShape(QTextBrowser.Shape.NoFrame)

    def show_stage(self, key: str) -> None:
        self.setHtml(STAGE_HELP.get(key, "") + "<p><i>Press F1 for the full user guide.</i></p>")


DOCUMENTS = {
    "User guide": "docs/user-guide.md",
    "Methodology and definitions": "docs/methodology.md",
    "File formats": "docs/file-formats.md",
}


class GuideDialog(QDialog):
    """Reads the bundled Markdown documents (offline)."""

    def __init__(self, parent: QWidget | None = None, document: str = "User guide") -> None:
        super().__init__(parent)
        self.setWindowTitle("Inferential Model Studio — documentation")
        self.resize(900, 760)
        self.choice = QComboBox()
        self.choice.addItems(list(DOCUMENTS))
        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(True)
        layout = QVBoxLayout(self)
        layout.addWidget(self.choice)
        layout.addWidget(self.browser, 1)
        self.choice.currentTextChanged.connect(self.show_document)
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, activated=self.close)
        self.choice.setCurrentText(document)
        self.show_document(document)

    def show_document(self, name: str) -> None:
        path = resource_path(DOCUMENTS[name])
        try:
            self.browser.setMarkdown(path.read_text(encoding="utf-8"))
        except OSError:
            self.browser.setPlainText(f"{path} is not available in this installation.")
