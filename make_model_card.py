import json
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

REPORTS_DIR = Path("reports")
BOSE_PATH = Path("docs/BOSE_checklist.md")
OUTPUT_PDF = Path("model_card.pdf")


def read_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def main() -> None:
    metrics = read_json(REPORTS_DIR / "metrics.json")
    bias = read_json(REPORTS_DIR / "bias_audit.json")
    bose_text = BOSE_PATH.read_text() if BOSE_PATH.exists() else ""

    c = canvas.Canvas(str(OUTPUT_PDF), pagesize=letter)
    width, height = letter
    y = height - 50

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Model Card")
    y -= 30

    if metrics:
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y, "Evaluation Metrics")
        y -= 20
        c.setFont("Helvetica", 12)
        for k, v in metrics.items():
            c.drawString(60, y, f"{k}: {v}")
            y -= 15
        y -= 10

    if bias:
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y, "Bias Audit Results")
        y -= 20
        c.setFont("Helvetica", 12)
        for k, v in bias.items():
            c.drawString(60, y, f"{k}: {v}")
            y -= 15
        y -= 10

    if bose_text:
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y, "Basic Online Safety Expectations")
        y -= 20
        c.setFont("Helvetica", 12)
        for line in bose_text.splitlines():
            if not line.strip():
                continue
            c.drawString(60, y, line.strip())
            y -= 15
            if y < 50:
                c.showPage()
                y = height - 50
                c.setFont("Helvetica", 12)
    c.save()
    print(f"Saved model card to {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
