"""
MindCheck PDF Report Generator
"""
import io, datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, HRFlowable)
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT

PRIMARY = colors.HexColor('#2563EB')
DANGER  = colors.HexColor('#DC2626')
SUCCESS = colors.HexColor('#16A34A')
DARK    = colors.HexColor('#1E293B')
GREY    = colors.HexColor('#64748B')
LIGHT_BLUE = colors.HexColor('#EFF6FF')
LIGHT_GREY = colors.HexColor('#F8FAFC')
BORDER  = colors.HexColor('#E2E8F0')


def _tbl_style(header=False):
    base = [
        ('FONTSIZE',     (0,0), (-1,-1), 10),
        ('GRID',         (0,0), (-1,-1), 0.5, BORDER),
        ('TOPPADDING',   (0,0), (-1,-1), 7),
        ('BOTTOMPADDING',(0,0), (-1,-1), 7),
        ('LEFTPADDING',  (0,0), (-1,-1), 10),
        ('BACKGROUND',   (0, int(header)), (0,-1), LIGHT_BLUE),
        ('TEXTCOLOR',    (0, int(header)), (0,-1), PRIMARY),
        ('FONTNAME',     (0, int(header)), (0,-1), 'Helvetica-Bold'),
        ('ROWBACKGROUNDS',(0, int(header)),(-1,-1),
         [colors.white, LIGHT_GREY] * 20),
    ]
    if header:
        base += [
            ('BACKGROUND', (0,0), (-1,0), PRIMARY),
            ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
            ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ]
    return TableStyle(base)


def generate_pdf(model_name, student_name, result, prob,
                 input_data: dict, metrics: dict,
                 business_alerts: list = [],
                 explanation_notes: list = []) -> io.BytesIO:
    """
    Generate a PDF prediction report.

    Args:
        model_name:        e.g. "KNN (K=5)"
        student_name:      student's name string
        result:            0 or 1
        prob:              [prob_no_dep, prob_dep] as floats
        input_data:        dict of field→value for student info table
        metrics:           dict with keys acc, prec, rec, f1
        business_alerts:   list of alert strings
        explanation_notes: list of explanation strings

    Returns:
        BytesIO PDF buffer ready for st.download_button
    """
    styles = getSampleStyleSheet()

    title_s = ParagraphStyle('T', fontSize=24, textColor=PRIMARY,
                              fontName='Helvetica-Bold', alignment=TA_CENTER,
                              spaceAfter=4)
    sub_s   = ParagraphStyle('S', fontSize=10, textColor=GREY,
                              alignment=TA_CENTER, spaceAfter=2)
    h1_s    = ParagraphStyle('H1', fontSize=13, textColor=PRIMARY,
                              fontName='Helvetica-Bold',
                              spaceBefore=14, spaceAfter=6)
    body_s  = ParagraphStyle('B', fontSize=10, textColor=DARK,
                              spaceAfter=4, leading=14)
    alert_s = ParagraphStyle('A', fontSize=10, textColor=DANGER,
                              spaceAfter=6, leftIndent=8, leading=16)
    note_s  = ParagraphStyle('N', fontSize=10,
                              textColor=colors.HexColor('#1D4ED8'),
                              spaceAfter=6, leftIndent=8, leading=16)
    disc_s  = ParagraphStyle('D', fontSize=8, textColor=GREY, leading=12)
    foot_s  = ParagraphStyle('F', fontSize=8, textColor=GREY,
                              alignment=TA_CENTER)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                             topMargin=1.5*cm, bottomMargin=1.5*cm,
                             leftMargin=2*cm, rightMargin=2*cm)
    W = 17 * cm  # usable width
    story = []

    # ── Header ────────────────────────────────────────────────
    story.append(Paragraph("MindCheck", title_s))
    story.append(Spacer(6, 8))
    story.append(Paragraph(
        "Student Depression Risk — Prediction Report", sub_s))
    story.append(Paragraph(
        f"Generated: {datetime.datetime.now().strftime('%d %B %Y, %I:%M %p')}",
        sub_s))
    story.append(HRFlowable(width="100%", thickness=2,
                              color=PRIMARY, spaceAfter=14))

    # ── Result banner ─────────────────────────────────────────
    risk_col = DANGER if result == 1 else SUCCESS
    risk_lbl = ("⚠  DEPRESSION RISK DETECTED"
                if result == 1 else "✓  NO DEPRESSION DETECTED")
    story.append(Table(
        [[Paragraph(f"<b>{risk_lbl}</b>",
                    ParagraphStyle('BN', fontSize=14,
                                   textColor=colors.white,
                                   alignment=TA_CENTER))]],
        colWidths=[W],
        style=TableStyle([
            ('BACKGROUND',  (0,0),(-1,-1), risk_col),
            ('TOPPADDING',  (0,0),(-1,-1), 13),
            ('BOTTOMPADDING',(0,0),(-1,-1), 13),
        ])
    ))
    story.append(Spacer(1, 12))

    # ── Prediction summary ────────────────────────────────────
    story.append(Paragraph("Prediction Summary", h1_s))
    story.append(Table([
        ["Model Used",      model_name],
        ["Student Name",    student_name],
        ["No Depression",   f"{prob[0]*100:.1f}%"],
        ["Depression Risk", f"{prob[1]*100:.1f}%"],
        ["Final Prediction","Depression" if result==1 else "No Depression"],
    ], colWidths=[6*cm, 11*cm], style=_tbl_style(header=False)))
    story.append(Spacer(1, 12))

    # ── Student info ──────────────────────────────────────────
    story.append(Paragraph("Student Information", h1_s))
    story.append(Table(
        [[k, str(v)] for k,v in input_data.items()],
        colWidths=[6*cm, 11*cm], style=_tbl_style(header=False)))
    story.append(Spacer(1, 12))

    # ── Model performance ─────────────────────────────────────
    story.append(Paragraph("Live Model Performance", h1_s))
    story.append(Table([
        ["Metric",    "Score"],
        ["Accuracy",  f"{metrics.get('acc',0):.2f}%"],
        ["Precision", f"{metrics.get('prec',0):.2f}%"],
        ["Recall",    f"{metrics.get('rec',0):.2f}%"],
        ["F1 Score",  f"{metrics.get('f1',0):.2f}%"],
    ], colWidths=[6*cm, 11*cm], style=_tbl_style(header=True)))
    story.append(Spacer(1, 12))

    # ── Business rule alerts ──────────────────────────────────
    if business_alerts:
        story.append(Paragraph("Risk Alerts", h1_s))
        for alert in business_alerts:
            story.append(Paragraph(f"⚠  {alert}", alert_s))
        story.append(Spacer(1, 8))

    # ── Explanation notes ─────────────────────────────────────
    if explanation_notes:
        story.append(Paragraph("Prediction Explanation", h1_s))
        for note in explanation_notes:
            story.append(Paragraph(f"•  {note}", note_s))
        story.append(Spacer(1, 8))

    # ── Recommendation ────────────────────────────────────────
    story.append(Paragraph("Recommendation", h1_s))
    if result == 1:
        story.append(Paragraph(
            "This student has been flagged as at risk for depression. "
            "It is recommended to: (1) Follow up with a counsellor or mental health professional, "
            "(2) Monitor academic performance and attendance, "
            "(3) Provide peer support resources and campus mental health services.",
            body_s))
    else:
        story.append(Paragraph(
            "This student shows low risk of depression based on their current profile. "
            "Continue to: (1) Maintain a healthy academic-social balance, "
            "(2) Stay aware of changes in mental wellbeing, "
            "(3) Reach out to campus support services if needed.",
            body_s))
    story.append(Spacer(1, 12))

    # ── Footer ────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1,
                              color=BORDER, spaceAfter=8))
    story.append(Paragraph(
        "<i>This report is generated by MindCheck — an AI-powered student mental "
        "health screening tool. This is a screening tool only and does not constitute "
        "professional medical advice. Please consult a qualified mental health "
        "professional for diagnosis and treatment.</i>", disc_s))
    story.append(Spacer(1, 6))

    doc.build(story)
    buf.seek(0)
    return buf