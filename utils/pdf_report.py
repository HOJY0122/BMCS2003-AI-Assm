"""
MindCheck — Enhanced PDF Report Generator
Produces a professional A4 PDF with:
  - Result banner
  - Confidence gauge chart
  - Feature contribution chart
  - Model performance table + bar chart
  - Confusion matrix table
  - Student info table
  - Risk alerts + recommendations
"""
import io
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, HRFlowable, Image, KeepTogether
)

# ── Colour palette ─────────────────────────────────────────────
C_PRIMARY   = colors.HexColor('#2563EB')
C_DANGER    = colors.HexColor('#DC2626')
C_SUCCESS   = colors.HexColor('#16A34A')
C_WARNING   = colors.HexColor('#D97706')
C_DARK      = colors.HexColor('#1E293B')
C_GREY      = colors.HexColor('#64748B')
C_LIGHT     = colors.HexColor('#F8FAFC')
C_BORDER    = colors.HexColor('#E2E8F0')
C_BLUE_LIGHT= colors.HexColor('#EFF6FF')
C_GREEN_LT  = colors.HexColor('#F0FDF4')
C_RED_LT    = colors.HexColor('#FEF2F2')
C_AMBER_LT  = colors.HexColor('#FFFBEB')

W = 17 * cm  # usable page width


# ── Helper: chart → ReportLab Image ──────────────────────────
def _fig_to_img(fig, width_cm=17, height_cm=6):
    buf = io.BytesIO()
    fig.savefig(buf, format='PNG', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return Image(buf, width=width_cm*cm, height=height_cm*cm)


# ── Helper: styled paragraph styles ──────────────────────────
def _styles():
    base = getSampleStyleSheet()
    return {
        'title'  : ParagraphStyle('T', fontSize=26, textColor=C_PRIMARY,
                                   fontName='Helvetica-Bold',
                                   alignment=TA_CENTER, spaceAfter=4),
        'sub'    : ParagraphStyle('S', fontSize=10, textColor=C_GREY,
                                   alignment=TA_CENTER, spaceAfter=3),
        'h1'     : ParagraphStyle('H1', fontSize=13, textColor=C_PRIMARY,
                                   fontName='Helvetica-Bold',
                                   spaceBefore=14, spaceAfter=6),
        'h2'     : ParagraphStyle('H2', fontSize=11, textColor=C_DARK,
                                   fontName='Helvetica-Bold',
                                   spaceBefore=10, spaceAfter=4),
        'body'   : ParagraphStyle('B', fontSize=10, textColor=C_DARK,
                                   spaceAfter=4, leading=15),
        'small'  : ParagraphStyle('SM', fontSize=8, textColor=C_GREY,
                                   spaceAfter=3, leading=12),
        'alert'  : ParagraphStyle('AL', fontSize=10, textColor=C_DANGER,
                                   spaceAfter=5, leftIndent=8, leading=15),
        'note'   : ParagraphStyle('NT', fontSize=10,
                                   textColor=colors.HexColor('#1D4ED8'),
                                   spaceAfter=5, leftIndent=8, leading=15),
        'footer' : ParagraphStyle('F', fontSize=8, textColor=C_GREY,
                                   alignment=TA_CENTER),
        'disc'   : ParagraphStyle('D', fontSize=8, textColor=C_GREY,
                                   leading=12, spaceAfter=4),
        'center' : ParagraphStyle('C', fontSize=10, textColor=C_DARK,
                                   alignment=TA_CENTER),
    }


def _tbl(data, col_widths, header=True, row_colors=None):
    """Build a styled table."""
    tbl = Table(data, colWidths=col_widths)
    style = [
        ('FONTSIZE',     (0,0), (-1,-1), 10),
        ('GRID',         (0,0), (-1,-1), 0.5, C_BORDER),
        ('TOPPADDING',   (0,0), (-1,-1), 7),
        ('BOTTOMPADDING',(0,0), (-1,-1), 7),
        ('LEFTPADDING',  (0,0), (-1,-1), 10),
        ('VALIGN',       (0,0), (-1,-1), 'MIDDLE'),
    ]
    if header:
        style += [
            ('BACKGROUND', (0,0), (-1,0), C_PRIMARY),
            ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
            ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ]
    if row_colors:
        for i, c in enumerate(row_colors):
            if c:
                style.append(('BACKGROUND', (0,i), (-1,i), c))
    tbl.setStyle(TableStyle(style))
    return tbl


# ── Chart 1: Confidence gauge (horizontal bar) ────────────────
def _chart_confidence(prob_no, prob_dep, model_name):
    fig, ax = plt.subplots(figsize=(8, 1.2))
    ax.barh([''], [prob_no*100], color='#16A34A', height=0.5)
    ax.barh([''], [prob_dep*100], left=[prob_no*100],
            color='#DC2626', height=0.5)
    ax.axvline(50, color='white', lw=2, ls='--', alpha=0.8)

    if prob_no > 0.12:
        ax.text(prob_no*50, 0, f'No Depression\n{prob_no*100:.1f}%',
                ha='center', va='center', fontsize=10,
                fontweight='bold', color='white')
    if prob_dep > 0.08:
        ax.text(prob_no*100 + prob_dep*50, 0,
                f'Depression\n{prob_dep*100:.1f}%',
                ha='center', va='center', fontsize=10,
                fontweight='bold', color='white')

    ax.set_xlim(0, 100)
    ax.set_xlabel('Probability (%)', fontsize=9)
    ax.set_title(f'{model_name} — Prediction Confidence',
                 fontsize=11, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])
    plt.tight_layout()
    return _fig_to_img(fig, width_cm=17, height_cm=3)


# ── Chart 2: Model metrics bar chart ─────────────────────────
def _chart_metrics(metrics, model_name, color):
    fig, ax = plt.subplots(figsize=(8, 3))
    m_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    m_vals  = [metrics['acc'], metrics['prec'], metrics['rec'], metrics['f1']]
    bars = ax.bar(m_names, m_vals,
                  color=color, edgecolor='white', alpha=0.9, width=0.5)
    for bar, val in zip(bars, m_vals):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+0.8,
                f'{val:.1f}%', ha='center', va='bottom',
                fontsize=11, fontweight='bold', color='#1E293B')
    ax.set_ylim(0, 115)
    ax.set_ylabel('Score (%)', fontsize=10)
    ax.set_title(f'{model_name} — Live Model Performance',
                 fontsize=11, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(100, color='#E2E8F0', ls='--', lw=1)
    plt.tight_layout()
    return _fig_to_img(fig, width_cm=17, height_cm=5)


# ── Chart 3: Feature contribution ────────────────────────────
def _chart_features(feature_importance: dict):
    """feature_importance: {feature_name: contribution_value}"""
    if not feature_importance:
        return None
    sorted_items = sorted(feature_importance.items(), key=lambda x: x[1])
    names  = [k for k,v in sorted_items]
    values = [v for k,v in sorted_items]
    colors_bar = ['#DC2626' if v > 0 else '#16A34A' for v in values]

    fig, ax = plt.subplots(figsize=(8, max(3, len(names)*0.5)))
    bars = ax.barh(names, values, color=colors_bar,
                   edgecolor='white', height=0.6, alpha=0.9)
    ax.axvline(0, color='#1E293B', lw=1.5)
    for bar, val in zip(bars, values):
        ha = 'left' if val >= 0 else 'right'
        offset = 0.003 if val >= 0 else -0.003
        ax.text(val+offset, bar.get_y()+bar.get_height()/2,
                f'{val:+.3f}', va='center', ha=ha,
                fontsize=9, fontweight='bold')
    ax.set_xlabel('Contribution (→ Depression   ← No Depression)', fontsize=9)
    ax.set_title('Feature Contribution to This Prediction',
                 fontsize=11, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return _fig_to_img(fig, width_cm=17, height_cm=max(4, len(names)*0.7))


# ── Chart 4: Confusion matrix heatmap ────────────────────────
def _chart_confusion(cm_array, model_name, color_hex):
    fig, ax = plt.subplots(figsize=(5, 4))
    tn, fp, fn, tp = cm_array.ravel()
    data = [[tn, fp], [fn, tp]]
    labels = [['TN', 'FP'], ['FN', 'TP']]

    base_color = colors.HexColor(color_hex)
    cmap = plt.cm.Blues if color_hex == '#2563EB' else \
           plt.cm.Greens if color_hex == '#16A34A' else plt.cm.Reds

    im = ax.imshow(data, cmap=cmap, alpha=0.7, aspect='auto')
    for i in range(2):
        for j in range(2):
            ax.text(j, i,
                    f'{labels[i][j]}\n{data[i][j]}',
                    ha='center', va='center',
                    fontsize=16, fontweight='bold',
                    color='white' if data[i][j] > max(tn,fp,fn,tp)*0.5 else '#1E293B')

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred: No Depression', 'Pred: Depression'], fontsize=9)
    ax.set_yticklabels(['Actual: No Depression', 'Actual: Depression'], fontsize=9)
    ax.set_title(f'{model_name} — Confusion Matrix',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    return _fig_to_img(fig, width_cm=10, height_cm=6)


# ══════════════════════════════════════════════════════════════
# MAIN GENERATE FUNCTION
# ══════════════════════════════════════════════════════════════
def generate_pdf(
    model_name:        str,
    student_name:      str,
    result:            int,
    prob:              list,
    input_data:        dict,
    metrics:           dict,
    business_alerts:   list = [],
    explanation_notes: list = [],
    feature_importance: dict = {},
    cm_array           = None,
    model_color:       str  = '#2563EB',
) -> io.BytesIO:
    """
    Generate a full professional PDF report with charts and tables.

    Args:
        model_name:         e.g. "KNN (K=5)"
        student_name:       student's name
        result:             0 or 1
        prob:               [prob_no_dep, prob_dep]
        input_data:         dict of field→value
        metrics:            dict with acc, prec, rec, f1, cm
        business_alerts:    list of alert strings
        explanation_notes:  list of explanation strings
        feature_importance: dict {feature: contribution_value}
        cm_array:           confusion matrix numpy array (optional)
        model_color:        hex color for this model's charts
    """
    S = _styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        topMargin=1.5*cm, bottomMargin=1.5*cm,
        leftMargin=2*cm,  rightMargin=2*cm
    )
    story = []

    risk_color = C_DANGER if result == 1 else C_SUCCESS
    risk_label = ("⚠  DEPRESSION RISK DETECTED"
                  if result == 1 else "✓  NO DEPRESSION DETECTED")

    # ── PAGE 1 ────────────────────────────────────────────────

    # Header
    story.append(Paragraph("MindCheck", S['title']))
    story.append(Paragraph("Student Depression Risk — Prediction Report", S['sub']))
    story.append(Paragraph(
        f"Generated: {datetime.datetime.now().strftime('%d %B %Y, %I:%M %p')}  ·  "
        f"Model: {model_name}", S['sub']))
    story.append(HRFlowable(
        width='100%', thickness=2, color=C_PRIMARY, spaceAfter=14))

    # Result banner
    story.append(KeepTogether([
        Table([[Paragraph(f"<b>{risk_label}</b>",
                           ParagraphStyle('BN', fontSize=16,
                                          textColor=colors.white,
                                          alignment=TA_CENTER))]],
              colWidths=[W],
              style=TableStyle([
                  ('BACKGROUND',   (0,0),(-1,-1), risk_color),
                  ('TOPPADDING',   (0,0),(-1,-1), 14),
                  ('BOTTOMPADDING',(0,0),(-1,-1), 14),
                  ('ROUNDEDCORNERS', [8]),
              ])),
        Spacer(1, 10),
    ]))

    # Confidence chart
    story.append(Paragraph("Prediction Confidence", S['h1']))
    story.append(_chart_confidence(prob[0], prob[1], model_name))
    story.append(Spacer(1, 10))

    # Prediction summary table
    story.append(Paragraph("Prediction Summary", S['h1']))
    dep_pct   = f"{prob[1]*100:.1f}%"
    nodep_pct = f"{prob[0]*100:.1f}%"
    summary_rows = [
        ["Field", "Value"],
        ["Model Used",          model_name],
        ["Student Name",        student_name],
        ["No Depression Prob",  nodep_pct],
        ["Depression Risk Prob",dep_pct],
        ["Final Prediction",    "Depression" if result==1 else "No Depression"],
    ]
    row_c = [None, None,
             C_GREEN_LT, C_RED_LT,
             C_RED_LT if result==1 else C_GREEN_LT, None]
    story.append(_tbl(summary_rows, [6*cm, 11*cm], header=True, row_colors=row_c))
    story.append(Spacer(1, 12))

    # Student info table
    story.append(Paragraph("Student Information", S['h1']))
    info_rows = [["Field", "Value"]] + [[k, str(v)] for k,v in input_data.items()]
    story.append(_tbl(info_rows, [6*cm, 11*cm], header=True))
    story.append(Spacer(1, 12))

    # Business alerts
    if business_alerts:
        story.append(Paragraph("⚠  Risk Alerts", S['h1']))
        for alert in business_alerts:
            story.append(Paragraph(f"•  {alert}", S['alert']))
        story.append(Spacer(1, 8))

    # ── PAGE 2 — CHARTS ───────────────────────────────────────

    # Model performance bar chart
    story.append(Paragraph("Live Model Performance", S['h1']))
    story.append(_chart_metrics(metrics, model_name, model_color))
    story.append(Spacer(1, 8))

    # Performance table
    perf_rows = [
        ["Metric", "Score", "Interpretation"],
        ["Accuracy",  f"{metrics['acc']:.2f}%",
         "Overall correct predictions"],
        ["Precision", f"{metrics['prec']:.2f}%",
         "Of Depression predictions, how many were correct"],
        ["Recall",    f"{metrics['rec']:.2f}%",
         "Of actual Depression cases, how many were caught ← most critical"],
        ["F1 Score",  f"{metrics['f1']:.2f}%",
         "Balance between Precision and Recall"],
    ]
    story.append(_tbl(perf_rows, [4*cm, 4*cm, 9*cm], header=True))
    story.append(Spacer(1, 12))

    # Confusion matrix
    if cm_array is not None:
        story.append(Paragraph("Confusion Matrix", S['h1']))
        tn,fp,fn,tp = cm_array.ravel()
        cm_rows = [
            ["",                "Predicted: No Depression", "Predicted: Depression"],
            ["Actual: No Dep",  str(tn) + " ✅ (True Negative)",
                                str(fp) + " ❌ (False Positive)"],
            ["Actual: Dep",     str(fn) + " ⚠️ (False Negative — missed!)",
                                str(tp) + " ✅ (True Positive)"],
        ]
        story.append(_tbl(cm_rows, [4.5*cm, 6.25*cm, 6.25*cm], header=False,
                          row_colors=[
                              colors.HexColor('#F1F5F9'),
                              None, None
                          ]))
        story.append(Spacer(1, 6))
        if fn > 0:
            story.append(Paragraph(
                f"⚠  False Negatives = {fn}: {fn} depressed student(s) were "
                f"missed by this model (not predicted as depressed). "
                f"False Negative Rate: {fn/(fn+tp)*100:.1f}%",
                S['alert']))
        story.append(Spacer(1, 12))

    # Feature contribution chart
    if feature_importance:
        story.append(Paragraph("Feature Contribution to This Prediction", S['h1']))
        story.append(_chart_features(feature_importance))
        story.append(Spacer(1, 8))

    # Explanation notes
    if explanation_notes:
        story.append(Paragraph("Prediction Explanation", S['h1']))
        for note in explanation_notes:
            story.append(Paragraph(f"•  {note}", S['note']))
        story.append(Spacer(1, 8))

    # Recommendation
    story.append(Paragraph("Recommendation", S['h1']))
    if result == 1:
        rec_text = (
            "This student has been flagged as <b>at risk for depression</b>. "
            "Recommended actions: "
            "(1) Arrange a follow-up appointment with a counsellor or mental health professional. "
            "(2) Monitor academic performance and attendance closely. "
            "(3) Connect the student with campus mental health resources and peer support groups. "
            "(4) Consider re-screening after 4–6 weeks of intervention."
        )
    else:
        rec_text = (
            "This student shows <b>low risk of depression</b> based on their current profile. "
            "Recommended actions: "
            "(1) Maintain awareness of changes in behaviour or academic performance. "
            "(2) Encourage participation in campus wellness programmes. "
            "(3) Re-screen if new risk factors emerge (anxiety, panic attacks, academic difficulties)."
        )
    story.append(Paragraph(rec_text, S['body']))
    story.append(Spacer(1, 14))

    # Footer
    story.append(HRFlowable(
        width='100%', thickness=1, color=C_BORDER, spaceAfter=8))
    story.append(Paragraph(
        "<i>This report is generated by MindCheck — an AI-powered student mental health "
        "screening tool built for BMCS2003 Artificial Intelligence, TARUMT 202605. "
        "This is a <b>screening tool only</b> and does not constitute professional "
        "medical advice. Please consult a qualified mental health professional.</i>",
        S['disc']))
    story.append(Paragraph(
        "MindCheck  ·  BMCS2003 Artificial Intelligence  ·  "
        "Tutorial Group 3  ·  Tutor: Dr Goh  ·  TARUMT 202605",
        S['footer']))

    doc.build(story)
    buf.seek(0)
    return buf