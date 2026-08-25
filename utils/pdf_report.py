"""
MindCheck — Mental Health Report Generator
"""
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, HRFlowable, Image, KeepTogether,
    PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.units import cm, mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import io
import datetime
import matplotlib
matplotlib.use('Agg')


PAGE_W, PAGE_H = A4
W = 17 * cm

C_NAVY      = colors.HexColor('#0F172A')
C_PRIMARY   = colors.HexColor('#1D4ED8')
C_DANGER    = colors.HexColor('#B91C1C')
C_SUCCESS   = colors.HexColor('#15803D')
C_WARNING   = colors.HexColor('#B45309')
C_GREY      = colors.HexColor('#475569')
C_LIGHT     = colors.HexColor('#F8FAFC')
C_BORDER    = colors.HexColor('#CBD5E1')
C_RED_DARK  = colors.HexColor('#7F1D1D')
C_GREEN_DARK= colors.HexColor('#14532D')
C_BLUE_LT   = colors.HexColor('#EFF6FF')
C_RED_LT    = colors.HexColor('#FEF2F2')
C_GREEN_LT  = colors.HexColor('#F0FDF4')
C_AMBER_LT  = colors.HexColor('#FFFBEB')
C_SLATE     = colors.HexColor('#64748B')
C_WHITE     = colors.white


def _styles():
    return {
        'brand'   : ParagraphStyle('BR', fontSize=28, textColor=C_PRIMARY,
                                    fontName='Helvetica-Bold',
                                    alignment=TA_CENTER, spaceAfter=2, spaceBefore=8),
        'tagline' : ParagraphStyle('TG', fontSize=10, textColor=C_GREY,
                                    alignment=TA_CENTER, spaceAfter=2),
        'h1'      : ParagraphStyle('H1', fontSize=11, textColor=C_PRIMARY,
                                    fontName='Helvetica-Bold',
                                    spaceBefore=12, spaceAfter=5),
        'h2'      : ParagraphStyle('H2', fontSize=10, textColor=C_NAVY,
                                    fontName='Helvetica-Bold',
                                    spaceBefore=8, spaceAfter=4),
        'body'    : ParagraphStyle('BD', fontSize=10, textColor=C_NAVY,
                                    leading=16, spaceAfter=4, alignment=TA_JUSTIFY),
        'body_sm' : ParagraphStyle('BS', fontSize=9, textColor=C_GREY,
                                    leading=14, spaceAfter=3),
        'alert'   : ParagraphStyle('AL', fontSize=10, textColor=C_DANGER,
                                    fontName='Helvetica-Bold',
                                    spaceAfter=4, leftIndent=8, leading=15),
        'note'    : ParagraphStyle('NT', fontSize=10, textColor=C_PRIMARY,
                                    spaceAfter=4, leftIndent=8, leading=15),
        'footer'  : ParagraphStyle('FT', fontSize=8, textColor=C_SLATE,
                                    alignment=TA_CENTER, leading=12),
        'disc'    : ParagraphStyle('DC', fontSize=8, textColor=C_SLATE,
                                    leading=12, spaceAfter=3, alignment=TA_JUSTIFY),
        'center'  : ParagraphStyle('CN', fontSize=10, textColor=C_NAVY,
                                    alignment=TA_CENTER),
    }


def _section_header(title, S):
    tbl = Table([[Paragraph(title, ParagraphStyle('SH', fontSize=10,
                  textColor=C_WHITE, fontName='Helvetica-Bold'))]],
                colWidths=[W])
    tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,-1), C_PRIMARY),
        ('TOPPADDING',    (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING',   (0,0), (-1,-1), 10),
    ]))
    return tbl


def _info_table(rows, col_w=None):
    if col_w is None:
        col_w = [5.5*cm, 11.5*cm]
    tbl = Table(rows, colWidths=col_w)
    tbl.setStyle(TableStyle([
        ('FONTSIZE',       (0,0), (-1,-1), 9),
        ('FONTNAME',       (0,0), (0,-1), 'Helvetica-Bold'),
        ('TEXTCOLOR',      (0,0), (0,-1), C_SLATE),
        ('TEXTCOLOR',      (1,0), (1,-1), C_NAVY),
        ('GRID',           (0,0), (-1,-1), 0.4, C_BORDER),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [C_WHITE, C_LIGHT]),
        ('TOPPADDING',     (0,0), (-1,-1), 6),
        ('BOTTOMPADDING',  (0,0), (-1,-1), 6),
        ('LEFTPADDING',    (0,0), (-1,-1), 8),
        ('VALIGN',         (0,0), (-1,-1), 'MIDDLE'),
    ]))
    return tbl


def _confidence_bar(prob_no, prob_dep):
    pct_dep = int(round(prob_dep * 100))
    pct_no  = int(round(prob_no  * 100))
    MIN_W   = 0.05 * W
    dep_w   = max(MIN_W, prob_dep * W) if prob_dep > 0 else 0
    no_w    = max(MIN_W, prob_no  * W) if prob_no  > 0 else 0

    if prob_dep == 0:
        no_w = W; dep_w = 0
    elif prob_no == 0:
        dep_w = W; no_w = 0
    else:
        total = no_w + dep_w
        no_w  = no_w  / total * W
        dep_w = dep_w / total * W

    if dep_w == 0:
        row = [[Paragraph(f'<b>No Depression  {pct_no}%</b>',
                    ParagraphStyle('CB', fontSize=10, textColor=C_WHITE,
                                   fontName='Helvetica-Bold', alignment=TA_CENTER))]]
        tbl = Table(row, colWidths=[W])
        tbl.setStyle(TableStyle([
            ('BACKGROUND',    (0,0), (-1,-1), C_SUCCESS),
            ('TOPPADDING',    (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ]))
    elif no_w == 0:
        row = [[Paragraph(f'<b>Depression Risk  {pct_dep}%</b>',
                    ParagraphStyle('CB', fontSize=10, textColor=C_WHITE,
                                   fontName='Helvetica-Bold', alignment=TA_CENTER))]]
        tbl = Table(row, colWidths=[W])
        tbl.setStyle(TableStyle([
            ('BACKGROUND',    (0,0), (-1,-1), C_DANGER),
            ('TOPPADDING',    (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ]))
    else:
        row = [[
            Paragraph(f'<b>No Depression  {pct_no}%</b>',
                ParagraphStyle('CB', fontSize=10, textColor=C_WHITE,
                               fontName='Helvetica-Bold', alignment=TA_CENTER)),
            Paragraph(f'<b>Depression Risk  {pct_dep}%</b>',
                ParagraphStyle('CB', fontSize=10, textColor=C_WHITE,
                               fontName='Helvetica-Bold', alignment=TA_CENTER)),
        ]]
        tbl = Table(row, colWidths=[no_w, dep_w])
        tbl.setStyle(TableStyle([
            ('BACKGROUND',    (0,0), (0,0), C_SUCCESS),
            ('BACKGROUND',    (1,0), (1,0), C_DANGER),
            ('TOPPADDING',    (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 10),
            ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
        ]))
    return tbl


# ══════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ══════════════════════════════════════════════════════════════
def generate_pdf(
    model_name:         str,
    student_name:       str,
    result:             int,
    prob:               list,
    input_data:         dict,
    metrics:            dict,
    business_alerts:    list = [],
    explanation_notes:  list = [],
    feature_importance: dict = {},
    cm_array                 = None,
    model_color:        str  = '#1D4ED8',
) -> io.BytesIO:

    S   = _styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        topMargin=1.8*cm, bottomMargin=1.8*cm,
        leftMargin=2*cm,  rightMargin=2*cm
    )
    story = []
    now    = datetime.datetime.now()
    ref_no = f"MC-{now.strftime('%Y%m%d')}-{hash(student_name) % 9000 + 1000}"

    # ── HEADER ────────────────────────────────────────────────
    story.append(Spacer(1, 0.3*cm))
    logo_tbl = Table([[
        Paragraph("Mind<font color='#1D4ED8'>Check</font>",
                  ParagraphStyle('LG', fontSize=26, textColor=C_NAVY,
                                  fontName='Helvetica-Bold')),
        Paragraph(
            f"<b>Report No:</b> {ref_no}<br/>"
            f"<b>Date:</b> {now.strftime('%d %B %Y')}<br/>"
            f"<b>Time:</b> {now.strftime('%I:%M %p')}",
            ParagraphStyle('RF', fontSize=8, textColor=C_SLATE,
                           alignment=TA_RIGHT, leading=13)
        )
    ]], colWidths=[9*cm, 8*cm])
    logo_tbl.setStyle(TableStyle([
        ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
        ('LEFTPADDING',   (0,0), (-1,-1), 0),
        ('RIGHTPADDING',  (0,0), (-1,-1), 0),
    ]))
    story.append(logo_tbl)
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "Student Mental Health Screening Report",
        ParagraphStyle('DT2', fontSize=12, textColor=C_SLATE, fontName='Helvetica-Bold')
    ))
    story.append(Paragraph(
        f"AI-Assisted Depression Risk Assessment  ·  {model_name}",
        ParagraphStyle('MT2', fontSize=8, textColor=C_SLATE)
    ))
    story.append(Spacer(1, 0.4*cm))
    story.append(HRFlowable(width='100%', thickness=2,
                              color=C_PRIMARY, spaceAfter=0.5*cm))

    # ── RESULT BANNER ─────────────────────────────────────────
    risk_color = C_DANGER if result == 1 else C_SUCCESS
    risk_icon  = "⚠" if result == 1 else "✓"
    risk_text  = "DEPRESSION RISK DETECTED" if result == 1 else "NO DEPRESSION DETECTED"
    risk_sub   = ("This student shows indicators of depression risk. "
                  "Professional follow-up is recommended."
                  if result == 1 else
                  "This student shows no significant depression indicators "
                  "at this time. Continue routine monitoring.")

    banner = Table([
        [Paragraph(f"<b>{risk_icon}  {risk_text}</b>",
                   ParagraphStyle('BN', fontSize=16, textColor=C_WHITE,
                                   fontName='Helvetica-Bold', alignment=TA_CENTER))],
        [Paragraph(risk_sub,
                   ParagraphStyle('BS2', fontSize=9, textColor=C_WHITE,
                                   alignment=TA_CENTER, leading=13))],
    ], colWidths=[W])
    banner.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,-1), risk_color),
        ('TOPPADDING',    (0,0), (-1,0),  12),
        ('BOTTOMPADDING', (0,0), (-1,0),  4),
        ('TOPPADDING',    (0,1), (-1,1),  0),
        ('BOTTOMPADDING', (0,1), (-1,1),  12),
        ('LEFTPADDING',   (0,0), (-1,-1), 12),
        ('RIGHTPADDING',  (0,0), (-1,-1), 12),
    ]))
    story.append(banner)
    story.append(Spacer(1, 0.4*cm))
    story.append(_confidence_bar(prob[0], prob[1]))
    story.append(Spacer(1, 0.5*cm))

    # ── SECTION A — STUDENT PROFILE ───────────────────────────
    story.append(_section_header("A.  STUDENT PROFILE", S))
    story.append(Spacer(1, 0.2*cm))
    info_rows = [["Student Name", student_name]]
    for k, v in input_data.items():
        info_rows.append([k, str(v)])
    story.append(_info_table(info_rows))
    story.append(Spacer(1, 0.4*cm))

    # ── SECTION B — SCREENING RESULT ──────────────────────────
    story.append(_section_header("B.  SCREENING RESULT", S))
    story.append(Spacer(1, 0.2*cm))
    result_rows = [
        ["Assessment Method",         f"Machine Learning — {model_name}"],
        ["Target Variable",           "Depression (Binary: Yes / No)"],
        ["Depression Probability",    f"{prob[1]*100:.1f}%"],
        ["No Depression Probability", f"{prob[0]*100:.1f}%"],
        ["Screening Outcome",         "AT RISK — Depression Detected"
                                      if result==1 else "LOW RISK — No Depression Detected"],
        ["Confidence Level",          "High"     if max(prob) > 0.80
                                 else "Moderate" if max(prob) > 0.60
                                 else "Low"],
        ["Screening Date",            now.strftime("%d %B %Y, %I:%M %p")],
        ["Report Reference",          ref_no],
    ]
    row_colors = [None, None,
                  C_RED_LT   if result==1 else C_GREEN_LT,
                  C_GREEN_LT if result==1 else C_RED_LT,
                  C_RED_LT   if result==1 else C_GREEN_LT,
                  None, None, None]
    tbl_r = Table(result_rows, colWidths=[5.5*cm, 11.5*cm])
    style_r = [
        ('FONTSIZE',      (0,0), (-1,-1), 9),
        ('FONTNAME',      (0,0), (0,-1), 'Helvetica-Bold'),
        ('TEXTCOLOR',     (0,0), (0,-1), C_SLATE),
        ('TEXTCOLOR',     (1,0), (1,-1), C_NAVY),
        ('GRID',          (0,0), (-1,-1), 0.4, C_BORDER),
        ('TOPPADDING',    (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING',   (0,0), (-1,-1), 8),
        ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
    ]
    for i, rc in enumerate(row_colors):
        if rc:
            style_r.append(('BACKGROUND', (0,i), (-1,i), rc))
    tbl_r.setStyle(TableStyle(style_r))
    story.append(tbl_r)
    story.append(Spacer(1, 0.4*cm))

    # ── SECTION C — RISK INDICATORS ───────────────────────────
    # Status column: Yes (red) / No (green) instead of Present/Absent
    story.append(_section_header("C.  RISK INDICATORS IDENTIFIED", S))
    story.append(Spacer(1, 0.2*cm))

    risk_map = {
        'Anxiety':      ('Anxiety',       'Co-morbid factor — strongly associated with depression'),
        'Panic Attack': ('Panic Attack',  'Strongest predictor in dataset (r=0.341)'),
        'Panic':        ('Panic Attack',  'Strongest predictor in dataset (r=0.341)'),
        'Marital':      ('Married',       'Additional personal stress — significant predictor'),
        'Treatment':    ('Sought Treatment','Indicates awareness of condition'),
    }
    risk_items = []
    for k, v in input_data.items():
        if k in risk_map:
            is_yes = str(v).lower() == 'yes'
            risk_items.append((risk_map[k][0], risk_map[k][1], is_yes))

    if risk_items:
        ri_hdr = [
            Paragraph('<b>Risk Indicator</b>',
                ParagraphStyle('RH', fontSize=9, textColor=C_WHITE, fontName='Helvetica-Bold')),
            Paragraph('<b>Clinical Significance</b>',
                ParagraphStyle('RH', fontSize=9, textColor=C_WHITE, fontName='Helvetica-Bold')),
            Paragraph('<b>Present?</b>',
                ParagraphStyle('RH', fontSize=9, textColor=C_WHITE,
                               fontName='Helvetica-Bold', alignment=TA_CENTER)),
        ]
        ri_rows = [ri_hdr]
        for name, significance, is_yes in risk_items:
            ri_rows.append([
                Paragraph(name, ParagraphStyle('RI', fontSize=9,
                          textColor=C_DANGER if is_yes else C_NAVY,
                          fontName='Helvetica-Bold' if is_yes else 'Helvetica')),
                Paragraph(significance, ParagraphStyle('RI2', fontSize=9, textColor=C_NAVY)),
                Paragraph(
                    "<b>Yes</b>" if is_yes else "No",
                    ParagraphStyle('RS', fontSize=9,
                                   textColor=C_DANGER if is_yes else C_SUCCESS,
                                   fontName='Helvetica-Bold' if is_yes else 'Helvetica',
                                   alignment=TA_CENTER)
                ),
            ])
        ri_tbl = Table(ri_rows, colWidths=[4.5*cm, 10.5*cm, 2*cm])
        ri_tbl.setStyle(TableStyle([
            ('BACKGROUND',    (0,0), (-1,0),  C_PRIMARY),
            ('GRID',          (0,0), (-1,-1), 0.4, C_BORDER),
            ('ROWBACKGROUNDS',(0,1), (-1,-1), [C_WHITE, C_LIGHT]),
            ('TOPPADDING',    (0,0), (-1,-1), 7),
            ('BOTTOMPADDING', (0,0), (-1,-1), 7),
            ('LEFTPADDING',   (0,0), (-1,-1), 8),
            ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
        ]))
        story.append(ri_tbl)
    else:
        story.append(Paragraph("No risk indicators in input data.", S['body_sm']))

    if business_alerts:
        story.append(Spacer(1, 0.3*cm))
        for alert in business_alerts:
            story.append(Paragraph(f"⚠  {alert}", S['alert']))
    story.append(Spacer(1, 0.4*cm))

    # ── SECTION D — SCREENING NOTES ───────────────────────────
    if explanation_notes:
        story.append(_section_header("D.  SCREENING NOTES", S))
        story.append(Spacer(1, 0.2*cm))
        for note in explanation_notes:
            story.append(Paragraph(f"•   {note}", S['body_sm']))
        story.append(Spacer(1, 0.4*cm))

    # ── GENERATED BY ──────────────────────────────────────────
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "This report was automatically generated by the MindCheck AI Screening System. "
        "It is not a clinical diagnosis and should not replace professional medical advice.",
        ParagraphStyle('GEN', fontSize=8, textColor=C_SLATE,
                       alignment=TA_CENTER, leading=13)
    ))
    story.append(Spacer(1, 0.4*cm))

    # ── FOOTER ────────────────────────────────────────────────
    story.append(HRFlowable(width='100%', thickness=0.5,
                              color=C_BORDER, spaceAfter=0.2*cm))
    story.append(Paragraph(
        f"Report Ref: {ref_no}  ·  Generated: {now.strftime('%d %b %Y %H:%M')}  ·  "
        f"MindCheck AI Screening System",
        S['footer']
    ))
    story.append(Spacer(1, 0.15*cm))
    story.append(Paragraph(
        "CONFIDENTIALITY NOTICE: This report contains sensitive mental health information. "
        "It is intended solely for the authorised recipient. This is an AI-assisted "
        "SCREENING TOOL ONLY and does not constitute a clinical diagnosis.",
        S['disc']
    ))

    doc.build(story)
    buf.seek(0)
    return buf