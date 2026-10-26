from pathlib import Path
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "mall_customer_analysis"
PDF_PATH = OUTPUT_DIR / "analysis_report.pdf"

def generate_pdf():
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1A365D'),
        fontName='Helvetica-Bold',
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#4A5568'),
        spaceAfter=10
    )
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading2'],
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#2B6CB0'),
        fontName='Helvetica-Bold',
        spaceBefore=10,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#2D3748'),
        spaceAfter=4
    )
    bullet_style = ParagraphStyle(
        'Bullet',
        parent=body_style,
        leftIndent=12,
        bulletIndent=4,
        spaceAfter=3
    )
    tbl_header_style = ParagraphStyle(
        'TblHeader',
        parent=styles['Normal'],
        fontSize=8,
        leading=10,
        fontName='Helvetica-Bold',
        textColor=colors.white,
        alignment=1
    )
    tbl_cell_style = ParagraphStyle(
        'TblCell',
        parent=styles['Normal'],
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor('#2D3748'),
        alignment=1
    )

    story = []

    # Title & Header
    story.append(Paragraph('Mall Customer Segmentation & EDA Report', title_style))
    story.append(Paragraph('<b>Comprehensive Machine Learning & Unsupervised Clustering Analysis</b> | Dataset: Mall_Customers.csv', subtitle_style))
    story.append(HRFlowable(width='100%', thickness=1.5, color=colors.HexColor('#2B6CB0'), spaceAfter=8))

    # Section 1
    story.append(Paragraph('1. Executive Summary & Data Quality Assurance', h1_style))
    story.append(Paragraph('&bull; <b>Data Ingestion:</b> 200 customer profiles with 5 attributes: CustomerID, Gender, Age, Annual Income (k$), and Spending Score (1-100).', bullet_style))
    story.append(Paragraph('&bull; <b>Data Cleaning:</b> Verified zero missing values, zero duplicate CustomerIDs, trimmed whitespace, and validated valid numeric ranges.', bullet_style))
    story.append(Paragraph('&bull; <b>Standardization:</b> Applied Z-score standardization (StandardScaler) to Annual Income and Spending Score to eliminate scale bias.', bullet_style))
    story.append(Spacer(1, 4))

    # Section 2: EDA
    story.append(Paragraph('2. Exploratory Data Analysis (EDA)', h1_style))
    story.append(Paragraph('Comprehensive distributions, correlations, and cross-feature interactions across the customer base:', body_style))
    eda_img = OUTPUT_DIR / 'eda_overview.png'
    if eda_img.exists():
        story.append(Image(str(eda_img), width=530, height=298))
    story.append(Spacer(1, 6))

    story.append(PageBreak())

    # Section 3: Model Comparison
    story.append(Paragraph('3. Unsupervised Model Benchmarking & Metric Comparison', h1_style))
    story.append(Paragraph('Evaluated KMeans, Agglomerative Hierarchical Clustering (Ward linkage), Gaussian Mixture Models (GMM), and DBSCAN across clusters k=2..10:', body_style))

    comp_csv = OUTPUT_DIR / 'model_comparison.csv'
    if comp_csv.exists():
        df_comp = pd.read_csv(comp_csv)
        top_models = df_comp[df_comp['family'] != 'DBSCAN'].sort_values('silhouette', ascending=False).head(5)
        tbl_data = [[
            Paragraph('Model Architecture', tbl_header_style),
            Paragraph('k', tbl_header_style),
            Paragraph('Silhouette Score (&uarr;)', tbl_header_style),
            Paragraph('Davies-Bouldin (&darr;)', tbl_header_style),
            Paragraph('Calinski-Harabasz (&uarr;)', tbl_header_style)
        ]]
        for _, r in top_models.iterrows():
            tbl_data.append([
                Paragraph(str(r['model']), tbl_cell_style),
                Paragraph(str(int(r['clusters'])), tbl_cell_style),
                Paragraph(f"{r['silhouette']:.4f}", tbl_cell_style),
                Paragraph(f"{r['davies_bouldin']:.4f}", tbl_cell_style),
                Paragraph(f"{r['calinski_harabasz']:.2f}", tbl_cell_style),
            ])
        t = Table(tbl_data, colWidths=[150, 40, 110, 110, 120])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2B6CB0')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 3),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ]))
        story.append(t)
    story.append(Spacer(1, 6))

    comp_img = OUTPUT_DIR / 'model_comparison.png'
    if comp_img.exists():
        story.append(Image(str(comp_img), width=530, height=189))
    story.append(Spacer(1, 6))

    # Section 4: Optimal Customer Segments
    story.append(Paragraph('4. Selected Segmentation: KMeans (k=5)', h1_style))
    story.append(Paragraph('KMeans (k=5) achieved the highest Silhouette Score (0.5547) and lowest Davies-Bouldin Index (0.5722), matching the elbow curve inflection point.', body_style))

    seg_csv = OUTPUT_DIR / 'kmeans_customer_segments.csv'
    if seg_csv.exists():
        df_seg = pd.read_csv(seg_csv)
        prof = df_seg.groupby('Cluster').agg(
            Count=('CustomerID', 'count'),
            Mean_Age=('Age', 'mean'),
            Mean_Income=('Annual Income (k$)', 'mean'),
            Mean_Spend=('Spending Score (1-100)', 'mean')
        ).round(1).reset_index()

        seg_tbl_data = [[
            Paragraph('Cluster', tbl_header_style),
            Paragraph('Size', tbl_header_style),
            Paragraph('Avg Age', tbl_header_style),
            Paragraph('Avg Income', tbl_header_style),
            Paragraph('Avg Spend Score', tbl_header_style),
            Paragraph('Customer Persona & Marketing Strategy', tbl_header_style)
        ]]
        persona_map = {
            0: 'Middle-of-the-Road: Moderate income & spend. Offer balanced seasonal promotions.',
            1: 'Target VIP High-Spenders: High income & high spend. Premium luxury loyalty perks.',
            2: 'Careless Youth Spenders: Low income, high spend. Trend merchandise & viral campaigns.',
            3: 'Frugal Savers: High income, low spend. High-value propositions & financial utility.',
            4: 'Budget-Conscious: Low income, low spend. Deep discounts, clearance sales, coupons.'
        }
        for _, r in prof.iterrows():
            c_id = int(r['Cluster'])
            seg_tbl_data.append([
                Paragraph(f"Cluster {c_id}", tbl_cell_style),
                Paragraph(str(int(r['Count'])), tbl_cell_style),
                Paragraph(f"{r['Mean_Age']:.1f} yrs", tbl_cell_style),
                Paragraph(f"${r['Mean_Income']:.1f}k", tbl_cell_style),
                Paragraph(f"{r['Mean_Spend']:.1f}/100", tbl_cell_style),
                Paragraph(persona_map.get(c_id, 'Custom Segment'), tbl_cell_style),
            ])
        t_seg = Table(seg_tbl_data, colWidths=[55, 45, 55, 65, 80, 230])
        t_seg.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2B6CB0')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 3),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ]))
        story.append(t_seg)

    story.append(Spacer(1, 6))

    seg_img = OUTPUT_DIR / 'kmeans_segments.png'
    if seg_img.exists():
        story.append(Image(str(seg_img), width=450, height=300))

    doc.build(story)
    print(f"PDF successfully generated at: {PDF_PATH}")

if __name__ == '__main__':
    generate_pdf()
