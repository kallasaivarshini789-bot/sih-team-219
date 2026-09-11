"""
Satellite Mission Intelligence & Telemetry PDF Exporter
-------------------------------------------------------
Generates a comprehensive executive intelligence PDF report
including input/output imagery, optical enhancement telemetry,
and post-processing sector analysis breakdowns.
"""

import os
from datetime import datetime
from fpdf import FPDF

class SatelliteReportPDF(FPDF):
    def header(self):
        # Cosmic telemetry top bar
        self.set_fill_color(8, 12, 20)
        self.rect(0, 0, 210, 26, 'F')
        
        # Cyan accent line
        self.set_fill_color(6, 182, 212)
        self.rect(0, 25, 210, 1.5, 'F')

        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(255, 255, 255)
        self.set_xy(10, 6)
        self.cell(0, 8, "SATELLITE CLOUD REMOVAL & OPTICAL ANALYSIS REPORT", 0, 1, 'L')
        
        self.set_font('Helvetica', '', 8)
        self.set_text_color(148, 163, 184)
        self.set_xy(10, 14)
        self.cell(0, 6, "Smart India Hackathon (SIH 2026) | Problem Statement ID: 26209 | Aerospace Telemetry", 0, 1, 'L')
        self.ln(8)

    def footer(self):
        self.set_y(-16)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(148, 163, 184)
        self.cell(0, 10, f"Page {self.page_no()} | Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')} | Confidential Earth Observation Data", 0, 0, 'C')

def generate_mission_pdf(
    report_path,
    input_img_path,
    output_img_path,
    analysis_img_path=None,
    metrics=None,
    analysis_data=None,
    scene_name="Target Region"
):
    pdf = SatelliteReportPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    
    # Mission Metadata Card
    pdf.set_fill_color(248, 250, 252)
    pdf.set_draw_color(203, 213, 225)
    pdf.rect(10, 32, 190, 24, 'DF')

    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_text_color(15, 23, 42)
    pdf.set_xy(14, 34)
    pdf.cell(90, 6, f"MISSION SCENE: {scene_name.upper()}", 0, 0)
    pdf.cell(90, 6, f"TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'R')

    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(71, 85, 105)
    pdf.set_xy(14, 42)
    pdf.cell(90, 6, f"Sensor Platform: Multi-Spectral Optical Satellite", 0, 0)
    pdf.cell(90, 6, f"Dehazing Algorithm: Dark Channel Prior + Guided CLAHE", 0, 1, 'R')
    pdf.set_xy(14, 48)
    pdf.cell(90, 6, f"Processing Engine: OpenCV / PyTorch CV Pipeline", 0, 0)
    pdf.cell(90, 6, f"Execution Status: Nominal / Cleared", 0, 1, 'R')

    # Optical Enhancement Telemetry Grid
    pdf.set_xy(10, 60)
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 6, "1. OPTICAL ENHANCEMENT TELEMETRY", 0, 1)

    if metrics:
        # Table of metrics
        headers = ["Processing Time", "Input Size", "Output Size", "Contrast Boost", "Haze Reduction", "Cloud Cover"]
        values = [
            f"{metrics.get('processing_time', '0.16')}s",
            f"{metrics.get('input_size_kb', '0')} KB",
            f"{metrics.get('output_size_kb', '0')} KB",
            f"+{metrics.get('contrast_boost_pct', '0')}%",
            f"{metrics.get('haze_reduction_pct', '0')}%",
            f"{metrics.get('cloud_coverage_pct', '0')}%"
        ]

        col_w = 190 / len(headers)
        
        pdf.set_fill_color(30, 41, 59)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Helvetica', 'B', 8)
        pdf.set_xy(10, 68)
        for h in headers:
            pdf.cell(col_w, 7, h, 1, 0, 'C', True)
        pdf.ln()

        pdf.set_fill_color(241, 245, 249)
        pdf.set_text_color(15, 23, 42)
        pdf.set_font('Helvetica', 'B', 9)
        pdf.set_xy(10, 75)
        for v in values:
            pdf.cell(col_w, 8, v, 1, 0, 'C', True)
        pdf.ln(12)

    # Side-by-Side Images
    current_y = pdf.get_y()
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_text_color(15, 23, 42)
    pdf.set_xy(10, current_y)
    pdf.cell(90, 6, "A. ORIGINAL CLOUDY INPUT", 0, 0, 'C')
    pdf.cell(10, 6, "", 0, 0)
    pdf.cell(90, 6, "B. ENHANCED OPTICAL OUTPUT", 0, 1, 'C')

    img_y = current_y + 8
    img_w = 88
    img_h = 60

    if os.path.exists(input_img_path):
        pdf.image(input_img_path, x=12, y=img_y, w=img_w, h=img_h)
        pdf.rect(12, img_y, img_w, img_h)

    if os.path.exists(output_img_path):
        pdf.image(output_img_path, x=110, y=img_y, w=img_w, h=img_h)
        pdf.rect(110, img_y, img_w, img_h)

    # Analysis Section
    next_y = img_y + img_h + 10
    pdf.set_xy(10, next_y)
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 6, "2. POST-PROCESSING SECTOR INTELLIGENCE", 0, 1)

    if analysis_data and analysis_img_path and os.path.exists(analysis_img_path):
        ana_y = next_y + 8
        pdf.image(analysis_img_path, x=12, y=ana_y, w=88, h=60)
        pdf.rect(12, ana_y, 88, 60)

        # Stats on the right
        pdf.set_xy(108, ana_y)
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_text_color(6, 182, 212)
        pdf.cell(90, 6, f"DOMAIN: {analysis_data.get('domain', 'Multi-Sector CV').upper()}", 0, 1)

        pdf.set_font('Helvetica', 'B', 9)
        pdf.set_text_color(15, 23, 42)
        
        stat_y = ana_y + 8
        for key, val in analysis_data.items():
            if key in ['module_id', 'title', 'domain', 'chart']:
                continue
            label = key.replace('_', ' ').title()
            pdf.set_xy(108, stat_y)
            pdf.set_font('Helvetica', '', 8)
            pdf.set_text_color(71, 85, 105)
            pdf.cell(48, 5, f"{label}:", 0, 0)
            pdf.set_font('Helvetica', 'B', 8)
            pdf.set_text_color(15, 23, 42)
            pdf.cell(42, 5, f"{val}", 0, 1, 'R')
            stat_y += 6
            if stat_y > ana_y + 55:
                break
    else:
        pdf.set_font('Helvetica', 'I', 9)
        pdf.set_text_color(100, 116, 139)
        pdf.set_xy(10, next_y + 8)
        pdf.cell(0, 6, "Run individual sector modules (NDVI, Flood, Buildings, Land Cover) for full analytical telemetry.", 0, 1)

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    pdf.output(report_path)
    return report_path
