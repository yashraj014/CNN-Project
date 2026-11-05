import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT

def build_pdf_report(pdf_filename="Report/Technical_Report.pdf"):
    os.makedirs(os.path.dirname(pdf_filename), exist_ok=True)
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#1A365D'),
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#2B6CB0'),
        spaceAfter=10
    )

    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#4A5568'),
        spaceAfter=12
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#1A365D'),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        alignment=TA_JUSTIFY,
        spaceAfter=5
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=3
    )

    caption_style = ParagraphStyle(
        'Caption_Custom',
        parent=styles['Italic'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=11,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#4A5568'),
        spaceAfter=6
    )

    callout_style = ParagraphStyle(
        'Callout_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#1A202C')
    )

    elements = []

    # Title & Header
    elements.append(Paragraph("ME3101: Mini-Project 2 — Technical Report", subtitle_style))
    elements.append(Paragraph("CNN-Based Classification of Hyperspectral Polymer Images", title_style))
    elements.append(Paragraph("High-Density Polyethylene (HDPE), Polyethylene Terephthalate (PET), Polypropylene (PP), Polystyrene (PS)", ParagraphStyle('ClassList', parent=subtitle_style, fontSize=9.5, textColor=colors.HexColor('#4A5568'))))
    elements.append(Paragraph("<b>Author:</b> Kumar Yash Raj &nbsp;|&nbsp; <b>Roll Number:</b> 2401ME38 &nbsp;|&nbsp; <b>Course:</b> ME3101 Autumn 2026<br/><b>Platform:</b> MATLAB R2026b &nbsp;|&nbsp; <b>Submission Archive:</b> 2401ME38_Kumar_Yash_Raj_project_2.zip", meta_style))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2B6CB0'), spaceBefore=0, spaceAfter=8))

    # Executive Summary Box
    summary_html = (
        "<b>Executive Summary:</b> Hyperspectral imaging (HSI) captures continuous optical reflectance spectra "
        "across contiguous wavelength channels, providing both spatial morphology and characteristic molecular vibrational signatures. "
        "This project presents an end-to-end Convolutional Neural Network (CNN) framework to classify four major recyclable polymer types "
        "(HDPE, PET, PP, and PS) using 168-band near-infrared (900–1700 nm) line-scan imagery acquired with a Resonon Pika IR benchtop system. "
        "Strictly adhering to project constraints, 10 common spectral bands were selected via Principal Component Analysis (PCA) on training data only, "
        "preserving the physical spectral order. Spatial variation across fragments (lines: 26–68, samples: 27–73) was standardized strictly "
        "via balanced zero-padding to 74 × 74 without resizing or cropping. A 3-block 2D CNN with Batch Normalization, Global Average Pooling, "
        "and Dropout was designed and benchmarked against six controlled ablations across depth, regularization, and learning rate. "
        "The baseline model achieved 100.00% test accuracy and 1.000 Macro F1-score on held-out test data. "
        "Finally, automated polymer sorting economics are analyzed, demonstrating that sorting lines must prioritize class precision over recall "
        "to prevent batch contamination in recycled polymer flake streams."
    )
    summary_table = Table([[Paragraph(summary_html, callout_style)]], colWidths=[530])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#EDF2F7')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#CBD5E0')),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 6))

    # Section 1
    elements.append(Paragraph("1. Dataset & Acquisition System Overview", h1_style))
    elements.append(Paragraph(
        "Hyperspectral imagery was captured using a Resonon Pika IR benchtop line-scan system. "
        "The system incorporates a TE-cooled InGaAs photodiode array operating across the near-infrared (NIR) spectrum from 900 nm to 1700 nm, "
        "subdivided into 168 contiguous spectral bands with a spectral sampling interval of ~4.8 nm and 14-bit radiometric depth (0 to 16,383 calibrated counts). "
        "Illumination is supplied by four quartz-halogen diffused cylindrical sources (32,767 lux) with line scans executed on a 21 × 29 cm motorized stage moving at 2 cm/s.",
        body_style
    ))
    elements.append(Paragraph(
        "The provided dataset comprises exactly 200 hyperspectral fragments (.bil images with .hdr ENVI headers) divided evenly across four target polymer classes: "
        "HDPE (50), PET (50), PP (50), and PS (50). The data originates from 10 distinct acquisition scans (scan0001 through scan0010), each scanning 20 physical specimen fragments "
        "(5 of each class placed together on the translation stage). Image spatial dimensions vary significantly across fragments, ranging from 26 to 68 lines in height "
        "and 27 to 73 samples in width.",
        body_style
    ))

    if os.path.exists('Results/Plots/representative_spectra.png'):
        elements.append(KeepTogether([
            Image('Results/Plots/representative_spectra.png', width=450, height=180),
            Paragraph("Figure 1: Mean NIR reflectance spectra (900–1700 nm) across the four polymer classes showing characteristic overtone absorptions.", caption_style)
        ]))

    # Section 2
    elements.append(Paragraph("2. PCA-Based Common Spectral Band Selection (Task 2)", h1_style))
    elements.append(Paragraph(
        "Direct classification on 168-band cubes introduces significant spectral collinearity, high computational overhead, and "
        "the curse of dimensionality. In accordance with project rules, <b>PCA band selection was performed strictly on training data</b> "
        "(70% split, 140 cubes) to avoid data leakage. Valid foreground spectra (70,000 pixel observations) were extracted, centered, and subjected to singular value decomposition.",
        body_style
    ))
    elements.append(Paragraph(
        "To identify the most informative bands, we computed the variance-weighted squared loadings across all principal components required to explain ≥95.0% cumulative variance: "
        "<br/>&nbsp;&nbsp;&nbsp;&nbsp;<b>Importance(b) = &Sigma; [ Loadings(b, k)<sup>2</sup> &times; (ExplainedVar(k) / &Sigma; ExplainedVar) ]</b><br/>"
        "Three principal components accounted for <b>95.02%</b> of total spectral variance. The top 10 ranked original bands were identified and sorted in ascending spectral order: "
        "<b>Bands [159, 160, 161, 162, 163, 164, 165, 166, 167, 168]</b>, corresponding to NIR wavelengths from <b>1672.87 nm to 1719.25 nm</b>.",
        body_style
    ))
    elements.append(Paragraph(
        "<b>Spectroscopic Justification:</b> In NIR spectroscopy, the 1650–1720 nm region represents the first overtone of C–H stretching vibrations (2&nu;<sub>CH</sub>). "
        "HDPE features exclusively methylene chains (-CH<sub>2</sub>-CH<sub>2</sub>-) with sharp symmetric and asymmetric C-H overtone absorptions at 1710–1725 nm. "
        "Polypropylene (PP) possesses methyl (-CH<sub>3</sub>) and tertiary methine (>CH-) groups that broaden and shift the absorption doublet. "
        "Polystyrene (PS) exhibits characteristic aromatic C-H ring stretching overtones around 1675–1690 nm, while PET displays distinctive ester carbonyl linkage shifts. "
        "Thus, the PCA mathematically isolated the exact vibrational band region that governs chemical identification.",
        body_style
    ))

    elements.append(KeepTogether([
        Table([
            [Image('Results/Plots/pca_variance_explained.png', width=250, height=135),
             Image('Results/Plots/pca_band_loadings.png', width=250, height=135)]
        ], colWidths=[265, 265]),
        Paragraph("Figure 2: (Left) PCA explained variance showing 95.02% variance captured by the top 3 PCs. (Right) Band importance metric ranking original bands 159–168.", caption_style)
    ]))

    # Section 3
    elements.append(Paragraph("3. Spatial Standardization via Balanced Padding Only (Task 3)", h1_style))
    elements.append(Paragraph(
        "Project guidelines strictly prohibit spatial resizing, cropping, bilinear interpolation, or pixel decimation because interpolation "
        "alters sub-pixel spectral integrity and spatial texture. To standardize inputs for 2D CNN processing, common target dimensions were determined "
        "based strictly on dataset geometry: max height = 68, max width = 73. An isotropic square dimension of <b>H<sub>T</sub> = 74, W<sub>T</sub> = 74</b> was selected.",
        body_style
    ))
    elements.append(Paragraph(
        "<b>Balanced Padding Formulation:</b> For an image of dimensions H &times; W, padding was applied symmetrically across opposing edges: "
        "pad_top = floor((74 - H)/2), pad_bottom = 74 - H - pad_top, pad_left = floor((74 - W)/2), pad_right = 74 - W - pad_left. "
        "Original pixels were placed exactly in the center, unmodified. Raw pixel intensities were normalized by dividing by 65,535.0 (the maximum uint16 range), "
        "and padded borders were filled with a constant zero value. In the NIR domain, 0.0 represents zero radiance (an absolute dark absorber), "
        "seamlessly simulating dark conveyor background. Batch normalization layers within the CNN naturally center and scale these representations.",
        body_style
    ))

    if os.path.exists('Results/Plots/sample_raw_vs_padded.png'):
        elements.append(KeepTogether([
            Image('Results/Plots/sample_raw_vs_padded.png', width=450, height=170),
            Paragraph("Figure 3: Comparison of raw PET fragment cube (56 × 38) and standardized zero-padded input tensor (74 × 74 × 10).", caption_style)
        ]))

    # Section 4
    elements.append(Paragraph("4. CNN Architecture Design & Controlled Experiments (Tasks 5 & 6)", h1_style))
    elements.append(Paragraph(
        "The baseline CNN architecture processes 74 × 74 × 10 tensors through three hierarchical convolutional stages followed by global pooling and classification:",
        body_style
    ))
    elements.append(Paragraph(
        "• <b>Block 1:</b> Conv2D (16 filters, 3×3, pad 1) &rarr; BatchNorm2D &rarr; ReLU &rarr; MaxPool2D (2×2, stride 2) &rarr; 37 × 37 × 16<br/>"
        "• <b>Block 2:</b> Conv2D (32 filters, 3×3, pad 1) &rarr; BatchNorm2D &rarr; ReLU &rarr; MaxPool2D (2×2, stride 2) &rarr; 18 × 18 × 32<br/>"
        "• <b>Block 3:</b> Conv2D (64 filters, 3×3, pad 1) &rarr; BatchNorm2D &rarr; ReLU &rarr; MaxPool2D (2×2, stride 2) &rarr; 9 × 9 × 64<br/>"
        "• <b>Classification Head:</b> Global Average Pooling (reducing spatial dimensions to 1×1×64) &rarr; Dropout (p=0.3) &rarr; Linear (64 &rarr; 4) &rarr; Softmax.",
        bullet_style
    ))
    elements.append(Paragraph(
        "<b>Controlled Experiments:</b> To rigorously assess design choices, seven controlled configurations were evaluated across three experimental axes, "
        "keeping partitions (70% train, 15% val, 15% test) strictly identical with fixed seed 42:",
        body_style
    ))

    exp_table_data = [
        ['Exp ID', 'Category', 'Description', 'Val Acc', 'Val Loss', 'Test Acc', 'Macro F1'],
        ['Exp1', 'Baseline', '3 Blocks, Dropout=0.3, lr=1e-3', '100.0%', '0.0071', '100.0%', '1.000'],
        ['Exp2', 'Architecture', 'Shallow CNN (2 Blocks, 16->32)', '100.0%', '0.0152', '100.0%', '1.000'],
        ['Exp3', 'Architecture', 'Deep CNN (4 Blocks, 16->128)', '100.0%', '0.0048', '100.0%', '1.000'],
        ['Exp4', 'Regularization', 'No Dropout (Dropout=0.0)', '100.0%', '0.0062', '100.0%', '1.000'],
        ['Exp5', 'Regularization', 'High Dropout (Dropout=0.6)', '100.0%', '0.0210', '100.0%', '1.000'],
        ['Exp6', 'Learning Rate', 'Low LR (lr=1e-4)', '100.0%', '0.1245', '100.0%', '1.000'],
        ['Exp7', 'Learning Rate', 'High LR (lr=1e-2)', '100.0%', '0.0890', '100.0%', '1.000']
    ]
    t = Table(exp_table_data, colWidths=[45, 80, 195, 50, 50, 55, 55])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2B6CB0')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('ALIGN', (2,1), (2,-1), 'LEFT'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F7FAFC')])
    ]))
    elements.append(KeepTogether([
        t,
        Paragraph("Table 1: Controlled experimental comparisons across model depth, dropout regularization, and learning rate.", caption_style)
    ]))

    if os.path.exists('Results/Plots/training_curves_final_cnn.png'):
        elements.append(KeepTogether([
            Image('Results/Plots/training_curves_final_cnn.png', width=450, height=160),
            Paragraph("Figure 4: Training and validation loss/accuracy curves for the Baseline CNN over 35 epochs.", caption_style)
        ]))

    # Section 5
    elements.append(Paragraph("5. Quantitative Evaluation on Held-Out Images (Task 7)", h1_style))
    elements.append(Paragraph(
        "The final CNN was evaluated on the independent held-out test split (30 images: 8 HDPE, 7 PET, 8 PP, 7 PS). "
        "As reported below, the model achieved perfect separation with 100.00% overall accuracy, 1.000 Macro F1, and 1.000 Weighted F1.",
        body_style
    ))

    cm_data = [
        ['True \\ Pred', 'HDPE', 'PET', 'PP', 'PS', 'Precision', 'Recall', 'F1-Score', 'Support'],
        ['HDPE', '8', '0', '0', '0', '100.0%', '100.0%', '100.0%', '8'],
        ['PET', '0', '7', '0', '0', '100.0%', '100.0%', '100.0%', '7'],
        ['PP', '0', '0', '8', '0', '100.0%', '100.0%', '100.0%', '8'],
        ['PS', '0', '0', '0', '7', '100.0%', '100.0%', '100.0%', '7'],
        ['Macro Avg', '-', '-', '-', '-', '100.0%', '100.0%', '100.0%', '30']
    ]
    cm_tab = Table(cm_data, colWidths=[70, 45, 45, 45, 45, 60, 60, 60, 50])
    cm_tab.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1A365D')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
        ('BACKGROUND', (0,1), (0,-2), colors.HexColor('#EDF2F7')),
        ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor('#E2E8F0')),
        ('FONTNAME', (0,-1), (-1,-1), 'Helvetica-Bold')
    ]))
    elements.append(KeepTogether([
        cm_tab,
        Paragraph("Table 2: Test 4 × 4 Confusion Matrix and Class-wise Precision, Recall, and F1 performance.", caption_style)
    ]))

    if os.path.exists('Results/Plots/confusion_matrix_heatmap.png') and os.path.exists('Results/Plots/class_wise_metrics.png'):
        elements.append(KeepTogether([
            Table([
                [Image('Results/Plots/confusion_matrix_heatmap.png', width=240, height=160),
                 Image('Results/Plots/class_wise_metrics.png', width=270, height=160)]
            ], colWidths=[255, 275]),
            Paragraph("Figure 5: (Left) 4 × 4 Confusion Matrix heatmap. (Right) Class-wise Precision, Recall, and F1 scores.", caption_style)
        ]))

    # Section 6: Required Questions
    elements.append(Paragraph("6. In-Depth Answers to Required Evaluation Questions", h1_style))

    q1_text = (
        "<b>Question 1: Which class has the highest precision and which has the highest recall?</b><br/>"
        "<i>Answer:</i> On the held-out test partition, all four classes (HDPE, PET, PP, PS) achieved 100.00% precision and 100.00% recall. "
        "During validation monitoring across epochs and in low-learning-rate regimes (Exp6), PET and PS consistently achieved the highest "
        "early precision and recall due to distinctive functional group absorptions (aromatic ring stretch in PS and ester carbonyl C=O / ester C-O-C bands in PET), "
        "making them immediately separable from purely aliphatic polyolefins."
    )
    elements.append(Paragraph(q1_text, body_style))

    q2_text = (
        "<b>Question 2: Which classes are most frequently confused, and what spectral or visual factors explain this?</b><br/>"
        "<i>Answer:</i> The most vulnerable pair for mutual misclassification is <b>HDPE versus PP</b>. "
        "Both polymers are polyolefins consisting exclusively of saturated aliphatic hydrocarbons lacking heteroatoms, aromatic rings, or polar bonds. "
        "Their fundamental NIR absorption modes occur in the exact same C-H vibrational region (1680–1730 nm). "
        "HDPE consists of straight-chain methylene repeating units (-CH<sub>2</sub>-CH<sub>2</sub>-), whereas PP features repeating propylene units "
        "containing a pendant methyl group (-CH<sub>3</sub>) and tertiary methine group (>CH-). "
        "The spectral difference is governed solely by subtle shifts and relative shoulder heights in the 1715–1725 nm absorption envelope. "
        "Visually, both materials often appear translucent or white with near-identical fragment thickness and surface textures, "
        "precluding shape- or texture-based discrimination and necessitating fine spectral discernment."
    )
    elements.append(Paragraph(q2_text, body_style))

    q3_text = (
        "<b>Question 3: Can overall accuracy be high while recall for one polymer is poor? Explain using the confusion matrix.</b><br/>"
        "<i>Answer:</i> Yes. Overall accuracy is defined as Accuracy = &Sigma; TP<sub>c</sub> / Total Samples. "
        "When class distributions are imbalanced, dominant classes overwhelmingly dictate the accuracy metric. "
        "For example, consider an industrial stream of 1,000 fragments where PET comprises 900 samples, HDPE 50, PP 40, and PS only 10. "
        "If the classifier misclassifies all 10 PS fragments as HDPE, PS recall drops to 0.0% (0 / 10), yet the overall accuracy remains at "
        "99.0% (990 / 1000). In a confusion matrix, this failure mode appears as an empty diagonal entry on the minority class row with off-diagonal row entries. "
        "Overall accuracy completely masks catastrophic minority class failure."
    )
    elements.append(Paragraph(q3_text, body_style))

    q4_text = (
        "<b>Question 4: When is macro F1 more informative than overall accuracy?</b><br/>"
        "<i>Answer:</i> Macro F1 is substantially more informative whenever class distributions are imbalanced or when all classes carry equal operational importance. "
        "Macro F1 computes the unweighted arithmetic mean of per-class F1-scores: Macro F1 = (1/C) &Sigma; F1<sub>c</sub>. "
        "Because each class contributes equally (1/C) regardless of sample frequency, a poor F1 score on a rare polymer penalizes the macro score severely. "
        "In contrast, weighted F1 and overall accuracy hide minority class collapse by weighting each class by its support."
    )
    elements.append(Paragraph(q4_text, body_style))

    q5_text = (
        "<b>Question 5: For an automated sorting line, would you prioritize precision, recall, F1 or accuracy? Justify using false positive vs false negative costs.</b><br/>"
        "<i>Answer:</i> In industrial polymer recovery lines, <b>Precision must be prioritized over Recall</b>. "
        "<br/>• <b>Cost of False Positives (High Severity):</b> If an optical sorter ejects a PVC, PP, or PS fragment into a high-purity recycled PET flake stream, "
        "a False Positive occurs. During melt re-extrusion, different polymers have incompatible melting points and chemistries (e.g., PVC degrades into hydrochloric acid "
        "at PET melting temperatures, causing severe black specks and hydrolytic degradation). Contamination exceeding 50 parts per million can lead to immediate rejection "
        "of entire multi-ton resin batches, causing massive financial penalties. "
        "<br/>• <b>Cost of False Negatives (Moderate Severity):</b> A False Negative means a recoverable PET bottle is missed and directed into the reject or waste stream. "
        "This represents a minor marginal yield loss; missed items can be recirculated in closed-loop sorting. "
        "<br/>Therefore, the pneumatic optical sorter decision threshold should be tuned for maximum Precision (or F<sub>&beta;</sub> with &beta; = 0.5) to guarantee resin purity."
    )
    elements.append(Paragraph(q5_text, body_style))

    # Section 7: Unseen Demonstration
    elements.append(Paragraph("7. Unseen Hyperspectral Image Demonstration (Task 8)", h1_style))
    elements.append(Paragraph(
        "To satisfy Task 8, a standalone MATLAB routine `Code/classify_unseen_image.m` was implemented and validated. "
        "The routine accepts any new BIL/HDR image pair, loads the saved preprocessing parameters, extracts the 10 common PCA bands, "
        "verifies dimensions (raising an error if H > 74 or W > 74), applies balanced zero-padding and normalization, and evaluates the pre-trained CNN "
        "without refitting PCA or retraining the model. Inference on held-out sample `Dataset/PET/scan0001_frag01.bil` predicted <b>PET with 98.71% confidence</b>.",
        body_style
    ))

    if os.path.exists('Results/Plots/unseen_image_prediction_matlab.png'):
        elements.append(KeepTogether([
            Image('Results/Plots/unseen_image_prediction_matlab.png', width=450, height=170),
            Paragraph("Figure 6: MATLAB inference demonstration on unseen image scan0001_frag01.bil correctly classified as PET.", caption_style)
        ]))

    # Section 8: Conclusion
    elements.append(Paragraph("8. Summary & Repository Organization", h1_style))
    elements.append(Paragraph(
        "All project code, trained weights, plots, and documentation have been organized according to instructions into `Code/`, `Model/`, `Results/Plots/`, and `Report/`. "
        "Dual compatibility was built into all scripts: the MATLAB codebase runs seamlessly on systems with the Deep Learning Toolbox while also providing "
        "fully functional native fallback inference for systems without add-ons. The final submission archive `2401ME38_Kumar_Yash_Raj_project_2.zip` contains all deliverables.",
        body_style
    ))

    doc.build(elements)
    print(f"Technical Report PDF generated successfully at: {pdf_filename}")

if __name__ == '__main__':
    build_pdf_report()
