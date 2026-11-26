"""Build the final technical report as a Word file (TH SarabunPSK, A4).

Numbers come from docs/IMPROVEMENT_REPORT.md; figures from docs/figures/.

    python tools/make_report_docx.py
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cross_domain import DOMAIN_LABEL, MODEL_LABEL, build_matrix, transfer_mean  # noqa: E402
from src.domains import DOMAINS  # noqa: E402
from src.experiments import CROSS_DOMAIN_MODELS  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / 'docs' / 'figures'
EVAL_XD = ROOT / 'mini_project' / 'results' / 'cross_domain' / 'eval'   # eval JSONs from cross_domain_colab.ipynb
COLLAPSED = {('cnn', 'dibco'), ('kd', 'dibco')}   # output is all-white: val PSNR flat at 10.01, FM 0
OUT = ROOT / 'docs' / 'report'
FONT = 'TH SarabunPSK'
BODY, H1, H2, H3, TABLE = 16, 22, 20, 18, 14

MEMBERS = [
    ('น.ส.กัญญพัชร รอดคล้าย', '1670902111'),
    ('น.ส.คันธมาทน์ จันทร์สว่าง', '1670901428'),
    ('นายชยนนท์ สุขเนาวรัตน์', '1670903523'),
]


# ---------------------------------------------------------------- styling
def _set_font(rpr_owner, size=None, bold=None):
    """Apply FONT to every script slot (Thai is a complex script in Word)."""
    rpr = rpr_owner.get_or_add_rPr()
    fonts = rpr.find(qn('w:rFonts'))
    if fonts is None:
        fonts = OxmlElement('w:rFonts')
        rpr.insert(0, fonts)
    for slot in ('w:ascii', 'w:hAnsi', 'w:cs', 'w:eastAsia'):
        fonts.set(qn(slot), FONT)
    # Theme font slots override explicit ones (heading styles use them).
    for slot in ('w:asciiTheme', 'w:hAnsiTheme', 'w:cstheme', 'w:eastAsiaTheme'):
        fonts.attrib.pop(qn(slot), None)
    if size is not None:
        for tag in ('w:sz', 'w:szCs'):
            el = rpr.find(qn(tag))
            if el is None:
                el = OxmlElement(tag)
                rpr.append(el)
            el.set(qn('w:val'), str(int(size * 2)))
    if bold is not None:
        for tag in ('w:b', 'w:bCs'):
            el = rpr.find(qn(tag))
            if el is None:
                el = OxmlElement(tag)
                rpr.append(el)
            el.set(qn('w:val'), '1' if bold else '0')


def _style(doc, name, size, bold, before=0, after=0):
    st = doc.styles[name]
    _set_font(st.element, size, bold)
    st.font.color.rgb = RGBColor(0, 0, 0)
    st.font.italic = False
    pf = st.paragraph_format
    pf.space_before, pf.space_after = Pt(before), Pt(after)
    pf.line_spacing = 1.0
    pf.keep_with_next = name.startswith('Heading')


def _run(par, text, size=None, bold=False, italic=False):
    run = par.add_run(text)
    _set_font(run._element, size, bold)
    run.italic = italic
    return run


def _rich(par, text, size=None, italic=False):
    """Write text with **bold** spans."""
    for i, part in enumerate(re.split(r'\*\*(.+?)\*\*', text)):
        if part:
            _run(par, part, size, bold=bool(i % 2), italic=italic)


def _field(par, instr):
    run = par.add_run()
    for kind, text in (('begin', None), (None, instr), ('separate', None), ('end', None)):
        if kind:
            el = OxmlElement('w:fldChar')
            el.set(qn('w:fldCharType'), kind)
        else:
            el = OxmlElement('w:instrText')
            el.set(qn('xml:space'), 'preserve')
            el.text = text
        run._element.append(el)
    _set_font(run._element, BODY)


def _shade(cell, hex_fill):
    tcpr = cell._element.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), hex_fill)
    tcpr.append(shd)


# ---------------------------------------------------------------- builders
class Report:
    def __init__(self):
        self.doc = Document()
        sec = self.doc.sections[0]
        sec.page_width, sec.page_height = Inches(8.27), Inches(11.69)
        for side in ('left_margin', 'right_margin', 'top_margin', 'bottom_margin'):
            setattr(sec, side, Inches(1))
        _style(self.doc, 'Normal', BODY, False)
        _style(self.doc, 'Heading 1', H1, True, before=12, after=6)
        _style(self.doc, 'Heading 2', H2, True, before=10, after=4)
        _style(self.doc, 'Heading 3', H3, True, before=8, after=2)
        self.n_fig = self.n_tab = 0

    def h1(self, text):
        self.doc.add_heading(text, level=1)

    def h2(self, text):
        self.doc.add_heading(text, level=2)

    def h3(self, text):
        self.doc.add_heading(text, level=3)

    def p(self, text, indent=True):
        par = self.doc.add_paragraph()
        par.paragraph_format.first_line_indent = Inches(0.5) if indent else None
        par.paragraph_format.space_after = Pt(4)
        _rich(par, text)
        return par

    def items(self, items, numbered=False):
        for i, text in enumerate(items, 1):
            par = self.doc.add_paragraph()
            pf = par.paragraph_format
            pf.left_indent, pf.first_line_indent = Inches(0.75), Inches(-0.25)
            pf.space_after = Pt(2)
            _run(par, f'{i}.\t' if numbered else '•\t')
            pf.tab_stops.add_tab_stop(Inches(0.75))
            _rich(par, text)

    def table(self, caption, header, rows, size=TABLE, widths=None):
        self.n_tab += 1
        cap = self.doc.add_paragraph()
        cap.paragraph_format.space_before = Pt(6)
        cap.paragraph_format.keep_with_next = True
        _run(cap, f'ตารางที่ {self.n_tab} ', bold=True)
        _run(cap, caption)
        tab = self.doc.add_table(rows=1, cols=len(header))
        tab.style = 'Table Grid'
        tab.alignment = WD_TABLE_ALIGNMENT.CENTER
        for r, values in enumerate([header] + rows):
            cells = tab.rows[0].cells if r == 0 else tab.add_row().cells
            for c, value in enumerate(values):
                par = cells[c].paragraphs[0]
                par.paragraph_format.space_after = Pt(0)
                if c > 0 and re.fullmatch(r'[−\-<\d.,\s/()]+', str(value)):
                    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
                _rich(par, str(value), size)
                if r == 0:
                    for run in par.runs:
                        _set_font(run._element, size, True)
                    _shade(cells[c], 'E7E6E6')
        if widths:
            for row in tab.rows:
                for cell, w in zip(row.cells, widths):
                    cell.width = Inches(w)
        # Layout: repeat the header row on a new page, never split a row, and keep
        # short tables on one page (keep_with_next on every row but the last).
        keep_whole = len(tab.rows) <= 25
        for i, row in enumerate(tab.rows):
            tr_pr = row._tr.get_or_add_trPr()
            tr_pr.append(OxmlElement('w:cantSplit'))
            if i == 0:
                tr_pr.append(OxmlElement('w:tblHeader'))
            if keep_whole and i < len(tab.rows) - 1:
                for cell in row.cells:
                    for par in cell.paragraphs:
                        par.paragraph_format.keep_with_next = True
        self.doc.add_paragraph().paragraph_format.space_after = Pt(2)

    def figure(self, path, caption, width=6.0):
        self.n_fig += 1
        par = self.doc.add_paragraph()
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        par.paragraph_format.keep_with_next = True
        par.add_run().add_picture(str(path), width=Inches(width))
        cap = self.doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.paragraph_format.space_after = Pt(8)
        _run(cap, f'รูปที่ {self.n_fig} ', bold=True)
        _run(cap, caption)

    def image(self, path, width):
        par = self.doc.add_paragraph()
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        par.add_run().add_picture(str(path), width=Inches(width))

    def page_break(self):
        self.doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def cross_domain_rows(metric):
    """Rows of the 3x3 zero-shot matrix; in-domain cells marked with *, collapsed rows not shown as numbers."""
    m = build_matrix(EVAL_XD, metric)
    rows = []
    for model in CROSS_DOMAIN_MODELS:
        for src in DOMAINS:
            if (model, src) in COLLAPSED:
                cells = ['collapsed'] * len(DOMAINS)
            else:
                cells = [f"{m[(model, src, t)]:.2f}{'*' if t == src else ''}" for t in DOMAINS]
            rows.append([MODEL_LABEL[model], DOMAIN_LABEL[src]] + cells)
    return rows


def transfer_mean_rows():
    """Rank, model, mean PSNR and FM over the 6 zero-shot transfer tasks (source != target)."""
    psnr = transfer_mean(build_matrix(EVAL_XD, 'psnr'))
    fm = transfer_mean(build_matrix(EVAL_XD, 'fm'))
    order = sorted(CROSS_DOMAIN_MODELS, key=lambda m: -psnr[m])
    rows = []
    for rank, model in enumerate(order, 1):
        note = ' (collapsed เมื่อเทรนบน DIBCO)' if any(mm == model for mm, _ in COLLAPSED) else ''
        rows.append([str(rank), MODEL_LABEL[model] + note, f'{psnr[model]:.2f}', f'{fm[model]:.2f}'])
    return rows


def equation_png(path):
    fig = plt.figure(figsize=(8, 0.7))
    fig.text(0.5, 0.5,
             r'$\mathcal{L} = \Vert \hat{y} - y \Vert_1 + \lambda \, ('
             r'\Vert G_x * \hat{y} - G_x * y \Vert_1 + \Vert G_y * \hat{y} - G_y * y \Vert_1)$',
             ha='center', va='center', fontsize=17)
    fig.savefig(path, dpi=200, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


# ---------------------------------------------------------------- content
def cover(r):
    doc = r.doc
    for _ in range(5):
        doc.add_paragraph()
    for text, size in (('รายงานโครงงาน (Technical Report)', 22),
                       ('Text-Preserving Image Denoising', 26),
                       ('การลด noise ในภาพเอกสารโดยรักษาขอบตัวอักษร', 20)):
        par = doc.add_paragraph()
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        par.paragraph_format.space_after = Pt(6)
        _run(par, text, size, bold=True)
    for _ in range(5):
        doc.add_paragraph()
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _run(par, 'จัดทำโดย', 18, bold=True)
    for name, sid in MEMBERS:
        par = doc.add_paragraph()
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        par.paragraph_format.space_after = Pt(0)
        _run(par, f'{name}    รหัสนักศึกษา {sid}', 18)
    for _ in range(5):
        doc.add_paragraph()
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _run(par, 'รายงานนี้เป็นส่วนหนึ่งของรายวิชา AIE464 Applied AI', 18, bold=True)


# (level, title, page) read back from the exported PDF; Word here opens files
# read-only, so a TOC field cannot be updated and is written out statically.
CONTENTS = [
    (1, 'บทคัดย่อ (Abstract)', 1), (1, '1. บทนำ (Introduction)', 1),
    (1, '2. ข้อมูล (Dataset)', 2), (2, '2.1 การป้องกัน Data Leakage', 2),
    (2, '2.2 Augraphy Shift Test Set', 3), (2, '2.3 Metric ที่ใช้วัดผล', 3),
    (1, '3. วิธีการ (Method)', 3), (2, '3.1 SmallUNet (โมเดลหลัก)', 3),
    (2, '3.2 Sobel Edge Loss', 4), (2, '3.3 Training Augmentation', 5),
    (2, '3.4 เทคนิคหลัง Midterm', 5), (2, '3.5 โมเดลที่ใช้เปรียบเทียบ', 6),
    (1, '4. การตั้งค่าการทดลอง (Experimental Setup)', 7), (2, '4.1 การเปรียบเทียบที่ยุติธรรม', 8),
    (1, '5. ผลการทดลอง (Results)', 8), (2, '5.1 In-domain (Kaggle test, 24 ภาพ)', 8),
    (2, '5.2 Domain Shift (Augraphy, 24 ภาพ)', 10), (2, '5.3 Cross-dataset (DIBCO 2017, 20 ภาพ)', 10),
    (2, '5.4 Fine-tune และการลืมงานเดิม (Forgetting)', 11), (2, '5.5 ประสิทธิภาพ (Efficiency)', 12),
    (2, '5.6 การทดสอบนัยสำคัญ (Paired Wilcoxon Signed-rank, Two-sided)', 13),
    (2, '5.7 Cross-domain Benchmark (Zero-shot 3 × 3)', 13), (2, '5.8 ผลเชิงคุณภาพ', 15),
    (1, '6. อภิปรายผล (Discussion)', 17),
    (1, '7. ข้อจำกัดและสรุป (Limitations and Conclusion)', 18), (2, '7.1 ข้อจำกัด', 18),
    (2, '7.2 สรุป', 18), (1, 'เอกสารอ้างอิง (References)', 19),
]


def contents(r):
    par = r.doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.paragraph_format.space_after = Pt(12)
    _run(par, 'สารบัญ', H1, bold=True)
    for level, title, page in CONTENTS:
        par = r.doc.add_paragraph()
        pf = par.paragraph_format
        pf.left_indent = Inches(0 if level == 1 else 0.4)
        pf.space_after = Pt(2)
        pf.tab_stops.add_tab_stop(Inches(6.27), alignment=2, leader=1)  # right, dots
        _run(par, f'{title}\t{page}', BODY, bold=level == 1)


def body(r, eq_png):
    r.h1('บทคัดย่อ (Abstract)')
    r.p('งานนี้ลด noise ในภาพเอกสารสแกนโดยรักษาขอบตัวอักษรไว้ ใช้โมเดลหลัก SmallUNet (487,009 params) '
        'ที่เทรนด้วย L1 + Sobel edge loss และขนาด input เท่ากับ output เสมอ (H × W → H × W) '
        'Final project ต่อยอดจาก midterm ด้วยการทดสอบนอก domain 2 แบบ คือ Augraphy shift และ DIBCO 2017 '
        'ผลหลักมี 4 ข้อ ดังนี้')
    r.items([
        '**Augmentation:** บน Kaggle ผลไม่ต่างจาก midterm อย่างมีนัยสำคัญ (31.19 เทียบ 31.84 dB, p = 0.121) '
        'แต่บน Augraphy shift PSNR เพิ่มจาก 14.53 เป็น 17.48 dB (p < 0.001)',
        '**Fine-tune:** บน DIBCO 2017 การ fine-tune ด้วย DIBCO 2018 + 2019A เพิ่ม F-measure จาก 61.9 เป็น 84.4 '
        'และ LoRA r=8 ได้ผลเท่ากัน (84.4) โดยเทรนเพิ่มเพียง 63,304 params',
        '**NAFNet:** pretrained NAFNet หลัง fine-tune ได้ FM สูงสุดที่ 88.0 แต่ช้ากว่า U-Net ราว 16 เท่า',
        '**Knowledge Distillation:** ทำให้ ShallowCNN (7,265 params) ดีขึ้น 1.15 dB (p < 0.001)',
    ])
    r.p('Cross-domain benchmark แบบ zero-shot 3 × 3 ชี้ว่าโมเดลทุกตัวเสียผลเมื่อทดสอบนอก domain ที่เทรน '
        'NAFNet ทนที่สุด และ ShallowCNN / KD-CNN collapse เมื่อเทรนบน DIBCO')
    r.p('ข้อจำกัดหลักคือการ fine-tune บน DIBCO ทำให้ผลบน Kaggle ลดลงเหลือ 20.0–23.0 dB')

    r.h1('1. บทนำ (Introduction)')
    r.p('ภาพเอกสารสแกนมักมีคราบ รอยพับ และหมึกซึม ซึ่งทำให้อ่านยาก งานนี้เป็น image denoising '
        '(การลบสิ่งรบกวนออกจากภาพ) ที่ต้องเก็บขอบตัวอักษรให้คม ไม่ใช่ super-resolution หรือ inpainting')
    r.p('ใน midterm เราเทรน SmallUNet ด้วย Sobel edge loss บน Kaggle Denoising Dirty Documents และได้ 31.84 dB '
        'บน test set แต่ยังไม่รู้ว่าโมเดลทนต่อ noise แบบใหม่หรือเอกสารจากแหล่งอื่นได้แค่ไหน '
        'Final project จึงมีเป้าหมาย 3 ข้อ')
    r.items([
        'ทำให้โมเดลทนต่อ noise ที่ไม่เคยเห็นตอนเทรน (domain shift) ทดสอบด้วย Augraphy shift test set',
        'ทำให้โมเดลใช้กับ dataset อื่นได้ (cross-dataset) ทดสอบบน DIBCO 2017',
        'ใช้เทคนิคหลัง midterm ได้แก่ transfer learning (W9), Knowledge Distillation (W10) และ LoRA (W12)',
    ], numbered=True)
    r.p('ทุกโมเดลวัดผลบน test set, metric และ GPU (Colab Tesla T4) ชุดเดียวกัน รวมถึง checkpoint ของ midterm '
        'ที่นำมาประเมินซ้ำด้วยโค้ดชุดเดียวกัน')

    r.h1('2. ข้อมูล (Dataset)')
    r.p('งานนี้ใช้ข้อมูล 4 ชุด เทรนหลักบน Kaggle ส่วนชุดอื่นใช้ทดสอบความทนทานหรือ fine-tune เท่านั้น')
    r.table('ชุดข้อมูลที่ใช้', ['ชุดข้อมูล', 'ใช้ทำ', 'จำนวน', 'Metric'], [
        ['Kaggle Denoising Dirty Documents', 'train / val / test (in-domain)',
         '100 / 20 / 24 คู่ภาพ จาก 25 / 5 / 6 หน้าต้นฉบับ', 'PSNR, SSIM, Text PSNR'],
        ['Augraphy shift', 'test เท่านั้น', '24 ภาพ สร้างจาก Kaggle test', 'PSNR, SSIM, Text PSNR'],
        ['DIBCO 2017', 'test (cross-dataset) เท่านั้น', '20 ภาพ (printed 10, handwritten 10)', 'FM, pFM, PSNR, DRD'],
        ['DIBCO 2018 + 2019A', 'fine-tune และ train ของ cross-domain', '20 หน้า (train 16 / val 4)', 'val PSNR'],
        ['Augraphy train', 'train ของ cross-domain เท่านั้น', '100 / 20 คู่ภาพ สร้างจาก Kaggle train / val '
         '(seed 7, ไม่มี Folding)', 'val PSNR'],
    ], widths=[1.7, 1.5, 1.8, 1.3])
    r.figure(FIG / 'data_samples.png', 'ตัวอย่างภาพ input และ ground truth จาก Kaggle, Augraphy shift และ DIBCO 2017')
    r.h2('2.1 การป้องกัน Data Leakage')
    r.p('Data leakage คือการที่ข้อมูล test รั่วเข้าไปตอนเทรน ทำให้ผลดูดีเกินจริง เราป้องกันไว้ 2 จุด')
    r.items([
        'Kaggle มี noise 4 แบบต่อหน้า เราแบ่ง train/val/test ตามหน้าต้นฉบับก่อนตัดเป็น patch '
        'หน้าเดียวกันจึงไม่อยู่คนละชุด',
        'DIBCO ใช้ปี 2018 + 2019A ในการ fine-tune และใช้ปี 2017 ทดสอบอย่างเดียว',
    ])
    r.h2('2.2 Augraphy Shift Test Set')
    r.p('Augraphy shift คือภาพ Kaggle test ที่ใส่ความเสียหายแบบใหม่ด้วยไลบรารี Augraphy ได้แก่ InkBleed, '
        'BleedThrough, LightingGradient และ Folding สร้างแบบ deterministic (รันกี่ครั้งก็ได้ภาพเดิม) '
        'ความเสียหายเหล่านี้ไม่ถูกใช้ตอนเทรนเลย')
    r.h2('2.3 Metric ที่ใช้วัดผล')
    r.items([
        '**PSNR / SSIM:** วัดความใกล้เคียงกับภาพสะอาด (ground truth) ทั้งภาพ ยิ่งสูงยิ่งดี',
        '**Text PSNR:** PSNR เฉพาะบริเวณตัวอักษร',
        '**FM (F-measure):** วัดว่าพิกเซลหมึกหลัง binarize ด้วย Otsu ตรงกับ ground truth แค่ไหน ยิ่งสูงยิ่งดี',
        '**pFM:** F-measure ที่ดูโครงเส้นตัวอักษร (skeleton)',
        '**DRD:** วัดความผิดเพี้ยนที่สายตามนุษย์สังเกตได้ ยิ่งต่ำยิ่งดี',
    ])

    r.h1('3. วิธีการ (Method)')
    r.h2('3.1 SmallUNet (โมเดลหลัก)')
    r.p('SmallUNet มี 487,009 params (width 16) และทำนาย noise แทนการทำนายภาพสะอาดโดยตรง')
    r.items([
        '**Encoder 3 ชั้น:** ลดขนาด feature map ลงครึ่งหนึ่งด้วย max pool (16×256×256 → 64×64×64)',
        '**Bridge:** ประมวลผลที่ขนาดเล็กที่สุด 128×32×32',
        '**Decoder 3 ชั้น:** ขยายกลับด้วย nearest upsampling แล้วต่อ (concat) กับ skip connection จาก encoder',
        '**Residual output:** output = input − predicted noise',
    ])
    r.figure(FIG / 'unet_diagram.png', 'โครงสร้าง SmallUNet พร้อม shape จริงจากการ forward ภาพ 256×256')
    r.p('ชื่อ "U-Net edge" ในงานนี้หมายถึง SmallUNet ที่เทรนด้วย Sobel edge loss '
        'ไม่ได้หมายถึงโครงสร้างโมเดลแบบพิเศษ')
    r.h2('3.2 Sobel Edge Loss')
    r.p('Sobel คือ kernel ขนาด 3×3 ที่หาความเปลี่ยนแปลงของความสว่างในแนวนอน (Gx) และแนวตั้ง (Gy) '
        'ขอบตัวอักษรจะมีค่าสูง ในโค้ด kernel ถูกหารด้วย 8 และเทียบ Gx กับ Gy แยกกันโดยตรง ไม่ได้คำนวณ magnitude')
    r.table('Sobel kernel ที่ใช้ (ก่อนหารด้วย 8)', ['', 'Gx', '', '', 'Gy', ''], [
        ['−1', '0', '1', '−1', '−2', '−1'],
        ['−2', '0', '2', '0', '0', '0'],
        ['−1', '0', '1', '1', '2', '1'],
    ], size=BODY, widths=[0.6] * 6)
    r.figure(FIG / 'sobel_explainer.png', 'ตัวอย่างผลของ Sobel Gx และ Gy บนภาพเอกสาร ขอบตัวอักษรมีค่าสูง')
    r.p('Loss รวมที่ใช้เทรนคือ', indent=True)
    r.image(eq_png, 5.8)
    r.p('โดย ŷ คือภาพที่โมเดลทำนาย y คือ ground truth และ λ คือน้ำหนักของ edge loss ค่าหลักคือ 0.1 '
        'และทดลอง 0 กับ 0.5 เป็น ablation (การทดลองปรับหรือตัดส่วนประกอบออกเพื่อดูผล)')
    r.h2('3.3 Training Augmentation')
    r.p('ใช้ augmentation แบบ global (ทั้งภาพ) เท่านั้น และไม่ใช้ Augraphy ตอนเทรน '
        'นอกจากนี้ทุก job ใช้ cosine learning rate, AMP (mixed precision) และ gradient clipping')
    r.table('Augmentation ที่ใช้ตอนเทรน', ['Augmentation', 'ความน่าจะเป็น', 'ช่วงค่า', 'ใช้กับ'], [
        ['Horizontal flip', '0.5', '-', 'noisy และ clean'],
        ['Photometric (brightness, contrast, gamma)', '0.8', '±0.15, ×0.7–1.3, γ 0.7–1.5', 'noisy เท่านั้น'],
        ['Gaussian blur', '0.3', 'σ 0.3–1.2', 'noisy เท่านั้น'],
        ['JPEG compression', '0.3', 'quality 30–95', 'noisy เท่านั้น'],
        ['Gaussian noise', '0.3', 'σ 0–0.03', 'noisy เท่านั้น'],
    ], widths=[2.4, 1.0, 1.7, 1.2])
    r.h2('3.4 เทคนิคหลัง Midterm')
    r.table('เทคนิคหลัง midterm และ job ที่ใช้', ['สัปดาห์', 'เทคนิค', 'Job'], [
        ['W9', 'Fine-tune ข้าม dataset (Kaggle → DIBCO 2018+2019A) แบบ full และ freeze encoder',
         'ft_full, ft_freeze_enc, nafnet_ft_dibco_full'],
        ['W9', 'Pretrained model ภายนอก: NAFNet (SIDD) ทดสอบ zero-shot แล้ว fine-tune บน Kaggle',
         'nafnet_zeroshot, nafnet_ft'],
        ['W10', 'Knowledge Distillation: teacher unet_edge_aug → student ShallowCNN', 'kd_cnn'],
        ['W12', 'LoRA บน conv rank 4 (+31,652 params) และ rank 8 (+63,304 params)',
         'ft_lora4, ft_lora8, nafnet_ft_dibco_lora8'],
    ], widths=[0.7, 3.4, 2.2])
    r.p('**LoRA (Low-Rank Adaptation)** คือการ fine-tune โดยแช่แข็งน้ำหนักเดิม แล้วเทรนเฉพาะ adapter '
        'ขนาดเล็กที่บวกเข้าไป ส่วน **Knowledge Distillation** คือการให้โมเดลเล็ก (student) '
        'เรียนจาก output ของโมเดลใหญ่ (teacher)')
    r.p('**LoRA กับโมเดลภาพ:** LoRA ถูกเสนอครั้งแรกสำหรับ language model แต่หลักการใช้ได้กับทุกชั้นที่เป็นการคูณน้ำหนัก '
        'งานนี้ใช้กับ Conv2d (src/lora.py) โดยแช่แข็ง conv เดิม แล้วบวก adapter 2 ชั้น คือ conv A ขนาด kernel เท่าเดิม '
        'ที่ลด channel เหลือ r และ conv B ขนาด 1×1 ที่ขยายกลับ output = conv(x) + (α/r)·B(A(x)) '
        'B เริ่มเป็นศูนย์ โมเดลตอนเริ่ม fine-tune จึงเท่ากับโมเดลเดิมทุกประการ และหลังเทรนรวม B·A กลับเข้า conv เดิมได้ '
        '(merged_conv) ซึ่งในหลักการจะไม่เพิ่มเวลาตอนใช้งาน (ยังไม่ได้วัด latency หลัง merge ดูหัวข้อ 5.5)')
    r.p('**ทำไม ft_lora8 ได้ FM เท่ากับ ft_full:** ตัวเลขไม่ได้เท่ากันทุกค่า FM 84.38 เทียบ 84.36 (p = 1.0) '
        'แต่ PSNR 15.54 เทียบ 15.71 dB และ DRD 6.86 เทียบ 6.64 ส่วน rank 4 ได้ต่ำกว่า (FM 82.86) '
        'และบน NAFNet LoRA r=8 ได้ต่ำกว่า full fine-tune ชัดเจน (85.28 เทียบ 87.98) '
        'ข้อสันนิษฐานคือการปรับ SmallUNet จาก Kaggle ไป DIBCO ต้องการการเปลี่ยนน้ำหนักไม่มาก '
        'และข้อมูล fine-tune มีน้อย (DIBCO 2018 + 2019A) full fine-tune จึงไม่ได้เปรียบจากจำนวน params ที่มากกว่า '
        'ข้อนี้ยังไม่ได้พิสูจน์ด้วยการทดลองเพิ่ม เช่น วัด rank ของ ΔW จริง')
    r.h2('3.5 โมเดลที่ใช้เปรียบเทียบ')
    r.p('**Proposed method ของงานนี้** คือ SmallUNet + Sobel edge loss + augmentation (job unet_edge_aug) '
        'ส่วน Knowledge Distillation ไม่ได้ใช้กับ SmallUNet แต่ใช้ SmallUNet ตัวนี้เป็น teacher '
        'เพื่อสอน ShallowCNN (job kd_cnn และ xd_*_kd) ให้ได้โมเดลเล็กที่เร็วที่สุด '
        'โมเดลอื่นในตารางเป็น baseline (ShallowCNN, DnCNN-lite) หรือโมเดลใหญ่ที่ใช้เปรียบเทียบ (NAFNet)')
    r.table('โมเดลทั้งหมดในงานนี้', ['โมเดล', 'Params', 'โครงสร้าง', 'บทบาท'], [
        ['SmallUNet', '487,009', 'U-Net 3 ชั้น + skip (concat), residual', 'โมเดลหลัก'],
        ['NAFNet', '29,159,715', 'U-Net 4 ชั้นของ NAFBlock + skip (add), pretrained บน SIDD, '
                                 'แปลงภาพขาวดำเป็น 3 channel', 'โมเดลใหญ่เปรียบเทียบ'],
        ['DnCNN-lite', '47,009', 'conv 3×3 7 ชั้น, BatchNorm, residual (no-BN 46,849)', 'baseline จาก midterm'],
        ['ShallowCNN', '7,265', 'conv 3×3 5 ชั้น, Sigmoid, ทำนายภาพสะอาดตรง', 'baseline และ KD student'],
    ], widths=[1.1, 1.0, 2.8, 1.4])
    r.figure(FIG / 'nafnet_diagram.png', 'โครงสร้าง NAFNet ตัวเลขในวงเล็บคือจำนวน NAFBlock')
    r.figure(FIG / 'dncnn_diagram.png', 'โครงสร้าง DnCNN-lite', width=5.2)
    r.figure(FIG / 'shallowcnn_diagram.png', 'โครงสร้าง ShallowCNN', width=5.2)
    r.p('นอกจากนี้ยังเทียบกับวิธี classical คือ median filter 3×3, Sauvola และ Otsu thresholding')

    r.h1('4. การตั้งค่าการทดลอง (Experimental Setup)')
    r.p('ทุก job เทรนและประเมินบน Google Colab Tesla T4 (16 GB) ด้วย PyTorch และใช้ seed 42 '
        'ยกเว้น unet_edge_aug_s43 และ unet_edge_aug_s44 ที่ใช้ดูความแปรปรวนจาก seed')
    r.table('จำนวน epoch และ sample ต่อ epoch', ['กลุ่ม job', 'Epochs', 'Samples ต่อ epoch'], [
        ['UNet / DnCNN / KD (บน Kaggle)', '30', '2,000'],
        ['nafnet_ft (บน Kaggle)', '20', '1,000'],
        ['Fine-tune U-Net → DIBCO', '20', '1,000'],
        ['Fine-tune NAFNet → DIBCO', '10', '1,000'],
    ], size=BODY, widths=[3.0, 1.0, 1.8])
    r.p('ก่อนเทรนเต็ม เรารัน pilot (ทดลองรันสั้น ๆ) เพื่อตรวจว่า pipeline ทำงานและวัดเวลาที่ใช้ '
        'ผล pilot ไม่นำมาใช้ในข้อสรุป')
    r.h2('4.1 การเปรียบเทียบที่ยุติธรรม')
    r.items([
        'ทุกโมเดลวัดบน test set ชุดเดียวกันทุกภาพ ด้วยโค้ดประเมินชุดเดียวกัน',
        'checkpoint ของ midterm (midterm_*) ถูกนำมาประเมินซ้ำ ไม่ได้คัดลอกตัวเลขเก่า',
        'ทดสอบนัยสำคัญด้วย paired Wilcoxon signed-rank test (two-sided) เทียบผลรายภาพของ 2 โมเดล',
        'วัดความเร็วบนภาพ 540×420, batch 1 ใช้ค่า median จาก 50 รอบ',
        'บน DIBCO ภาพที่ denoise แล้วจะถูก binarize ด้วย Otsu ก่อนวัด FM, pFM และ DRD',
    ])
    r.figure(FIG / 'curves.png', 'Validation PSNR ระหว่างเทรน ซ้ายคือ Kaggle ขวาคือ fine-tune บน DIBCO')

    r.h1('5. ผลการทดลอง (Results)')
    r.h2('5.1 In-domain (Kaggle test, 24 ภาพ)')
    r.p('ทุกโมเดล deep learning ได้ PSNR ใกล้เคียงกันที่ 31.2–32.2 dB ยกเว้น ShallowCNN '
        'ส่วนวิธี classical ได้ต่ำกว่าภาพ input')
    r.table('ผลบน Kaggle test', ['Model', 'PSNR (dB)', 'SSIM', 'Text PSNR (dB)'], [
        ['noisy input', '16.30', '0.849', '20.70'],
        ['classical_median3', '14.29', '0.704', '9.91'],
        ['classical_sauvola', '15.53', '0.866', '11.10'],
        ['classical_otsu', '15.21', '0.867', '11.10'],
        ['midterm_cnn_l1', '27.71', '0.981', '21.59'],
        ['kd_cnn', '28.84', '0.982', '23.42'],
        ['midterm_dncnn_lite', '31.67', '0.992', '25.53'],
        ['dncnn_nobn_aug', '31.94', '0.991', '26.32'],
        ['midterm_unet_l1', '31.51', '0.986', '25.40'],
        ['midterm_unet_edge', '31.84', '0.989', '25.44'],
        ['unet_edge_aug (λ = 0.1)', '31.19', '0.990', '25.95'],
        ['unet_edge_aug_s43', '31.37', '0.990', '25.56'],
        ['unet_edge_aug_s44', '31.64', '0.992', '26.06'],
        ['unet_aug_l0 (λ = 0)', '31.48', '0.991', '26.10'],
        ['unet_aug_l05 (λ = 0.5)', '32.15', '0.992', '26.37'],
        ['nafnet_zeroshot', '14.97', '0.781', '12.24'],
        ['nafnet_ft', '31.19', '0.990', '25.20'],
    ], widths=[2.4, 1.2, 1.0, 1.4])
    r.figure(FIG / 'slide_kaggle.png', 'PSNR บน Kaggle test', width=5.5)
    r.h2('5.2 Domain Shift (Augraphy, 24 ภาพ)')
    r.p('โมเดลที่เทรนด้วย augmentation ได้ 17.16–17.54 dB ส่วนโมเดล midterm ได้เพียง 14.53–15.30 dB')
    r.table('ผลบน Augraphy shift', ['Model', 'PSNR (dB)', 'SSIM', 'Text PSNR (dB)'], [
        ['noisy input', '9.93', '0.614', '9.09'],
        ['classical_sauvola', '14.55', '0.772', '8.08'],
        ['midterm_cnn_l1', '15.30', '0.728', '6.95'],
        ['kd_cnn', '16.55', '0.800', '9.05'],
        ['midterm_dncnn_lite', '14.85', '0.711', '6.46'],
        ['dncnn_nobn_aug', '16.72', '0.810', '9.35'],
        ['midterm_unet_edge', '14.53', '0.698', '5.93'],
        ['unet_edge_aug', '17.48', '0.850', '10.89'],
        ['unet_aug_l0', '17.54', '0.850', '10.96'],
        ['unet_aug_l05', '17.16', '0.840', '10.30'],
        ['nafnet_zeroshot', '9.67', '0.537', '7.83'],
        ['nafnet_ft', '17.16', '0.845', '10.73'],
    ], widths=[2.4, 1.2, 1.0, 1.4])
    r.figure(FIG / 'slide_augraphy.png', 'PSNR บน Augraphy shift', width=5.5)
    r.h2('5.3 Cross-dataset (DIBCO 2017, 20 ภาพ)')
    r.p('โมเดลที่ไม่ได้ fine-tune ได้ FM ต่ำกว่า Otsu บนภาพดิบ (61.9 เทียบ 78.0) '
        'หลัง fine-tune NAFNet ได้สูงสุดที่ 88.0')
    r.table('ผลบน DIBCO 2017 (DRD ยิ่งต่ำยิ่งดี)',
            ['Model', 'FM all', 'FM printed', 'FM hand-written', 'pFM', 'PSNR (dB)', 'DRD'], [
        ['Otsu on raw input', '78.04', '83.93', '72.16', '80.52', '13.96', '14.90'],
        ['classical_sauvola', '77.64', '79.55', '75.73', '87.16', '14.73', '7.28'],
        ['midterm_unet_edge', '62.37', '70.87', '53.86', '72.29', '13.32', '10.21'],
        ['unet_edge_aug', '61.91', '63.59', '60.22', '74.19', '12.80', '12.23'],
        ['nafnet_zeroshot', '76.50', '82.45', '70.56', '78.54', '13.46', '16.42'],
        ['nafnet_ft', '66.53', '70.06', '63.00', '75.55', '13.02', '11.77'],
        ['ft_full', '84.36', '82.47', '86.25', '87.30', '15.71', '6.64'],
        ['ft_freeze_enc', '77.53', '82.66', '72.40', '82.96', '15.23', '6.94'],
        ['ft_lora4', '82.86', '80.78', '84.93', '86.02', '15.46', '7.11'],
        ['ft_lora8', '84.38', '82.56', '86.21', '86.70', '15.54', '6.86'],
        ['nafnet_ft_dibco_full', '87.98', '88.09', '87.86', '90.71', '16.89', '5.05'],
        ['nafnet_ft_dibco_lora8', '85.28', '86.59', '83.98', '90.24', '16.04', '5.70'],
    ], widths=[1.8, 0.65, 0.75, 0.85, 0.65, 0.75, 0.6])
    r.figure(FIG / 'slide_dibco.png', 'F-measure บน DIBCO 2017', width=5.5)
    r.h2('5.4 Fine-tune และการลืมงานเดิม (Forgetting)')
    r.p('หลัง fine-tune บน DIBCO ทุกวิธีทำให้ PSNR บน Kaggle ตกจาก 31.19 เหลือ 20.03–22.98 dB')
    r.table('ผลก่อนและหลัง fine-tune บน DIBCO', ['Model', 'DIBCO FM', 'DIBCO DRD', 'Kaggle PSNR (dB)'], [
        ['unet_edge_aug (ก่อน fine-tune)', '61.91', '12.23', '31.19'],
        ['ft_full', '84.36', '6.64', '22.51'],
        ['ft_freeze_enc', '77.53', '6.94', '21.53'],
        ['ft_lora4', '82.86', '7.11', '21.20'],
        ['ft_lora8', '84.38', '6.86', '20.03'],
        ['nafnet_ft (ก่อน fine-tune)', '66.53', '11.77', '31.19'],
        ['nafnet_ft_dibco_full', '87.98', '5.05', '21.43'],
        ['nafnet_ft_dibco_lora8', '85.28', '5.70', '22.98'],
    ], widths=[2.6, 1.1, 1.1, 1.5])
    r.h2('5.5 ประสิทธิภาพ (Efficiency)')
    r.p('วัดบนภาพ 540×420, batch 1, Tesla T4 พบว่า U-Net เร็วกว่า NAFNet ราว 16 เท่า (8.8 เทียบ 140.6 ms) '
        'และใช้ VRAM น้อยกว่าราว 3.5 เท่า')
    r.table('ความเร็วและหน่วยความจำ', ['Model', 'Params', 'Median latency (ms)', 'Peak VRAM (MB)'], [
        ['kd_cnn', '7,265', '4.0', '31'],
        ['dncnn_nobn_aug', '46,849', '12.0', '57'],
        ['unet_edge_aug', '487,009', '8.8', '107'],
        ['ft_lora4', '518,661', '17.6', '133'],
        ['ft_lora8', '550,313', '18.1', '133'],
        ['nafnet_ft', '29,159,715', '140.6', '375'],
        ['nafnet_ft_dibco_lora8', '30,131,731', '212.5', '491'],
    ], widths=[2.2, 1.3, 1.5, 1.3])
    r.p('LoRA ช้ากว่าเพราะยังไม่ได้ merge น้ำหนัก adapter กลับเข้า conv เดิม ถ้า merge แล้วควรเร็วเท่า U-Net ปกติ '
        'แต่ยังไม่ได้วัดจริง')
    r.h2('5.6 การทดสอบนัยสำคัญ (Paired Wilcoxon Signed-rank, Two-sided)')
    r.table('ผลการทดสอบนัยสำคัญ', ['A', 'B', 'Test set', 'Metric', 'n', 'median(A − B)', 'p'], [
        ['unet_edge_aug', 'midterm_unet_edge', 'Kaggle', 'PSNR', '24', '−0.39', '0.121'],
        ['unet_aug_l05', 'midterm_unet_edge', 'Kaggle', 'PSNR', '24', '0.46', '0.375'],
        ['dncnn_nobn_aug', 'midterm_dncnn_lite', 'Kaggle', 'PSNR', '24', '−0.02', '0.406'],
        ['kd_cnn', 'midterm_cnn_l1', 'Kaggle', 'PSNR', '24', '1.15', '< 0.001'],
        ['nafnet_ft', 'unet_edge_aug', 'Kaggle', 'PSNR', '24', '0.18', '1.000'],
        ['unet_edge_aug', 'midterm_unet_edge', 'Augraphy', 'PSNR', '24', '2.74', '< 0.001'],
        ['ft_full', 'unet_edge_aug', 'DIBCO 2017', 'FM', '20', '17.06', '< 0.001'],
        ['ft_lora8', 'ft_full', 'DIBCO 2017', 'FM', '20', '0.16', '1.000'],
        ['nafnet_ft_dibco_full', 'ft_full', 'DIBCO 2017', 'FM', '20', '3.80', '0.006'],
    ], widths=[1.5, 1.5, 0.9, 0.6, 0.4, 0.9, 0.6])
    r.h2('5.7 Cross-domain Benchmark (Zero-shot 3 × 3)')
    r.p('ทดลองเพิ่มตามคำแนะนำของอาจารย์ เทรนแต่ละโมเดลบน 1 domain (Kaggle, Augraphy หรือ DIBCO) แล้วทดสอบทันที '
        'โดยไม่ fine-tune บน test set ทั้ง 3 ชุด (Kaggle test, Augraphy shift, DIBCO 2017) ทุก job ใช้ recipe เดียวกัน '
        '(edge λ = 0.1, augmentation, cosine LR, AMP) และ budget เท่ากับ job บน Kaggle ส่วน KD-CNN ใช้ teacher '
        'เป็น SmallUNet ที่เทรนบน domain เดียวกับ student เครื่องหมาย * คือ in-domain')
    header = ['Model', 'Train on'] + [f'{DOMAIN_LABEL[t]} test' for t in DOMAINS]
    r.table('PSNR (dB) ของ cross-domain benchmark', header, cross_domain_rows('psnr'),
            widths=[1.7, 1.0, 1.2, 1.2, 1.2])
    r.table('FM (%) ของ cross-domain benchmark (binarize ด้วย Otsu)', header, cross_domain_rows('fm'),
            widths=[1.7, 1.0, 1.2, 1.2, 1.2])
    r.p('ตารางด้านล่างสรุปทั้ง 6 task ในตารางเดียว (K → A, K → D, A → K, A → D, D → K, D → A) '
        'โดยเฉลี่ยเฉพาะช่องที่ source ต่างจาก test set ช่องที่ collapse นับด้วยค่าจริง (PSNR 10.32–11.07 dB, FM 0) '
        'เพราะเป็นผลที่เกิดขึ้นจริงเมื่อนำโมเดลไปใช้')
    r.table('ค่าเฉลี่ยของ 6 transfer task แบบ zero-shot เรียงตาม PSNR', ['อันดับ', 'Model', 'PSNR เฉลี่ย (dB)',
            'FM เฉลี่ย (%)'], transfer_mean_rows(), widths=[0.7, 3.0, 1.3, 1.3])
    r.items([
        '**ทดสอบบน Kaggle:** โมเดลที่เทรนจาก domain อื่นเสีย PSNR มาก (-7.1 ถึง -13.7 dB) '
        'NAFNet เสียน้อยที่สุด (Augraphy → Kaggle 24.10 dB, DIBCO → Kaggle 22.51 dB)',
        '**ทดสอบบน DIBCO:** โมเดลที่เทรนบน Augraphy ได้ FM 67.7–71.4 สูงกว่าโมเดลที่เทรนบน Kaggle (36.0–66.5) '
        'แต่ยังต่ำกว่าโมเดลที่เทรนบน DIBCO เอง (SmallUNet 85.1, NAFNet 86.6)',
        '**DIBCO → Kaggle ใน FM:** SmallUNet และ NAFNet ได้ 89.4 และ 88.9 สูงกว่า in-domain เล็กน้อย '
        'แต่ PSNR ต่ำกว่ามาก แปลว่าตัวอักษรยังอยู่ครบ แต่พื้นหลังยังไม่สะอาด',
        '**ภาพรวม 6 task:** NAFNet ได้ PSNR เฉลี่ยสูงสุด (17.57 dB) และ SmallUNet ตามมาใกล้กัน (16.95 dB) '
        'แต่ใน FM เฉลี่ย SmallUNet สูงกว่าเล็กน้อย (76.98 เทียบ 76.63) ทั้งที่มี params น้อยกว่า 60 เท่า '
        'ShallowCNN และ KD-CNN อยู่ท้ายเพราะแถวที่เทรนบน DIBCO collapse',
        '**ShallowCNN และ KD-CNN บน DIBCO collapse:** ทั้งสองแถวออกภาพขาวทั้งหน้า '
        '(val PSNR คงที่ 10.01 dB ตั้งแต่ epoch แรก, FM = 0 ทุก test set) จึงไม่แสดงเป็นตัวเลข ดูการวิเคราะห์ในหัวข้อ 6',
    ])

    r.h2('5.8 ผลเชิงคุณภาพ')
    r.items([
        '**Kaggle test:** ทุกโมเดลลบคราบพื้นหลังได้ใกล้เคียงกัน nafnet_ft ให้ตัวอักษรหนาและเข้มกว่า ground truth',
        '**Augraphy shift:** midterm U-Net ทำตัวอักษรขาดและจาง ส่วน unet_edge_aug คงเส้นได้คม '
        'ทุกโมเดลยังคงบรรทัดที่บิดโค้งจาก Folding ไว้ เพราะ denoising ไม่ได้แก้รูปทรงเรขาคณิต',
        '**DIBCO 2017:** ถ้าไม่ fine-tune หมึกจางจะถูกลบทิ้ง ft_full กู้ตัวอักษรกลับมาได้แต่เส้นหนา '
        'ส่วน nafnet_ft_dibco_full ให้เส้นบางใกล้ ground truth กว่า แต่บางภาพมีตัวอักษรขาดหาย',
    ])
    r.figure(FIG / 'qual_augraphy_shift.png', 'ผลเชิงคุณภาพบน Augraphy shift')
    r.figure(FIG / 'qual_dibco2017.png', 'ผลเชิงคุณภาพบน DIBCO 2017')
    xd_fig = FIG / 'cross_domain' / 'xd_qual.png'
    if xd_fig.exists():
        r.figure(xd_fig, 'ผลเชิงคุณภาพของ 6 transfer task แบบ zero-shot (ภาพขยายบริเวณที่มีตัวอักษรมากที่สุด) '
                         'แต่ละแถวคือ source → test set และแต่ละคอลัมน์คือโมเดล')
    else:
        print('skip figure', xd_fig, '(run tools/make_figures.py --xd-only after downloading xd_* checkpoints)')

    r.h1('6. อภิปรายผล (Discussion)')
    r.items([
        '**Augmentation ช่วยเรื่อง domain shift ไม่ใช่ in-domain:** บน Kaggle ผลไม่ต่างจาก midterm '
        'อย่างมีนัยสำคัญ (p = 0.121 และ 0.375) แต่บน Augraphy ดีขึ้น median +2.74 dB (p < 0.001) '
        'และ Text PSNR เพิ่มจาก 5.93 เป็น 10.89 dB ผลนี้คงที่ทั้ง 3 seeds (17.41–17.48 dB)',
        '**ความแปรปรวนจาก seed ใหญ่พอ ๆ กับความต่างระหว่างโมเดล:** unet_edge_aug 3 seeds ได้ '
        '31.19 / 31.37 / 31.64 dB ช่วงกว้าง 0.45 dB การเทียบ λ = 0, 0.1, 0.5 บน Kaggle จึงยังสรุปไม่ได้',
        '**Knowledge Distillation ได้ผลชัดกับโมเดลเล็ก:** kd_cnn ดีขึ้น +1.15 dB (p < 0.001) '
        'โดยขนาดเท่าเดิม 7,265 params และใช้เวลา 4.0 ms',
        '**โมเดลที่เรียนจาก Kaggle ลบหมึกจางของเอกสารเก่าทิ้ง:** บน DIBCO ได้ FM 61.9 ต่ำกว่า Otsu '
        'บนภาพดิบ (78.0) แสดงว่า augmentation แบบ global ไม่พอสำหรับเอกสารคนละแหล่ง',
        '**Fine-tune แก้ปัญหานี้ได้ และ LoRA ได้ผลเท่ากัน:** ft_full เพิ่ม FM เป็น 84.4 (+17.1 median, '
        'p < 0.001) และ DRD ลดจาก 12.23 เป็น 6.64 ส่วน ft_lora8 ได้ 84.4 เท่ากัน (p = 1.0) '
        'โดยเทรนเพิ่มเพียง 63,304 params (ราว 13% ของโมเดล) ส่วน freeze encoder ได้ต่ำสุดที่ 77.5',
        '**NAFNet ดีที่สุดบน DIBCO แต่ไม่คุ้มบน Kaggle:** บน DIBCO ได้ FM 88.0 ดีกว่า ft_full +3.8 '
        '(p = 0.006) แต่บน Kaggle ได้ 31.19 dB เท่า U-Net (p = 1.0) ทั้งที่มี params มากกว่า 60 เท่า '
        'และช้ากว่า 16 เท่า ส่วน zero-shot ได้เพียง 14.97 dB เพราะเดิมเทรนกับ noise ของกล้อง (SIDD)',
        '**Fine-tune ทำให้ลืมงานเดิม:** Kaggle PSNR ตกเหลือ 20.0–23.0 dB ทุกวิธี LoRA ไม่ช่วย '
        'เพราะ adapter ทำงานตลอดเวลา การใช้งานจริงต้องสลับ adapter ตาม domain หรือเทรนร่วมกันทั้งสอง domain',
        '**ไม่มี domain ไหนที่เทรนแล้วใช้ได้กับทุก domain:** ในตาราง 3 × 3 ช่อง in-domain ดีที่สุดเกือบทุกคอลัมน์ '
        'Augraphy เป็น source ที่ transfer ไปยัง DIBCO ได้ดีกว่า Kaggle เพราะมีหมึกซึมและแสงไม่สม่ำเสมอคล้ายเอกสารจริง',
        '**โมเดลเล็กมาก collapse บนข้อมูลที่ไม่สมดุล:** ground truth ของ DIBCO เป็นพื้นขาว 94% '
        'ShallowCNN (7,265 params, receptive field ราว 11 px) จึงเรียนรู้ที่จะตอบขาวทั้งภาพภายในราว 30 step '
        'จากนั้น sigmoid ที่ output saturate และ gradient ใกล้ 0 ทำให้ไม่หลุดออกมาอีก '
        'เราทดลองซ้ำในเครื่องได้ และลองลด learning rate (3e-4, 1e-4) กับเพิ่ม residual จาก input แล้วยัง collapse เหมือนเดิม '
        'KD ก็ไม่ช่วย เพราะ student ติดที่จุดเดียวกันก่อนจะได้ประโยชน์จาก teacher '
        'ส่วน SmallUNet, DnCNN-lite และ NAFNet ที่ใหญ่กว่าไม่มีปัญหานี้',
    ], numbered=True)
    r.p('**Trade-off:** ถ้าต้องการความเร็วและทนต่อ noise แบบใหม่ unet_edge_aug คุ้มที่สุด (8.8 ms) '
        'ถ้าเน้นความแม่นบนเอกสารเก่าและรับเวลา 140–155 ms ได้ ใช้ NAFNet ที่ fine-tune แล้ว '
        'ถ้าอุปกรณ์จำกัดมาก kd_cnn เร็วที่สุด (4.0 ms) และใช้ VRAM เพียง 31 MB')

    r.h1('7. ข้อจำกัดและสรุป (Limitations and Conclusion)')
    r.h2('7.1 ข้อจำกัด')
    r.items([
        '**Seed:** มีหลาย seed เฉพาะ unet_edge_aug (3 seeds) โมเดลอื่นใช้ seed เดียว '
        'การจัดอันดับโมเดลที่ต่างกันไม่เกินราว 0.5 dB จึงไม่น่าเชื่อถือ',
        '**ภาพ test ไม่เป็นอิสระต่อกัน:** Kaggle test 24 ภาพมาจาก 6 หน้าต้นฉบับ ค่า p ของ Wilcoxon จึงอาจดูดีเกินจริง',
        '**ขนาด DIBCO:** test มี 20 ภาพ (printed 10, handwritten 10) จึงมีขนาดเล็ก',
        '**pFM / DRD:** เป็น implementation ที่เขียนเอง ยังไม่ได้เทียบกับ evaluation tool ทางการของ DIBCO',
        '**Augraphy Folding:** ทำให้ตัวอักษรเลื่อนตำแหน่งเล็กน้อย ค่า PSNR สัมบูรณ์บนชุดนี้จึงต่ำ '
        'ควรดูเป็นค่าเปรียบเทียบระหว่างโมเดล',
        '**ไม่มี OCR:** ยังไม่ได้วัดว่าการ denoise ช่วยให้อ่านข้อความได้ถูกต้องขึ้นจริงหรือไม่',
        '**Reproducibility:** การ resume ไม่ deterministic ทุกบิต เพราะ cuDNN และ AMP ทำให้ผลต่างกันเล็กน้อยได้',
        '**Cross-domain:** Augraphy train ใช้หน้าข้อความชุดเดียวกับ Kaggle (ต่างกันแค่ความเสียหาย) และ DIBCO มีเพียง 16 หน้าสำหรับเทรน '
        'ภาพ Augraphy ที่สร้างบน Colab ไม่ตรงกับในเครื่องทุก byte เพราะไลบรารีคนละเวอร์ชัน '
        'และไม่ได้แก้ปัญหา collapse ของ ShallowCNN เพื่อให้ทุกโมเดลใช้ recipe เดียวกัน',
        '**License ของ DIBCO:** ไม่ได้ระบุชัด จึงใช้เพื่อการศึกษาเท่านั้น',
    ])
    r.h2('7.2 สรุป')
    r.p('Augmentation แบบ global ทำให้ SmallUNet ทนต่อ noise แบบใหม่ได้ชัดเจน (+2.74 dB บน Augraphy) '
        'โดยไม่เสียผล in-domain อย่างมีนัยสำคัญ แต่ไม่พอสำหรับ dataset อื่นอย่าง DIBCO ซึ่งต้อง fine-tune เพิ่ม '
        'LoRA r=8 ให้ผลเท่า full fine-tune ด้วย params ที่เทรนน้อยกว่ามาก แต่ทุกวิธียังลืมงานเดิม')
    r.p('งานต่อไปที่ควรทำ')
    r.items([
        'เทรนร่วมกันทั้ง Kaggle และ DIBCO หรือสลับ LoRA adapter ตาม domain เพื่อลดการลืมงานเดิม',
        'เพิ่ม seed ให้ทุกโมเดลและวัดผลด้วย OCR',
        'Merge น้ำหนัก LoRA แล้ววัด latency ใหม่',
    ])

    r.h1('เอกสารอ้างอิง (References)')
    r.items([
        'O. Ronneberger, P. Fischer, T. Brox. "U-Net: Convolutional Networks for Biomedical Image '
        'Segmentation." MICCAI, 2015.',
        'K. Zhang, W. Zuo, Y. Chen, D. Meng, L. Zhang. "Beyond a Gaussian Denoiser: Residual Learning of '
        'Deep CNN for Image Denoising." IEEE TIP, 2017.',
        'L. Chen, X. Chu, X. Zhang, J. Sun. "Simple Baselines for Image Restoration." ECCV, 2022. '
        'https://github.com/megvii-research/NAFNet',
        'A. Abdelhamed, S. Lin, M. S. Brown. "A High-Quality Denoising Dataset for Smartphone Cameras." CVPR, 2018.',
        'E. J. Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR, 2022.',
        'G. Hinton, O. Vinyals, J. Dean. "Distilling the Knowledge in a Neural Network." '
        'NIPS Deep Learning Workshop, 2015.',
        'Augraphy: an augmentation library for document images.',
        'Kaggle. "Denoising Dirty Documents" competition dataset.',
        'I. Pratikakis, K. Zagoris, G. Barlas, B. Gatos. "ICDAR 2017 Competition on Document Image '
        'Binarization (DIBCO 2017)." ICDAR, 2017.',
        'N. Otsu. "A Threshold Selection Method from Gray-Level Histograms." IEEE Trans. SMC, 1979.',
        'J. Sauvola, M. Pietikäinen. "Adaptive Document Image Binarization." Pattern Recognition, 2000.',
    ], numbered=True)


def add_page_numbers(section):
    section.footer.is_linked_to_previous = False
    par = section.footer.paragraphs[0]
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _field(par, 'PAGE')
    pg = OxmlElement('w:pgNumType')
    pg.set(qn('w:start'), '1')
    section._sectPr.append(pg)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    eq_png = OUT / 'sobel_loss_equation.png'
    equation_png(eq_png)
    r = Report()
    cover(r)
    r.doc.add_section(WD_SECTION.NEW_PAGE)
    contents(r)
    r.doc.add_section(WD_SECTION.NEW_PAGE)
    add_page_numbers(r.doc.sections[2])
    body(r, eq_png)
    path = OUT / 'AIE464_Final_Technical_Report.docx'
    r.doc.save(path)
    print('wrote', path, f'({r.n_fig} figures, {r.n_tab} tables)')


if __name__ == '__main__':
    main()
