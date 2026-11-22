"""Turn the executed Colab runner into the submission notebook.

    python3 tools/annotate_colab_notebook.py notebooks/final_colab_executed.ipynb notebooks/final_colab.ipynb

Keeps every code cell and its real Colab output untouched, replaces the title cell, puts a Thai
explanation before each code cell and appends a report section (tables, figures, conclusions)
that runs on its own in a fresh Colab session.
"""
import json
import sys
from pathlib import Path

TITLE = r"""# AIE464 Final Project: Text-Preserving Image Denoising

**Bangkok University | AIE464** | GPU: Google Colab Tesla T4 | โค้ดที่ใช้เทรนและประเมินผล: commit `75664d9`

Notebook นี้คือไฟล์ที่ใช้เทรนและประเมินผลจริงทั้งหมด output ของ cell 1–10 เป็นผลจากการรันจริงบน Colab T4 (ไม่ได้แก้ไข) ส่วนหัวข้อ **"ผลลัพธ์และการเปรียบเทียบ"** ท้ายไฟล์สร้างตาราง กราฟ และภาพเปรียบเทียบจากผลเหล่านั้น

> **วิธีรัน:** ห้ามกด Run all เพราะ cell 6 คือการเทรน 16 jobs (หลายชั่วโมง) ถ้าต้องการดูผลใหม่ ให้รันเฉพาะหัวข้อผลลัพธ์ (cell 11 เป็นต้นไป) ซึ่งตั้งค่าตัวเองได้โดยไม่ต้องรัน cell ก่อนหน้า

---

## 1. เป้าหมาย

- ปรับปรุงโมเดล denoise เอกสารจาก midterm ด้วยเทคนิคหลัง midterm (W9, W10, W12) และเทียบกับโมเดลเดิมแบบ **same circumstance** คือใช้ test set ชุดเดียวกัน วิธีวัดผลเดียวกัน และ GPU เดียวกัน
- ทดสอบความทนทานเมื่อข้อมูลเปลี่ยน 2 แบบ:
  1. **Augraphy shift:** ภาพ test ของ Kaggle ที่ใส่ความเสียหายแบบใหม่ที่โมเดลไม่เคยเห็น
  2. **DIBCO 2017:** dataset อื่น เป็นเอกสารเก่าจริงทั้งแบบพิมพ์และลายมือ
- ขนาด input เท่ากับ output เสมอ ($H \times W \rightarrow H \times W$) ไม่ใช่งาน super-resolution

## 2. ข้อมูลและที่มา

| ชุด | ที่มา | ใช้ทำอะไร | จำนวน | วิธีวัด |
|---|---|---|---|---|
| Kaggle Denoising Dirty Documents | [kaggle.com/c/denoising-dirty-documents](https://www.kaggle.com/c/denoising-dirty-documents) | train / val / **test (in-domain)** | test 24 ภาพจาก 6 หน้าต้นฉบับ แบ่งระดับหน้าเพื่อกัน data leakage | PSNR, SSIM, text PSNR |
| Augraphy shift | สร้างจาก Kaggle test ด้วยไลบรารี [Augraphy](https://github.com/sparkfish/augraphy) แบบ deterministic | test เท่านั้น | 24 ภาพ | เหมือน Kaggle |
| DIBCO 2017 | [Document Image Binarization Contest 2017](https://vc.ee.duth.gr/dibco2017/) | fine-tune + **test (cross-dataset)** | test 20 ภาพ (printed 10, handwritten 10) | binarize ด้วย Otsu แล้ววัด FM, pFM, PSNR, DRD |

- **Data leakage:** แบ่ง train/val/test ระดับหน้าเอกสารก่อนตัด patch ภาพจากหน้าเดียวกันจึงไม่อยู่คนละชุด
- **text PSNR:** PSNR เฉพาะบริเวณตัวอักษร
- **FM (F-measure):** พิกเซลหมึกที่ทำนายตรงกับ GT แค่ไหน ยิ่งสูงยิ่งดี
- **pFM:** F-measure ที่ดูโครงเส้นตัวอักษร (skeleton)
- **DRD:** ความผิดเพี้ยนที่สายตามนุษย์สังเกตได้ **ยิ่งต่ำยิ่งดี**

ไฟล์ข้อมูลทั้งหมดรวมไว้ใน `MyDrive/AIE464_runs/data/aie464_data.zip`

## 3. Edge-aware loss (Sobel)

Sobel คือ kernel ขนาด 3×3 ที่หาความเปลี่ยนแปลงของความสว่างในแนวนอน (Gx) และแนวตั้ง (Gy) ขอบตัวอักษรจะมีค่าสูง ใน loss เราเทียบ edge map ของภาพที่โมเดลทำนายกับของ GT เพื่อบังคับให้โมเดลรักษาขอบตัวอักษรไว้:

$$\text{loss} = L_1(\hat{y}, y) + \lambda \cdot L_1(\text{Sobel}(\hat{y}), \text{Sobel}(y))$$

ค่า λ ที่ทดลองคือ 0, 0.1 (ค่าหลัก) และ 0.5

## 4. โมเดล

- **Baseline จาก midterm:** ShallowCNN, DnCNN, U-Net (L1), U-Net + edge loss ใช้ checkpoint เดิมโดยไม่เทรนใหม่
- **SmallUNet:** encoder 3 ชั้น (max pool), decoder ใช้ nearest upsampling + skip connection, ทำนาย noise แล้วลบออกจาก input (residual) ขนาด 487,009 params
- **NAFNet-SIDD-width32:** pretrained model ภายนอก (29.2M params)

## 5. เทคนิคหลัง midterm ที่ใช้

| สัปดาห์ | เทคนิค | Job |
|---|---|---|
| W9 | Transfer learning / fine-tune ข้าม dataset (Kaggle → DIBCO): full FT และ freeze encoder | `ft_full`, `ft_freeze_enc`, `nafnet_ft_dibco_full` |
| W9 | Pretrained model ภายนอก: NAFNet (เทรนบน SIDD) ทดสอบ zero-shot แล้ว fine-tune บน Kaggle | `nafnet_zeroshot`, `nafnet_ft` |
| W10 | Knowledge Distillation: teacher `unet_edge_aug` → student ShallowCNN (7k params) | `kd_cnn` |
| W12 | Parameter-efficient fine-tune: LoRA บน conv (rank 4 / 8) | `ft_lora4`, `ft_lora8`, `nafnet_ft_dibco_lora8` |
| ทุก job | Augmentation (photometric, blur, JPEG, noise), cosine LR, AMP, gradient clipping | `*_aug` |

**Budget (ตั้งหลังรัน pilot):** UNet / DnCNN / KD 30 epochs × 2,000 samples, Fine-tune DIBCO 20 × 1,000, `nafnet_ft` 20 × 1,000, NAFNet → DIBCO 10 × 1,000 ทุก job ใช้ seed 42 ยกเว้น `unet_edge_aug_s43` และ `unet_edge_aug_s44`

---

## 6. กระบวนการเทรนและประเมินผล (cell 1–10, output จากการรันจริง)

โค้ดของ dataset, model, loss และ training loop อยู่ในโฟลเดอร์ `mini_project/src/` ของ repo notebook นี้ clone repo ตาม commit ที่ระบุ เพื่อให้ทุก session ของ Colab ใช้โค้ดชุดเดียวกันเป๊ะ
"""

BEFORE = {
    1: """### Cell 1: ตั้งค่า
รวมค่าทั้งหมดไว้ที่เดียว `GIT_SHA` ล็อก commit ของโค้ด, `BUDGETS` คือจำนวน epoch × samples ของแต่ละ job, `MAX_MINUTES` ให้หยุดพักอย่างปลอดภัยก่อน Colab ตัด session""",
    2: """### Cell 2: เตรียมโค้ด
mount Google Drive (ที่เก็บข้อมูล checkpoint และผล) แล้ว clone โค้ดตาม `GIT_SHA` output ยืนยัน commit ที่ใช้จริง""",
    3: """### Cell 3: ติดตั้งไลบรารีเสริม
`augraphy` ใช้สร้างความเสียหายเทียม และ `scikit-image` ใช้คำนวณ SSIM""",
    4: """### Cell 4: ข้อมูลและ unit test
แตก zip ข้อมูลจาก Drive รัน unit test ชุดเร็วเพื่อยืนยันว่าโค้ดทำงานถูกก่อนเทรน (output: test ผ่านทั้งหมด) และกำหนดฟังก์ชันโหลด train/val/test""",
    5: """### Cell 5: ลงทะเบียน jobs
ทุก job ถูกบันทึกใน `runs/jobs.json` บน Drive พร้อมสถานะ (pending / running / done) เพื่อให้รันต่อได้เมื่อ Colab หลุด output นี้คือสถานะตอนรันรอบสุดท้าย""",
    6: """### Cell 6: เทรน
เทรนทุก job ที่ค้างอยู่ต่อกัน บันทึก checkpoint ทุก epoch และหยุดพักก่อนหมดเวลา ถ้า Colab หลุดก็รันใหม่แล้วทำต่อจาก epoch ล่าสุด

output ด้านล่างเป็น log ของ session สุดท้าย (การเทรนทั้งหมดกระจายหลาย session) log ครบทุก epoch ของทั้ง 16 jobs อยู่ใน `runs/<job>/validation.json` และวาดเป็นกราฟในหัวข้อผลลัพธ์""",
    7: """### Cell 7: ประเมินผล
วัดทุกโมเดลบน test set ทั้ง 3 ชุดด้วย test set และวิธีวัดเดียวกัน (same circumstance) รวมโมเดล midterm, NAFNet zero-shot และวิธี classical บันทึกผลรายภาพไว้ใน `runs/eval/*.json`""",
    8: """### Cell 8: ความเร็ว
วัด latency บนภาพ 540×420, warm-up 10 รอบ วัดจริง 50 รอบ ใช้ค่า median พร้อม peak VRAM และจำนวน params""",
    9: """### Cell 9: ใช้กับโฟลเดอร์ภาพอื่น
ตั้ง `INPUT_DIR` ใน cell 1 เพื่อ denoise ทุกภาพในโฟลเดอร์ ถ้าตั้ง `GT_DIR` ด้วยจะคำนวณ metric ให้ (ไม่ได้ใช้ในรอบนี้ จึงไม่มี output)""",
    10: """### Cell 10: สรุปผล PSNR
รวมผลของทุกโมเดลไว้ใน `runs/summary.json` output แสดง PSNR บน test ทั้ง 3 ชุด""",
}

REPORT_INTRO = """---

## 7. ผลลัพธ์และการเปรียบเทียบ

ส่วนนี้รันแยกได้ใน session ใหม่ (ไม่ต้องรัน cell 1–10) โดยอ่านผลประเมินที่ cell 7–8 บันทึกไว้บน Drive แล้วสร้างตาราง กราฟ และภาพเปรียบเทียบ โค้ดสร้างตารางอยู่ใน `src/report.py` และโค้ดสร้างรูปอยู่ใน `tools/make_figures.py`"""

SETUP = """# 11. Report setup: runs on its own in a fresh session (do NOT rerun cells 1-10)
import os, subprocess, sys, json
from pathlib import Path
from google.colab import drive
drive.mount('/content/drive')
DRIVE_ROOT = '/content/drive/MyDrive/AIE464_runs'
REPO_URL = 'https://github.com/chayanon25/aie464-text-preserving-denoising.git'
RUNS = Path(DRIVE_ROOT) / 'runs'
REPO = Path('/content/report_repo')
if not REPO.exists():
    subprocess.run(['git', 'clone', '-q', REPO_URL, str(REPO)], check=True)
MINI = REPO / 'mini_project'
sys.path.insert(0, str(MINI))
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'augraphy', 'scikit-image'], check=True)
DATA_ROOT = Path('/content/data')
if not (DATA_ROOT / 'raw').exists():
    subprocess.run(['unzip', '-q', '-o', f'{DRIVE_ROOT}/data/aie464_data.zip', '-d', str(DATA_ROOT)], check=True)
os.environ['MIDTERM_CKPT_DIR'] = str(DATA_ROOT / 'checkpoints')
for sub in ('raw', 'external'):
    link = MINI / 'data' / sub
    link.parent.mkdir(exist_ok=True)
    if not link.exists():
        link.symlink_to(DATA_ROOT / sub)
sha = subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
print('report code sha:', sha, '| eval files:', len(list((RUNS / 'eval').glob('*.json'))))"""

TABLES_MD = """### 7.1 ตารางผล
ทุกตารางคำนวณจากผลรายภาพใน `runs/eval/*.json` ค่า p มาจาก paired Wilcoxon signed-rank test (two-sided) ซึ่งทดสอบว่าความต่างรายภาพระหว่าง 2 โมเดลมีนัยสำคัญหรือไม่"""

TABLES = """# 12. Result tables
from IPython.display import Markdown, display
from src.report import build_tables
TITLES = {'in_domain': 'In-domain (Kaggle test, 24 ภาพ)', 'augraphy_shift': 'Domain shift (Augraphy, 24 ภาพ)',
          'dibco2017': 'Cross-dataset (DIBCO 2017, 20 ภาพ)', 'finetune': 'Fine-tune และการลืมงานเดิม (forgetting)',
          'efficiency': 'Efficiency (540x420, batch 1, Tesla T4, median 50 รอบ)',
          'wilcoxon': 'ทดสอบนัยสำคัญ (paired Wilcoxon)'}
tables = build_tables(RUNS / 'eval', latency_path=RUNS / 'latency.json')
for key, table in tables.items():
    display(Markdown(f'#### {TITLES.get(key, key)}\\n\\n{table}'))"""

FIGS_MD = """### 7.2 ภาพประกอบและภาพเปรียบเทียบ
สร้างใหม่จาก checkpoint `best.pt` ของแต่ละ job บน Drive และ log การเทรน `validation.json` (seed 42)"""

FIGS = """# 13. Figures: data samples, Sobel, U-Net shapes, qualitative grids, zooms, training curves
from IPython.display import Image
FIG_DIR = Path('/content/figures')
_r = subprocess.run([sys.executable, str(MINI / 'tools' / 'make_figures.py'), '--runs', str(RUNS), '--out', str(FIG_DIR)],
                    capture_output=True, text=True)
print(_r.stdout[-2000:], _r.stderr[-2000:])
CAPTIONS = [
    ('data_samples', 'ตัวอย่างข้อมูล: Kaggle, Augraphy shift และ DIBCO 2017'),
    ('sobel_explainer', 'Sobel: kernel Gx/Gy และ edge map ที่ใช้ใน edge loss'),
    ('unet_diagram', 'SmallUNet: shape ทุกชั้นจากการรันจริงบนภาพ 256x256'),
    ('curves', 'Validation PSNR ระหว่างเทรน: ซ้าย Kaggle, ขวา fine-tune บน DIBCO'),
    ('qual_kaggle_test', 'Kaggle test: ทุกโมเดลลบคราบได้ใกล้เคียงกัน nafnet_ft ให้ตัวอักษรหนากว่า GT'),
    ('zoom_kaggle_test', 'Kaggle test (ขยาย)'),
    ('qual_augraphy_shift', 'Augraphy shift: unet_edge_aug คงตัวอักษรได้ดีกว่า midterm ทุกโมเดลยังคงบรรทัดที่บิดไว้ เพราะ denoising ไม่แก้รูปทรง'),
    ('zoom_augraphy_shift', 'Augraphy shift (ขยาย): midterm ทำตัวอักษรขาดและจาง'),
    ('qual_dibco2017', 'DIBCO 2017: ไม่ fine-tune หมึกจางหาย, ft_full กู้กลับมาได้, NAFNet FM สูงสุดแต่บางช่วงขาด'),
    ('zoom_dibco2017', 'DIBCO 2017 (ขยาย): ft_full เส้นหนาเป็นก้อน NAFNet เส้นบางใกล้ GT กว่า'),
]
for name, caption in CAPTIONS:
    path = FIG_DIR / f'{name}.png'
    if path.exists():
        display(Markdown(f'**{caption}**'))
        display(Image(filename=str(path)))"""

CONCLUSION = """## 8. ข้อสรุปที่มีตัวเลขรองรับ

1. **Augmentation ไม่ได้ทำให้ผล in-domain ดีขึ้นอย่างมีนัยสำคัญ แต่ทำให้ทนต่อ domain shift ได้ชัดเจน**
   - Kaggle: `unet_aug_l05` 32.15 dB เทียบ midterm 31.84 dB (p=0.375) ส่วน `unet_edge_aug` 31.19 dB (p=0.121)
   - Augraphy: `unet_edge_aug` 17.48 dB เทียบ midterm 14.53 dB (median +2.74 dB, p<0.001) text PSNR เพิ่มจาก 5.93 เป็น 10.89 dB และคงที่ทั้ง 3 seeds (17.41–17.48 dB)
2. **ความแปรปรวนจาก seed มีขนาดพอ ๆ กับความต่างระหว่างโมเดลบน Kaggle:** `unet_edge_aug` 3 seeds ได้ 31.19 / 31.37 / 31.64 dB (ช่วง 0.45 dB) ส่วนต่างที่เล็กกว่านี้ เช่นค่า λ จึงยังสรุปไม่ได้
3. **Knowledge Distillation (W10) ได้ผลชัดกับโมเดลเล็ก:** `kd_cnn` 28.84 dB เทียบ midterm CNN 27.71 dB (+1.15 dB median, p<0.001) ขนาดเท่าเดิม 7,265 params และใช้เวลา 4.0 ms
4. **โมเดลที่ไม่ได้ fine-tune ทำงานบน DIBCO แย่กว่า Otsu บนภาพดิบ:** `unet_edge_aug` FM 61.9 เทียบ Otsu 78.0 เพราะโมเดลที่เรียนจาก Kaggle ลบหมึกจาง ๆ ของเอกสารเก่าทิ้ง
5. **Fine-tune (W9) แก้ปัญหานี้ได้:** `ft_full` FM 84.4 จาก 61.9 (+17.1 median, p<0.001) DRD ลดจาก 12.2 เป็น 6.6
6. **LoRA r=8 (W12) ได้ผลเท่า full fine-tune:** FM 84.4 (p=1.0) โดยเทรนเพิ่มแค่ 63k params (ราว 13% ของโมเดล) ส่วน freeze encoder ได้ 77.5 ต่ำสุดในกลุ่ม
7. **NAFNet ดีที่สุดบน DIBCO หลัง fine-tune แต่ไม่คุ้มบน Kaggle:**
   - DIBCO: `nafnet_ft_dibco_full` FM 88.0, DRD 5.05 ดีกว่า `ft_full` (+3.8 FM median, p=0.006)
   - Kaggle: `nafnet_ft` 31.19 dB เท่ากับ U-Net (p=1.0) แต่ params มากกว่า 60 เท่าและช้ากว่า 16 เท่า (141 ms เทียบ 9 ms)
   - Zero-shot ได้แค่ 14.97 dB เพราะเดิมเทรนกับ noise ของกล้อง (SIDD) ไม่ใช่ noise ของเอกสาร
8. **Fine-tune บน DIBCO ทำให้ลืมงานเดิม (catastrophic forgetting):** Kaggle PSNR ตกจาก 31.2 เหลือ 20.0–23.0 dB ทุกวิธี LoRA ไม่ช่วยเพราะ adapter ทำงานตลอด การใช้งานจริงต้องสลับ adapter ตาม domain หรือเทรนร่วมทั้งสอง domain

## 9. ข้อจำกัด

- **โค้ดรันได้ไม่เท่ากับโมเดลดี:** pipeline รันครบบน T4 (16 jobs, eval 25 รายการ) แต่ข้อสรุปเรื่องคุณภาพยึดตามตัวเลขในหัวข้อ 7–8 เท่านั้น
- **Seed:** มีหลาย seed เฉพาะ `unet_edge_aug` โมเดลอื่นใช้ seed เดียว การจัดอันดับที่ต่างกันไม่เกินราว 0.5 dB จึงไม่น่าเชื่อถือ
- **ภาพ test ไม่เป็นอิสระต่อกัน:** Kaggle test 24 ภาพมาจาก 6 หน้าต้นฉบับ ค่า p อาจดูดีเกินจริง
- **ขนาด DIBCO:** test มีแค่ 20 ภาพ
- **pFM / DRD:** implement เอง ยังไม่ได้เทียบกับ evaluation tool ทางการของ DIBCO
- **Latency ของ LoRA:** วัดแบบยังไม่ merge น้ำหนักเข้ากับ conv"""


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(keepends=True)}


def code(text):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
            'source': text.splitlines(keepends=True)}


def annotate(nb):
    cells = nb['cells']
    assert cells[0]['cell_type'] == 'markdown' and len(cells) == 11, 'unexpected runner layout'
    out = [md(TITLE)]
    for i, cell in enumerate(cells[1:], start=1):
        assert cell['cell_type'] == 'code'
        out += [md(BEFORE[i]), cell]
    out += [md(REPORT_INTRO), code(SETUP), md(TABLES_MD), code(TABLES), md(FIGS_MD), code(FIGS), md(CONCLUSION)]
    nb['cells'] = out
    return nb


if __name__ == '__main__':
    src, dst = map(Path, sys.argv[1:3])
    nb = annotate(json.loads(src.read_text()))
    dst.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n')
    print('wrote', dst, len(nb['cells']), 'cells')
