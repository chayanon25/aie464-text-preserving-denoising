# Kaggle Full Training — รออนุมัติ

> **เอกสารประวัติการวางแผน/พัฒนา:** เนื้อหาและสถานะด้านล่างบันทึกไว้ก่อนชุด Final และอาจต่างจาก implementation จริง ผลล่าสุดใช้ [Final Notebook](../mini_project/FINAL_SUBMISSION/aie464-text-preserving-image-denoising-final.ipynb) และ [Presentation Brief](PRESENTATION_BRIEF.md) เป็นหลัก Notebook รุ่นพัฒนาที่กล่าวถึงเก็บไว้ในเครื่องและไม่รวมใน GitHub

Notebook ใหม่: `mini_project/notebooks/05_full_training_kaggle.ipynb`
ใช้ source/models/loss/manifest เดิม ไม่แก้ Training Pipeline หรือ Notebook 04
Input อ่านอย่างเดียว:
`/kaggle/input/datasets/thames98451/aie464-denoising-benchmark/mini_project`
ผลอยู่ `/kaggle/working/final_benchmark/<RUN_ID>/`

## เงื่อนไขและการรัน

- 4 โมเดลเหมือนเดิม: CNN + L1, DnCNN-lite, U-Net + L1, U-Net + L1 + Sobel Edge
- ค่าเริ่มต้นทุกโมเดล: 20 epochs/2,000 updates, seed 42, patch 256, batch 1,
  Adam lr=0.001, edge weight 0.1 ใช้ GPU 0/AMP
- Train 100 คู่/epoch, Val 20 หน้าเต็ม/epoch เลือก best.pt ด้วย Validation PSNR
- ไม่เริ่มจาก Smoke weights: Training ใหม่เริ่มด้วย seed เดิมทุกโมเดล
- TRAINING_APPROVED=False เริ่มต้น; ห้ามประเมิน Test ไม่ว่าตั้ง flags อย่างไร

1. อัปโหลด Notebook ใหม่ไป Kaggle เพิ่ม benchmark Dataset เดิมและเลือก GPU
   source 8 ไฟล์ต้องตรง checksum เดิม ใช้ kernel ใหม่ ไม่มี install/download commands
2. เพิ่ม **Smoke Backup ZIP ที่ดาวน์โหลดแล้ว** เป็น Kaggle Input ส่วนตัว
   ตั้ง SMOKE_ARCHIVE เป็นพาธ ZIP จริง Notebook ใส่ SHA256 ของ Backup ที่ตรวจแล้วให้
   ไม่ใช้พาธ local บน Kaggle และไม่ต้องอัปโหลดภาพซ้ำ
3. รัน cells ตั้งค่าและตรวจ Smoke/เวลา (ยังไม่ Training) Notebook ตรวจ archive CRC,
   hashes, CUDA/AMP, source/manifest, ID/crop, losses และ memory ครบ 4 โมเดล
   เทียบ GPU/software ปัจจุบันกับ Smoke ก่อนใช้ประมาณเวลา
4. อ่านประมาณการและเวลาคงเหลือจริง ตั้ง SESSION_BUDGET_HOURS ได้ตามงบของคุณ
   หาก upper estimate เกินงบ ให้หยุด ขออนุมัติลด epochs เท่ากันทุกโมเดลหรือแยก Session
5. **หลังได้รับอนุมัติ Full Training เท่านั้น** ตั้ง TRAINING_APPROVED=True
   และ TIME_BUDGET_CONFIRMED=True จากนั้นรัน Training cell ใช้ RUN_ID ใหม่
6. ตรวจ loss, Val PSNR/SSIM, best/last checkpoints, runtime, GPU memory และภาพ
   Notebook ตรวจว่าครบงบ, Val coverage, sampling/crops และ U-Net initial weights เหมือนกัน

## ผล Smoke ที่อ่านจาก Backup จริง

Backup ใน `mini_project/backups/`:
`kaggle_smoke_20261013_01_20261013T092425Z_e3853ca1.zip`
SHA256: `647250700fe5165bbb46bc42f78a1593722272b0384fd6afab640d2de2b8306a`
ตรวจแล้วตรงกับ .sha256, CRC ผ่านและ content hashes ครบ
GPU Tesla T4, PyTorch 2.11.0+cu128, CUDA 12.8, seed/patch/source/manifest เดิม

| Model | Smoke train 1 update | Val 2 หน้า | Peak allocated/reserved สูงสุด | ประมาณ 20 epochs |
|---|---:|---:|---:|---:|
| CNN + L1 | 1.319 s | 0.244 s | 0.076 GiB | 27–108 นาที |
| DnCNN-lite | 0.163 s | 0.045 s | 0.131 GiB | 3–14 นาที |
| U-Net + L1 | 0.391 s | 0.078 s | 0.203 GiB | 8–32 นาที |
| U-Net + Edge | 0.061 s | 0.040 s | 0.203 GiB | 1–6 นาที |

รวมคร่าว ๆ **40–160 นาที เป็นการคาดการณ์ ไม่ใช่เวลาที่ฝึกจริง**
คำนวณ epoch ≈ 100 × Smoke train time + 10 × Val สองหน้า, คูณ 20 epochs,
รวม final validation และเผื่อ 20% IO/checkpoints/ZIP จากนั้นแสดงช่วง 0.5–2 เท่า
ไม่รวม queue หรือการดาวน์โหลด Backup ของผู้ใช้

มีเพียงหนึ่ง update และ CNN เป็นโมเดลแรกที่มี cold-start overhead
จึงอาจประเมิน CNN สูงเกินจริง ไม่ควรใช้ตารางนี้จัดอันดับความเร็วโมเดล
memory จาก Smoke ไม่รับประกันการใช้หน่วยความจำใน Full Training; pipeline หยุดถ้าเกิน 6 GiB
20 epochs ยังไม่รับประกัน convergence หรือคุณภาพ ต้องอ่านผล Val จริง

## Backup และ Resume

ZIP + SHA256 ถูกสร้างหลังแต่ละโมเดล, เมื่อ Python exception และหลังตรวจผลสุดท้าย
รวม loss/metrics/checkpoints/ภาพ/metadata ของ run นี้เท่านั้น ไม่มี dataset/secrets
ไฟล์อยู่ `/kaggle/working/backups/` มีลิงก์ดาวน์โหลดท้าย Notebook
**ดาวน์โหลดไปเครื่อง/Drive หรือบันทึก Notebook Version พร้อม outputs ก่อนจบ Session**
ตรวจว่า ZIP/.sha256 เปิดได้และ outputs มีจริง ไม่ใช่ Save เฉพาะ source

Working ไม่ใช่ backup นอก Session หาก kernel ถูกตัดทันที finally อาจไม่ทำงาน
กู้ได้เฉพาะ snapshot ที่เก็บนอก Session แล้ว อาจเสียความคืบหน้ากลางโมเดลหลัง Backup ล่าสุด
Pipeline บันทึก last.pt ทุก epoch แต่ไม่ได้ upload หรือ backup ออกนอก Kaggle อัตโนมัติ

Session ใหม่: เพิ่ม **Full Training ZIP** เป็น Input ส่วนตัว ตั้ง RESTORE_ARCHIVE,
EXPECTED_BACKUP_SHA256, RESUME=True และ RUN_ID/EPOCHS/config เดิม
ระบบตรวจ allowlist, paths และ hashes ก่อน restore ห้าม overwrite run เดิม
source/data/software/GPU versions ต้องตรงเดิม Resume จาก last.pt ของ epoch ที่เสร็จล่าสุด
โมเดลที่ฝึกครบแล้วไม่เพิ่ม updates; โมเดลที่ยังไม่เริ่มจะเริ่มตาม seed เดิม
ห้ามใช้ Smoke ZIP ใน RESTORE_ARCHIVE ของ Training

อ้างอิงวิธีเก็บ outputs: [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks)

## การตรวจในเครื่อง

ตรวจ Notebook schema/Syntax, config 20/42/256, source checksums, approval gates,
ตัวอ่าน Smoke Backup จริง และ ZIP/restore ด้วยไฟล์จำลอง โดยไม่รัน Training/CUDA
ยังไม่ได้วัด Full Training จริง ไม่ประเมิน Test และไม่กล่าวอ้างคุณภาพโมเดล
