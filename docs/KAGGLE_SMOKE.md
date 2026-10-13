# Kaggle — GPU Smoke เท่านั้น

> **เอกสารประวัติการวางแผน/พัฒนา:** เนื้อหาและสถานะด้านล่างบันทึกไว้ก่อนชุด Final และอาจต่างจาก implementation จริง ผลล่าสุดใช้ [Final Notebook](../mini_project/FINAL_SUBMISSION/aie464-text-preserving-image-denoising-final.ipynb) และ [Presentation Brief](PRESENTATION_BRIEF.md) เป็นหลัก Notebook รุ่นพัฒนาที่กล่าวถึงเก็บไว้ในเครื่องและไม่รวมใน GitHub

Notebook: `mini_project/notebooks/04_final_benchmark_kaggle.ipynb`
ดัดแปลง environment จาก Colab โดยไม่เปลี่ยน Dataset, models, loss,
Train/Validation protocol หรือ frozen split manifest

## เตรียมและรันหลังได้รับอนุมัติ

1. อัปโหลด notebook ไป Kaggle และเพิ่ม Input Dataset เดิม
   `thames98451/aie464-denoising-benchmark` เลือก GPU accelerator
2. Source/data อ่านจาก read-only path:
   `/kaggle/input/datasets/thames98451/aie464-denoising-benchmark/mini_project`
   Notebook ตรวจ checksum ของ source 8 ไฟล์และ manifest ก่อนใช้
   หาก source ไม่ตรง ให้ปรับ Input Dataset เป็น source เวอร์ชันที่อนุมัติ
   ไม่แก้ pipeline ใน Notebook ไม่มี install/download commands
3. **รออนุมัติ GPU Smoke ก่อน** จึงตั้ง `GPU_SMOKE_APPROVED=True`
   และเลือก `RUN_ID` ใหม่ จากนั้นรัน cells ตามลำดับใน kernel ใหม่
   คง `MODE='smoke'`, `EPOCHS=1` และ flags Training/Test ทั้งหมดเป็น False
4. ใช้ GPU 0 เพียงตัวเดียว ตรวจ CUDA float16 autocast และ GradScaler
   ทุกโมเดลทำ 1 optimizer update, Validation สองหน้าเต็มจากสองกลุ่ม
   seed/optimizer/patch/loss เหมือน Colab ใช้ train_experiment เดิม
5. ผลอยู่ `/kaggle/working/final_benchmark/<RUN_ID>/`
   มี loss, PSNR/SSIM, runtime, peak GPU memory, best/last checkpoints และภาพ
   `smoke_verification.json` สรุปงบ update, page/crop sequence,
   U-Net initialization, artifacts และ finite forward/backward
   GPU memory จำกัด 6 GiB ตาม pipeline; PyTorch counters ไม่รวม driver overhead

Notebook นี้ปฏิเสธ Full Training/Final Test แม้เปิด flags
ต้องขออนุมัติและปรับ Notebook ในงานใหม่ก่อนขยายขอบเขต ไม่มีการเรียก final_benchmark
Test IDs ใช้เป็น metadata ตรวจ split เท่านั้น ไม่เปิดภาพ Test เพื่อวัดผล

## สำรองออกนอก Session

สร้าง ZIP + `.sha256` หลังแต่ละโมเดล เมื่อ training cell เกิดข้อผิดพลาด
และหลังสรุปผล โดยสำรองเฉพาะ checkpoints/logs/ภาพของ run นี้ ไม่มี dataset หรือ secrets
ไฟล์อยู่ `/kaggle/working/backups/` และมีลิงก์ดาวน์โหลดท้าย Notebook
หาก FileLink ใช้ไม่ได้ ให้ดาวน์โหลดผ่าน Kaggle Output/file browser

**ZIP ใน Working อย่างเดียวยังสูญหายพร้อม Session ได้** ก่อนปิด Session:

- ดาวน์โหลด ZIP ล่าสุดและ `.sha256` ไปเครื่อง/Drive แล้วตรวจว่าไฟล์เปิดได้
- หรือบันทึก Notebook Version พร้อม outputs และตรวจในหน้า Output ว่ามี ZIP/checkpoints
  การ Save เฉพาะ source notebook ไม่ได้ยืนยันว่ามี checkpoints สำรองแล้ว
- Save & Run All จะรันใหม่ จึงห้ามกดก่อนอนุมัติ GPU Smoke

อ้างอิง: [Kaggle Notebooks documentation](https://www.kaggle.com/docs/notebooks)

กู้คืน: เพิ่ม ZIP ของตัวเองเป็น Kaggle Input ส่วนตัว ใช้ RUN_ID เดิม,
ตั้ง `RESTORE_ARCHIVE`, `EXPECTED_BACKUP_SHA256` จาก `.sha256` และ `RESUME=True`
Notebook ตรวจ checksum, allowlist, paths และ hashes ทั้งหมดก่อนคัดลอกกลับ Working
ห้าม overwrite run เดิม จากนั้น pipeline เดิมตรวจ config/data/source/runtime versions
Resume จาก epoch ที่เสร็จล่าสุดหรือ epoch 0 หากหยุดก่อนจบ epoch แรก
Smoke ที่เสร็จแล้วไม่เพิ่ม optimizer updates เมื่อ Resume

## สถานะการตรวจ

ตรวจในเครื่องเฉพาะ notebook schema, Python syntax, approval gates,
source checksums และฟังก์ชัน ZIP/restore ด้วยไฟล์จำลอง ไม่มี optimizer update
ไม่ได้รัน CUDA preflight, GPU Smoke, Training หรือ Test evaluation
ยังไม่ยืนยันว่า path/ไฟล์บน Kaggle ตรงจริง และยังไม่มีผลคุณภาพโมเดล
