# Development Pipeline — AIE464 Text-Preserving Image Denoising

> **Final Implementation อยู่ใน [Final Notebook](FINAL_SUBMISSION/aie464-text-preserving-image-denoising-final.ipynb)** ซึ่งมีโค้ดและ outputs จริงครบชุด ดู [README ปัจจุบัน](../README.md) และ [Presentation Brief](../docs/PRESENTATION_BRIEF.md) สำหรับผลและการทำสไลด์
>
> `src/`, `prepare_benchmark_bundle.py` และ `tests/smoke_benchmark.py` เก็บไว้เป็น Development Pipeline รุ่นเดิม U-Net ใน `src/` ใช้ Sigmoid output ขณะที่ Final Notebook ใช้ residual output จึงไม่ใช่ implementation เดียวกับ Final Benchmark เนื้อหาวิธีรันและสถานะด้านล่างเป็นบันทึกช่วงพัฒนา Notebook `02a`, `03`, `04`, `05` อยู่ในเครื่องและเลิกติดตามใน Git; `prepare_benchmark_bundle.py` ยังต้องใช้ Notebook `03` ในเครื่อง จึงไม่ใช่คำสั่งรันจาก GitHub clone ใหม่โดยลำพัง

ลด noise ในภาพเอกสารโดยคงขนาดภาพเดิมด้วย PyTorch บน Google Colab T4
เปรียบเทียบ CNN + L1, DnCNN-lite, U-Net + L1 และ U-Net + L1 + Sobel Loss
วัด PSNR, SSIM, ภาพ Output และเวลา Inference (เวลาประมวลผลของโมเดล)

## โครงสร้าง

```text
AIE464/                         # Git root ที่เสนอ
├── AGENTS.md
├── docs/                       # ข้อกำหนดและหลักฐานการตรวจ
└── mini_project/
    ├── src/                    # Dataset, models, loss, train, benchmark
    ├── FINAL_SUBMISSION/       # Git เก็บ Final Notebook ฉบับเดียว
    ├── results/final/          # JSON, logs และ PNG ที่ตรวจตรงกับ ZIP
    ├── notebooks/              # Notebook รุ่นพัฒนาอยู่เฉพาะในเครื่อง
    ├── tests/                  # CPU smoke checks
    ├── prepare_benchmark_bundle.py
    ├── outputs/phase1/split_manifest.json  # Track เฉพาะ frozen manifest
    ├── data/                   # Local/Drive เท่านั้น ไม่ Track
    └── outputs/                # ผลรันและ checkpoints ไม่ Track
```

ใช้ split manifest เดิม: Train 100 คู่/25 กลุ่ม, Val 20 คู่/5 กลุ่ม,
Test 24 คู่/6 กลุ่ม โดย noisy สี่แบบของหน้า clean เดียวกันอยู่ชุดเดียวกัน
เลือก checkpoint ด้วย Validation PSNR และใช้ Test เฉพาะการประเมินสุดท้าย

## รันบน Colab

1. เตรียม paired images จาก Kaggle Dirty Documents ที่ได้รับสิทธิ์เข้าถึงแล้ว
   ไว้ใน `data/raw/train` และ `data/raw/train_cleaned` ชื่อไฟล์ต้องตรงกัน
   Repository ไม่รวมข้อมูล ไม่ดาวน์โหลดหรือใช้ Kaggle token ให้อัตโนมัติ
2. จาก AIE464 root สร้าง ZIP สำหรับ upload ครั้งเดียว:

   ```bash
   python3 mini_project/prepare_benchmark_bundle.py --include-test
   ```

   อัปโหลด `mini_project/outputs/benchmark_bundle.zip` ไป
   `MyDrive/AIE464/benchmark_bundle.zip` หากมีชื่อเดิมให้ใช้ชื่อใหม่ ไม่ overwrite
   ZIP นี้คัดลอก Test files เท่านั้น ไม่ประเมิน Test
3. เปิดไฟล์ในเครื่อง `notebooks/03_final_benchmark_colab.ipynb`
   ใน Colab เลือก T4 GPU และเริ่มใน runtime ใหม่ Mount Drive แล้วรัน Smoke ตามค่าเริ่มต้น
   ใช้ PyTorch, NumPy, Pillow และ Matplotlib ที่มีใน Colab ไม่มี install cell
4. `PROJECT_DIR` สำหรับ source/data คือ `/content/aie464_benchmark/mini_project`
   ผลและ checkpoints บันทึกที่
   `/content/drive/MyDrive/AIE464/mini_project/outputs/final_benchmark/<RUN_ID>/`
5. ตรวจ pilot T4 ก่อนขออนุมัติ Full Training ใช้ `RUN_ID` ใหม่สำหรับ run ใหม่
   Resume (รันต่อจากจุดที่หยุด) ใช้ config เดิมและ `RESUME=True`
   เปิด Final Test หลังได้รับอนุมัติแยกเท่านั้น

## สถานะและเอกสาร

ผ่านการตรวจ CPU smoke: shapes, aligned crops, backward, checkpoint reload,
Resume และ approval gates ยังไม่ได้ยืนยัน CUDA/AMP บน T4, Full Training
หรือ Test metrics **โค้ดผ่านไม่ได้แปลว่าโมเดลให้ภาพดี** OCR CER ยังไม่ได้ทำ

- [PROJECT.md](../docs/PROJECT.md) และ [DATASET.md](../docs/DATASET.md): เอกสารหลัก
- [FINAL_BENCHMARK.md](../docs/FINAL_BENCHMARK.md): protocol, ZIP, Resume และหลักฐานจริง
- [PHASE2A_SANITY.md](../docs/PHASE2A_SANITY.md): notebook CNN ขั้นต้น
- `PROJECT_SPEC.md`: เอกสารแนวคิดเดิม ใช้เอกสารหลักข้างต้นสำหรับขอบเขตปัจจุบัน

Git เก็บ Final Notebook, เอกสาร, source/scripts, frozen manifest และสำเนาผลสุดท้าย 8 ไฟล์ตาม allowlist หรือรายการอนุญาตเฉพาะไฟล์; Dataset, ZIP, checkpoints,
ผลทดลองอื่นและ secrets ถูก ignore อย่าใช้ `git add -f` กับไฟล์เหล่านี้
การตรวจ secret patterns ช่วยลดความเสี่ยง แต่ต้องทบทวน staged diff ก่อน commit เสมอ
