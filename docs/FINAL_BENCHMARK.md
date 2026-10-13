# Final Benchmark — AIE464

> **เอกสารประวัติการวางแผน/พัฒนา:** เนื้อหาและสถานะด้านล่างบันทึกไว้ก่อนชุด Final และอาจต่างจาก implementation จริง ผลล่าสุดใช้ [Final Notebook](../mini_project/FINAL_SUBMISSION/aie464-text-preserving-image-denoising-final.ipynb) และ [Presentation Brief](PRESENTATION_BRIEF.md) เป็นหลัก Notebook รุ่นพัฒนาที่กล่าวถึงเก็บไว้ในเครื่องและไม่รวมใน GitHub

## ขอบเขตและสถานะ

โค้ดพร้อมสำหรับ Smoke Test และ Training ที่ต้องได้รับอนุมัติแยก
**ยังไม่มี Full Training, ผล Colab T4 หรือผลประเมิน Test Set**
การรันสำเร็จไม่ได้ยืนยันว่าโมเดลลด noise หรือรักษาตัวอักษรได้ดี

ใช้ `split_manifest.json` เดิมเท่านั้น (SHA256
`5b569d65a4d21aacff84e25986fce522847ce61a4575cd720cd60ca4a4180539`):
Train 100 คู่/25 กลุ่ม, Validation 20 คู่/5 กลุ่ม, Test 24 คู่/6 กลุ่ม
ภาพ noisy สี่แบบของแต่ละหน้า clean อยู่ชุดเดียวกันทั้งหมด
Test เป็นข้อมูลสงวนจาก training archives ไม่ใช่ Kaggle public `test.zip`

## โมเดลและการเปรียบเทียบ

| Experiment | Model | Objective |
|---|---|---|
| `cnn_l1` | ShallowCNN เดิม: 5 convolutions, width 16 | L1 |
| `dncnn_lite` | 7 convolutions, width 32, BatchNorm, residual noise prediction | L1 |
| `unet_l1` | SmallUNet: 3 downsampling levels, widths 16/32/64/128 | L1 |
| `unet_edge` | SmallUNet เหมือน `unet_l1` รวมทั้งค่าเริ่มต้น weights | L1 + 0.1 × Sobel L1 |

DnCNN-lite เป็นโมเดลย่อ ไม่ใช่ DnCNN 17 ชั้นต้นฉบับ Sobel ใช้ kernels x/y
หาร 8, reflect padding และคำนวณ loss float32 ไม่มี SSIM loss ตามขอบเขตล่าสุด
SSIM เป็นตัววัดผล ส่วน OCR CER ใน PROJECT.md ยังไม่ได้ทำในขั้นตอนนี้

ทุกโมเดลใช้ seed 42, Adam lr=0.001, gradient clipping 1, batch 1,
aligned random crop 256×256 โดยใช้ generator แยกจากโมเดล กำหนดด้วย seed+epoch
ทำให้ลำดับภาพและตำแหน่ง crop เหมือนกันทุกโมเดล Train 100 updates ต่อ epoch
ค่าเริ่มต้นสำหรับ Training ที่เสนอคือ 20 epochs/2,000 updates ต่อโมเดล
หากลดงบให้ลดเท่ากันทั้งหมด ห้ามใช้ Test เลือก hyperparameters หรือจำนวน epochs

Validation ใช้ 20 หน้าเต็มทุก epoch และเลือก **best checkpoint จาก Val PSNR**
คะแนนทั้ง noisy baseline และ denoised ใช้ float32 PSNR/SSIM ชุดเดียวกัน
SSIM: Gaussian 11×11 sigma 1.5, valid windows, population covariance, data range 1
clip ผลภาพทุกโมเดลเป็น [0,1] เฉพาะขั้นตอนประเมิน/แสดงภาพ
DnCNN ฝึกจาก residual output ที่ยังไม่ clip เพื่อรักษา gradients
U-Net pad ภายในให้หาร 8 ลงตัวแล้วตัดกลับขนาดเดิม ไม่ resize หน้าเอกสาร

## ไฟล์และผลลัพธ์

- `mini_project/src/models/dncnn.py`, `unet.py`: โมเดลใหม่
- `mini_project/src/losses.py`: L1/Sobel
- `mini_project/src/train.py`: ขั้นตอนฝึกร่วม, checkpoint, Resume
- `mini_project/src/benchmark.py`: ตรวจ manifest/คู่ภาพ, validation, metrics/ภาพ,
  ฟังก์ชัน Final Test ที่มี approval gate
- `mini_project/notebooks/03_final_benchmark_colab.ipynb`: Colab orchestration
- `mini_project/prepare_benchmark_bundle.py`: ZIP แบบ allowlist ไม่มี upload
- `mini_project/tests/smoke_benchmark.py`: การตรวจ CPU ขนาดเล็ก

ไม่มีการแก้ dataset.py, evaluate.py, ShallowCNN เดิม, notebook Phase 2A,
manifest หรือข้อมูลภาพ

ผลแต่ละโมเดลอยู่ใต้
`/content/drive/MyDrive/AIE464/mini_project/outputs/final_benchmark/<RUN_ID>/<experiment>/`:

- `config.json`, `step_log.json`: loss รวม/L1/edge, ID/crop, gradient norm,
  เวลาสะสมและ peak allocated/reserved GPU memory
- `validation.json`: ผลรายหน้า/รายกลุ่มและค่าเฉลี่ยทุก epoch, train/val runtime
- `metrics.json`: ผล validation ของ checkpoint ที่เลือก, runtime, memory, versions
- `best.pt`: weights ที่เลือกด้วย Val PSNR และ provenance ใช้ inference เท่านั้น
- `last.pt`: weights, optimizer/scaler, RNG, ประวัติผล, source/data hashes,
  manifest checksum และ best weights ที่กู้คืนได้ ใช้ไฟล์นี้สำหรับ Resume
- `loss.png`, `comparisons.png`: Noisy/Denoised/Clean และ center crop ตำแหน่งเดียวกัน

ไฟล์ภายใน run เดิมอัปเดตได้เฉพาะ Resume ที่ระบุชัดเจน Run ใหม่ห้ามใช้ folder เดิม
Checkpoint และ JSON ใช้ temporary file แล้ว replace เพื่อไม่เผยแพร่ไฟล์ที่เขียนไม่เสร็จ
มี checkpoint epoch 0 จึง Resume ได้เมื่อหยุดใน epoch แรก
Google Drive/FUSE อาจมีความล่าช้าในการ sync ควรตรวจไฟล์ใน Drive ก่อนปิด runtime

## เตรียม ZIP และรัน Colab

1. จากโฟลเดอร์ AIE464 บนเครื่อง local สร้าง ZIP **หลังอนุมัติการเตรียมข้อมูล**:

   ```bash
   python3 mini_project/prepare_benchmark_bundle.py --include-test
   ```

   ได้ `mini_project/outputs/benchmark_bundle.zip` ทั้งหมด 298 ไฟล์:
   source/notebook/manifest 10 ไฟล์ และภาพ 288 ไฟล์ รวม Test 48 ไฟล์
   ไม่มีการ decode/ประเมิน Test, ดาวน์โหลด, upload หรือแก้ภาพต้นทาง
   หากเตรียมเฉพาะ pilot ใช้คำสั่งเดิมโดยตัด `--include-test` จะได้ 250 ไฟล์
   หาก ZIP ชื่อเดิมมีอยู่ ให้ใช้ `--output /path/to/new_bundle.zip` ห้าม overwrite

2. อัปโหลด ZIP ครั้งเดียวไป `MyDrive/AIE464/benchmark_bundle.zip`
   หากชื่อเดิมมีอยู่ ให้ใช้ชื่อใหม่และปรับ `BUNDLE_PATH` ไม่ overwrite โดยไม่ได้อนุมัติ
   เปิด notebook ใหม่ใน Colab และเลือก T4 GPU
3. Notebook mount Drive แล้วคัดลอก ZIP ของคุณไป `/content` และแตกไฟล์เพื่ออ่านเร็ว
   **PROJECT_DIR สำหรับ source/data:** `/content/aie464_benchmark/mini_project`
   **DRIVE_PROJECT_DIR สำหรับผล:** `/content/drive/MyDrive/AIE464/mini_project`
   ไม่แก้หรือ overwrite source ที่มีอยู่เดิมบน Drive ไม่มี pip install/download cell
4. รันโหมดเริ่มต้น Smoke: 1 update ต่อโมเดล, validation สองหน้าเต็มจากสองกลุ่ม
   ตั้ง `RUN_ID` ใหม่ทุก run แล้วตรวจ shapes, loss, ภาพ และ memory บน T4
5. ส่งผล pilot เพื่อขออนุมัติ Full Training ก่อนเปลี่ยน
   `RUN_FULL_TRAINING=True`, `TRAINING_APPROVED=True`, `EPOCHS=20` และ `RUN_ID` ใหม่
6. ถ้าหลุด ใช้ config/RUN_ID/environment เดิมและ `RESUME=True`
   ระบบกู้จาก epoch ที่เสร็จล่าสุด หากหยุดกลาง epoch จะ replay epoch นั้น
   จำนวน updates ที่บันทึกไม่เพิ่มเกินงบ ตรวจ source/data/version ตรงเดิมก่อน Resume
   ไม่รองรับเปลี่ยนงบ/config ระหว่าง run ให้เริ่ม run ใหม่เมื่อเปลี่ยนเงื่อนไข
7. เมื่อฝึกครบทั้งสี่โมเดล ให้ขออนุมัติ Test แยกก่อนเปิด
   `RUN_FINAL_TEST=True`, `FINAL_TEST_APPROVED=True`
   ระบบตรวจงบ/config/source/data/environment เท่ากันและครบก่อนอ่านภาพ Test

Final Test บันทึก `final_test/metrics.json`, `summary.csv`, JSON รายโมเดล และภาพ
หากหยุดกลางการประเมิน ใช้ flags และ run เดิมรันต่อได้จาก `final_test.partial/`
ตรวจ protocol/checkpoint fingerprints เดิมและข้ามเฉพาะผลโมเดลที่บันทึกครบแล้ว
เมื่อครบทั้งสี่จึงเผยแพร่ `final_test/` ผลที่เสร็จครบแล้วห้าม overwrite
ทุกโมเดลใช้ 24 คู่เดียวกันและ preview หน้า/ตำแหน่งเดียวกัน รายงานค่าเฉลี่ยรายคู่
และรายกลุ่ม (มีเพียง 6 กลุ่ม Test จึงไม่ควรอ้างว่า 24 คู่เป็นตัวอย่างอิสระ)
Inference time ใช้ batch 1/AMP, warm-up สอง forward, median ของห้า forward ต่อภาพ
ซิงก์ GPU และรวม model padding/output clamp แต่ไม่นับ IO/transfer/metrics
PyTorch memory counters ไม่รวม driver overhead หรือ process อื่น
T4 memory/time จริงยังต้องวัด ไม่มีการรับประกันจาก CPU smoke

## หลักฐานการตรวจจริง

รัน `mini_project/tests/smoke_benchmark.py` บน CPU PyTorch 2.8.0:

- ทั้งสี่โมเดลรักษา shapes 420×540, 258×540 และ 259×541; forward finite
- optimizer update หนึ่งครั้งต่อโมเดลผ่าน finite backward และ weights เปลี่ยนจริง
- crop noisy/clean ใช้พิกัดเดียวกัน ตรวจด้วย synthetic coordinate grid
- ทุกโมเดลใช้ Train ID/crop เดียวกันและ U-Net ทั้งสองมี initial weights hash เดียวกัน
- Sobel loss ตรงสูตร, ภาพเหมือนกัน loss เป็นศูนย์ และ gradient finite
- checkpoint reload ให้ prediction ตรงกันและ PNG/JSON/checkpoints เปิดได้
- จำลองหยุดหลังบันทึก epoch แล้ว Resume ได้ แม้ best.pt ยังไม่ถูกเขียน
  weights, crop และ loss ตรงกับ run ต่อเนื่องแบบ exact บน CPU
- จำลองหยุดก่อนเริ่ม epoch แรกแล้ว Resume ได้; stale best checkpoint ถูกปฏิเสธ
- ตรวจการกู้ artifact staging/protocol marker ด้วยไฟล์จำลอง โดยไม่ประเมิน Test
- approval gates, overwrite guard, changed-config Resume rejection ผ่าน
- manifest checksum ไม่เปลี่ยน และดัก Image.open ยืนยันเปิดภาพ Test **0 ไฟล์**

รวม **9 optimizer updates ต่อ verification run** (สี่โมเดล + การตรวจ Resume ของ CNN)
หลักฐานล่าสุดอยู่ `mini_project/outputs/final_benchmark_smoke_final/smoke_checks.json`
Smoke scores จาก validation สองหน้าไม่ใช่ Benchmark และไม่ควรใช้สรุปว่าโมเดลใดดีที่สุด
ยังไม่ได้รัน CUDA/AMP บน T4, Full Training, Final Test หรือ OCR

ประมาณการ Training จากแผนเดิม 2–4 ชั่วโมงรวม pilot/evaluation เป็นการคาดการณ์
ให้ปรับจาก pilot T4 จริง โดยคงงบเท่ากันทั้งสี่โมเดล
