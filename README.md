# AIE464 — Text-Preserving Image Denoising

โปรเจกต์ลดสัญญาณรบกวนในภาพเอกสารและศึกษาการรักษาขอบตัวอักษรด้วย PyTorch
สำหรับ Google Colab T4 เปรียบเทียบ CNN + L1, DnCNN-lite, U-Net + L1 และ
U-Net + L1 + Sobel Edge-aware Loss

- [โครงสร้างและวิธีรัน Colab](mini_project/README.md)
- [ข้อกำหนดโปรเจกต์](docs/PROJECT.md)
- [ข้อมูลและการแบ่งชุด](docs/DATASET.md)
- [คู่มือ Final Benchmark และสถานะการตรวจจริง](docs/FINAL_BENCHMARK.md)
- [กฎการทำงานของ Agent](AGENTS.md)

ผ่าน Smoke Test บน CPU แล้ว ยังไม่ได้ Full Training หรือประเมิน Test Set
โค้ดรันได้ไม่ได้ยืนยันว่าโมเดลให้ผลดี ต้องตรวจ pilot บน T4 ก่อนเริ่ม Training เต็ม

Repository นี้เก็บ source, notebooks, docs และ frozen split manifest เท่านั้น
Dataset, ZIP, checkpoints, ผลทดลอง, credentials และงานวิชาอื่นไม่รวมใน Git
