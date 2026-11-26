# mini_project — โครงสร้างโค้ดและผล

ภาพรวมโปรเจกต์และผลทั้งหมดอยู่ใน [README หลัก](../README.md)

| โฟลเดอร์ | คืออะไร |
|---|---|
| `notebooks/final_colab.ipynb` | Notebook ที่ส่งอาจารย์ (Final) สร้างจาก `build_final_colab.py` |
| `notebooks/cross_domain_colab.ipynb` | Notebook ของ cross-domain benchmark 3 × 3 สร้างจาก `build_cross_domain_colab.py` |
| `src/` | โค้ด pipeline ของ Final: dataset, models, losses, `train_engine`, `eval_suite`, `cross_domain` |
| `tests/` | pytest (`.venv/bin/pytest -q`) |
| `tools/` | สคริปต์สร้างรูป กราฟสไลด์ และไฟล์ report (`make_report_docx.py`) |
| `results/final_benchmark/` | ผลประเมินรายภาพของทุกโมเดล Final บน Kaggle / Augraphy / DIBCO พร้อม latency และ summary |
| `results/cross_domain/` | ผล cross-domain (`eval/` 15 ไฟล์ และตาราง `cross_domain.md`) |
| `outputs/` | split manifest ที่ตรึงไว้ (Kaggle `phase1/`, DIBCO `final/`) |
| `midterm/` | งาน midterm รวม `PROJECT_SPEC.md` (แผนเดิม) เก็บไว้เป็นประวัติ: notebook, โค้ดเดิม และผล (ดู tag `midterm`) |

ไม่ได้เก็บใน Git: `data/`, `runs/`, checkpoint (`*.pt`) และไฟล์ zip ทั้งหมด อยู่ในเครื่องหรือบน Google Drive
