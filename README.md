
# Human Body Measurement Estimation from Images Using Deep Learning for Commercial Applications​

This project trains a deep learning model to predict detailed human body measurements from simple front and side silhouette images.

It builds on the **BodyM Dataset** (Amazon & Boston University, 2022) and introduces a key improvement — a **synthetic scale reference** generated from real **mobile phone dimensions**.
By inserting a simulated phone into each image, the model learns consistent spatial scaling, enabling accurate measurement predictions without needing user-provided height or distance calibration.

---

### **Key Features**
- Enhanced **ResNet-101** backbone for multi-input learning (image + phone + gender)
- Synthetic phone simulation using real dimensions from a Kaggle phone-spec dataset
- Weighted loss for balanced prediction across 14 anthropometric features
- Automatic checkpointing, normalization, and evaluation tools

---

###  **Dataset Sources**
- **BodyM Dataset:** [https://github.com/nathanielruiz/bodym-dataset](https://github.com/nathanielruiz/bodym-dataset)
- **Phone Specs Dataset:** available on [Kaggle – Mobile Phone Dimensions](https://www.kaggle.com/datasets/) (any dataset containing brand, model, height, width, depth)

---

###  **Usage**
```bash
# Train
python src/train.py

# Evaluate
python src/test.py

License

Released under CC BY-NC 4.0 — for non-commercial research use only.
For commercial inquiries, please contact Kelechi Nwosu.
