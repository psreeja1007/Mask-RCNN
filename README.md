# Fine-Tuning Mask R-CNN on Penn-Fudan Dataset using Detectron2

## 🧠 Objective

This project aims to evaluate the effectiveness of transfer learning by fine-tuning a pre-trained Mask R-CNN model (originally trained on the COCO dataset) on the Penn-Fudan pedestrian detection dataset. The goal is to assess improvements in both bounding box and instance segmentation performance after fine-tuning, and to provide an interactive demo using Gradio.

---

## 📂 Dataset: Penn-Fudan Pedestrian Detection

- **Source:** [Penn-Fudan Database for Pedestrian Detection and Segmentation](https://www.cis.upenn.edu/~jshi/ped_html/)
- **Description:** Contains 170 images with 345 labeled pedestrian instances.
- **Annotations:** Each image comes with corresponding segmentation masks and bounding boxes.
- **Format Used:** Converted to COCO-style dataset format for compatibility with Detectron2.
- **Classes:** Single class - *person*

---

## ⚙️ Approach

1. **Framework Used:** [Detectron2](https://github.com/facebookresearch/detectron2) by Facebook AI Research.
2. **Base Model:** `mask_rcnn_R_50_FPN_3x.yaml` from the `COCO-InstanceSegmentation` model zoo.
3. **Environment:** Implemented in [Google Colab](https://colab.research.google.com/) using PyTorch backend.
4. **Steps Followed:**
   - Loaded and visualized the Penn-Fudan dataset.
   - Converted annotations to COCO format.
   - Configured Detectron2 for custom training.
   - Fine-tuned the pre-trained COCO model on Penn-Fudan.
   - Evaluated performance before and after fine-tuning.
   - Created a Gradio demo for live inference on test images.

---

## 📊 Results

### 📦 Bounding Box Metrics (COCO Evaluation)

| Metric                      | Pre-trained (COCO) | Fine-tuned (Penn-Fudan) |
|----------------------------|--------------------|--------------------------|
| AP (bbox)                  | 80.61              | 82.039                   |
| AP50 (bbox)                | 97.81              | 98.157                   |
| AP75 (bbox)                | 94.22              | 94.55                    |
| AP_small (bbox)            | 59.78              | 68.91                    |
| AP_medium (bbox)           | 78.99              | 83.68                    |
| AP_large (bbox)            | 83.13              | 84.30                    |

### 🧩 Segmentation Metrics (COCO Evaluation)

| Metric                      | Pre-trained (COCO) | Fine-tuned (Penn-Fudan) |
|----------------------------|--------------------|--------------------------|
| AP (segm)                  | 71.67              | 75.78                    |
| AP50 (segm)                | 97.81              | 98.15                    |
| AP75 (segm)                | 89.69              | 92.83                    |
| AP_small (segm)            | 40.59              | 44.33                    |
| AP_medium (segm)           | 62.67              | 61.12                    |
| AP_large (segm)            | 75.71              | 79.54                    |


---

## 🌐 Gradio Demo

Try out the fine-tuned model through an interactive demo built using [Gradio](https://gradio.app/):

It is included in the **demo.ipynb** notebook
