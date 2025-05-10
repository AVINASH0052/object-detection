# Object Detection Model Building

This project builds an object detection model using a pre-trained ResNet50 backbone and an SSD detection head, trained on the Pascal VOC 2012 dataset.

## Project Structure
```
ObjectDetection/
│
├── data/
│   └── VOC2012/
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── eval.py
│   └── utils.py
├── outputs/
│   ├── weights/
│   └── results/
├── requirements.txt
└── README.md
```

## Setup
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset
The project uses the Pascal VOC 2012 dataset. Ensure the dataset is placed in the `data/VOC2012/` directory with the following structure:
- `Annotations/`: XML files for bounding box annotations
- `JPEGImages/`: Image files
- `ImageSets/Main/`: Split files (train.txt, val.txt)

## Training
Run the training script:
```bash
python src/train.py
```

## Evaluation
Run the evaluation script:
```bash
python src/eval.py
```

## Notes
- The model uses a simplified SSD detection head.
- Adjust hyperparameters in `train.py` as needed. 