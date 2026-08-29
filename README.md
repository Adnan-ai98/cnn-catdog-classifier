# 🐾 CatDogCNN — Cat vs Dog Classifier (From Scratch)

A convolutional neural network built and trained **from scratch** (no pretrained weights) to classify images as cat or dog — plus a Flask web interface (**PawScan**) for live predictions.

> Part of the [Barakode AI Engineering Roadmap](#) — Week 1: *"Develop custom CNN architectures for image tasks."*

Add a screenshot of the PawScan interface here once you have one:
![PawScan interface](screenshots/pawscan-demo.png)

## Results

| Metric | Value |
|---|---|
| Best validation accuracy | **86.58%** (epoch 9) |
| Training set | 19,998 images |
| Validation set | 5,000 images |
| Dataset | [Microsoft Cats vs Dogs](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset) (Kaggle) |
| Training time | ~15.5 min on Kaggle T4 GPU |

Full training curve, architecture details, and limitations are documented in [`model_card_catdog_cnn.md`](model_card_catdog_cnn.md).

## Architecture

A 3-block CNN (Conv2d → ReLU → MaxPool ×3) with a fully connected classifier head, ~2.1M parameters, trained with `BCEWithLogitsLoss` and Adam. See the model card for the full layer-by-layer breakdown.

## Project structure

```
.
├── cnn_cat_dog_kaggle.ipynb       # Training notebook (run on Kaggle, GPU)
├── model_card_catdog_cnn.md       # Full model card: architecture, training config, results
├── pawscan_app/                   # Flask interface for live predictions
│   ├── app.py
│   ├── templates/index.html
│   └── requirements.txt
└── README.md
```

## Running the training notebook

1. Open the notebook on [Kaggle](https://www.kaggle.com) with the [Microsoft Cats vs Dogs dataset](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset) attached.
2. Turn on GPU (Settings → Accelerator → GPU T4).
3. Run all cells. The best checkpoint is saved as `cat_dog_cnn_best.pt`.

> **Note:** The trained weights (`cat_dog_cnn_best.pt`) aren't included in this repo (large binary file). Download it from your Kaggle notebook's Output tab after training, or train your own with the notebook above.

## Running the web interface

```bash
cd pawscan_app
pip install -r requirements.txt
# place cat_dog_cnn_best.pt in this folder, next to app.py
python app.py
```

Then open `http://localhost:5000` and drop in a photo of a cat or dog.

## What this demonstrates

- CNN architecture designed and implemented from scratch in PyTorch (no transfer learning)
- Custom training loop with loss/optimizer configuration, checkpointing, and per-epoch logging
- Data pipeline handling (corrupt file filtering, train/val split) on a real-world dataset
- Model documentation via a formal model card
- Deployment as a working web interface (Flask + vanilla JS)

## License

MIT (or your preferred license — update this line).
