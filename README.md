# PawScan — Cat vs Dog CNN Interface

A Flask interface for the from-scratch CNN trained on the Microsoft Cats vs
Dogs dataset (Barakode roadmap, Week 1 + Week 4).

## Setup

1. Download `cat_dog_cnn_best.pt` from your Kaggle notebook's Output tab
   (`/kaggle/working/cat_dog_cnn_best.pt`) and place it in this same folder,
   next to `app.py`.

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the app:
   ```
   python app.py
   ```

4. Open your browser at `http://localhost:5000`

## How it works

- Drop or browse a photo of a cat or dog.
- The image is resized to 128x128, normalized the same way as training,
  and passed through the CNN.
- The model outputs a single number (probability of "Dog"); the app turns
  that into a Cat/Dog label with a confidence score, shown as a scan report.

## Notes

- If the model file isn't found, the status pill in the top right will show
  "Model failed to load" and the interface will explain the error when you
  try to scan.
- `CatDogCNN` in `app.py` must match the architecture used during training
  exactly, or the saved weights won't load correctly.
