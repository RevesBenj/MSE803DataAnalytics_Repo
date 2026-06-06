# CIFAR-10 Multi-Level Classification

## Project Overview

This project use CIFAR-10 dataset for image classification.  
It is not only normal 10-class classification. It also do multi-level classification.

The model predict 3 levels:

1. **Coarse level** - animal or vehicle  
2. **Group level** - air vehicle, road vehicle, domestic animal, wild land animal, etc.  
3. **Fine level** - airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck  

This framework was adapted and extended from baseline CIFAR-10 pipelines to support advanced hierarchical visual taxonomies.

Main Repository URL: `github.com/dev-architect/cifar10-multilevel-classification`

---

## Dataset Used

Dataset: **CIFAR-10**  
Official source: https://www.cs.toronto.edu/~kriz/cifar.html

CIFAR-10 have:

- 60,000 colour images
- 32x32 image size
- 10 classes
- 50,000 training images
- 10,000 testing images

Classes are:

- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

---

## What the Notebook is Doing

The notebook do this steps:

1. Import TensorFlow, NumPy, Matplotlib, and sklearn.
2. Load CIFAR-10 dataset using `tf.keras.datasets.cifar10.load_data()`.
3. Normalize image pixels from 0-255 into 0-1.
4. Create hierarchical labels.
5. Build CNN model using TensorFlow/Keras.
6. Create 3 output layers:
   - `coarse_output`
   - `group_output`
   - `fine_output`
7. Train the model.
8. Evaluate testing accuracy.
9. Show classification reports.
10. Save trained model as `CIFAR_10_tens.h5`.

---

## Multi-Level Label Design

### Level 1: Coarse Class

| Fine Class | Coarse Class |
|---|---|
| airplane | vehicle |
| automobile | vehicle |
| bird | animal |
| cat | animal |
| deer | animal |
| dog | animal |
| frog | animal |
| horse | animal |
| ship | vehicle |
| truck | vehicle |

### Level 2: Group Class

| Fine Class | Group Class |
|---|---|
| airplane | air_vehicle |
| automobile | road_vehicle |
| bird | flying_animal |
| cat | domestic_animal |
| deer | wild_land_animal |
| dog | domestic_animal |
| frog | amphibian |
| horse | wild_land_animal |
| ship | water_vehicle |
| truck | road_vehicle |

### Level 3: Fine Class

This is the original CIFAR-10 class label.

---

## Model Architecture

The model is a CNN with shared feature extractor.

It uses:

- Conv2D layers
- BatchNormalization
- MaxPooling2D
- Dropout
- GlobalAveragePooling2D
- Dense layer
- 3 softmax output heads

The model learns one image feature representation, then it predicts all 3 levels.

---

## Output Model

After training, the notebook save the trained model as:

```text
CIFAR_10_tens.h5
```

This file contains the trained TensorFlow/Keras model.

You can load it again using:

```python
loaded_model = tf.keras.models.load_model("CIFAR_10_tens.h5")
```

---

## How to Run

Install required libraries:

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

Run the notebook:

```bash
jupyter notebook CIFAR10_Multi_Level_Classification_TensorFlow.ipynb
```

Then run all cells from top to bottom.

---

## Expected Result

After running the notebook, you will get:

- sample CIFAR-10 image preview
- model training accuracy and validation accuracy
- testing evaluation result
- classification report for 3 levels
- prediction sample images
- saved model file `CIFAR_10_tens.h5`

---

## Important Note

Training result can be different per computer.  
If epoch is small, accuracy may not be very high.  
For better accuracy, increase `EPOCHS` from 10 to 15 or 20.

---

## Short Conclusion

This project show how CIFAR-10 can be classified in more meaningful way.  
Instead of only predicting 10 classes, the model also understand bigger category and group category.  
This is useful because real image system may need simple and detailed prediction together.
