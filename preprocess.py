import random
import cv2
import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

DATA_PATH = 'data/indoorCVPR_09'
LABEL_ENCODER_PATH = 'model/label_encoder.pkl'


def extractImages(datadir, target_size=(64, 64)):
    image_data = []
    image_labels = []

    if not os.path.exists(datadir):
        raise ValueError(f"Directory {datadir} does not exist")

    class_folders = [f for f in os.listdir(datadir) if os.path.isdir(os.path.join(datadir, f))]
    if not class_folders:
        raise ValueError(f"No valid class folders found in {datadir}")

    for folder in class_folders:
        path = os.path.join(datadir, folder)
        image_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

        for image_file in image_files:
            full_path = os.path.join(path, image_file)
            try:
                # Load in color (RGB)
                img = cv2.imread(full_path)
                if img is None:
                    print(f"Failed to load image: {full_path}. Deleting file.")
                    try:
                        os.remove(full_path)
                    except Exception as e:
                        print(f"Failed to delete {full_path}: {str(e)}")
                    continue

                # Convert BGR (OpenCV default) to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, target_size)
                image_data.append(img)
                image_labels.append(folder)
            except Exception as e:
                print(f"Error processing image {full_path}: {str(e)}. Deleting file.")
                try:
                    os.remove(full_path)
                except Exception as e:
                    print(f"Failed to delete {full_path}: {str(e)}")
                continue

    if not image_data:
        raise ValueError("No valid images were loaded from the dataset")

    # Shuffle data
    combined = list(zip(image_data, image_labels))
    random.shuffle(combined)
    image_data, image_labels = zip(*combined)

    return list(image_data), list(image_labels)


def get_train_data(target_size=(64, 64)):
    # Extract images
    try:
        image_data, image_labels = extractImages(DATA_PATH, target_size)
    except ValueError as e:
        raise e

    num_classes = len(set(image_labels))
    if num_classes < 2:
        raise ValueError("Need at least 2 classes for classification")

    # Train-test split with validation
    # First split: 80% train, 20% temp (val + test)
    train_data, temp_data, train_labels, temp_labels = train_test_split(
        image_data, image_labels,
        shuffle=True,
        test_size=0.2,
        random_state=42,
        stratify=image_labels
    )
    # Second split: 10% val, 10% test (from the 20% temp)
    val_data, test_data, val_labels, test_labels = train_test_split(
        temp_data, temp_labels,
        shuffle=True,
        test_size=0.5,
        random_state=42,
        stratify=temp_labels
    )

    # Encode labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels)
    y_val = label_encoder.transform(val_labels)
    y_test = label_encoder.transform(test_labels)

    # Convert to one-hot
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_val = to_categorical(y_val, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)

    # Convert to numpy arrays and normalize
    X_train = np.array(train_data, dtype=np.float32)
    X_val = np.array(val_data, dtype=np.float32)
    X_test = np.array(test_data, dtype=np.float32)

    # Add channel dimension and normalize
    X_train = X_train / 255.0
    X_val = X_val / 255.0
    X_test = X_test / 255.0

    # Data augmentation for training data
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen.fit(X_train)

    # Ensure model directory exists
    os.makedirs(os.path.dirname(LABEL_ENCODER_PATH), exist_ok=True)

    # Save label encoder
    with open(LABEL_ENCODER_PATH, 'wb') as file:
        pickle.dump(label_encoder, file)

    return X_train, y_train, X_val, y_val, X_test, y_test, num_classes, datagen


if __name__ == '__main__':
    try:
        X_train, y_train, X_val, y_val, X_test, y_test, num_classes, datagen = get_train_data()

        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"X_val shape: {X_val.shape}")
        print(f"y_val shape: {y_val.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_test shape: {y_test.shape}")
        print(f"Number of classes: {num_classes}")

        # Additional validation
        print(f"X_train min/max: {X_train.min():.2f}/{X_train.max():.2f}")
        print(f"X_val min/max: {X_val.min():.2f}/{X_val.max():.2f}")
        print(f"X_test min/max: {X_test.min():.2f}/{X_test.max():.2f}")

    except Exception as e:
        print(f"Error in data preprocessing: {str(e)}")