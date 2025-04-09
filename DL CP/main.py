import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kaggle
import zipfile
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import random
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

def download_dataset():
    """
    Download the Devanagari character dataset from Kaggle
    """
    dataset_name = "ashokpant/devanagari-character-dataset-large"
    data_dir = "devanagari_data"

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

        print(f"Downloading dataset {dataset_name}...")
        kaggle.api.dataset_download_files(dataset_name, path=data_dir)

        zip_files = [f for f in os.listdir(data_dir) if f.endswith('.zip')]
        for zip_file in zip_files:
            zip_path = os.path.join(data_dir, zip_file)
            print(f"Extracting {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)

        print("Dataset downloaded and extracted successfully!")
    else:
        print(f"Directory {data_dir} already exists. Using existing data.")

    return data_dir

def load_data(data_dir):
    """
    Load and preprocess the Devanagari character dataset
    """
    train_dir = os.path.join(data_dir, 'dhcd', 'train')
    test_dir = os.path.join(data_dir, 'dhcd', 'test')

    X_train = []
    y_train = []
    X_test = []
    y_test = []
    class_mapping = {}
    class_counts = {}

    print("Loading training data...")
    classes = sorted(os.listdir(train_dir))
    for i, class_folder in enumerate(classes):
        class_mapping[i] = class_folder
        class_path = os.path.join(train_dir, class_folder)
        if os.path.isdir(class_path):
            class_images = 0
            for image_file in os.listdir(class_path):
                img_path = os.path.join(class_path, image_file)
                try:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (32, 32))
                    img = img / 255.0
                    X_train.append(img)
                    y_train.append(i)
                    class_images += 1
                except Exception as e:
                    print(f"Error loading {img_path}: {str(e)}")
            class_counts[class_folder] = class_images

    print("Loading test data...")
    test_class_counts = {}
    for i, class_folder in enumerate(classes):
        class_path = os.path.join(test_dir, class_folder)
        test_class_counts[class_folder] = 0
        if os.path.isdir(class_path):
            for image_file in os.listdir(class_path):
                img_path = os.path.join(class_path, image_file)
                try:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (32, 32))
                    img = img / 255.0
                    X_test.append(img)
                    y_test.append(i)
                    test_class_counts[class_folder] += 1
                except Exception as e:
                    print(f"Error loading {img_path}: {str(e)}")

    X_train = np.array(X_train).reshape(-1, 32, 32, 1)
    X_test = np.array(X_test).reshape(-1, 32, 32, 1)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    num_classes = len(classes)
    y_train_categorical = to_categorical(y_train, num_classes)
    y_test_categorical = to_categorical(y_test, num_classes)

    print(f"Data loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    return X_train, y_train_categorical, X_test, y_test_categorical, y_train, y_test, class_mapping, num_classes, class_counts, test_class_counts

def explore_data(X_train, y_train, class_mapping, class_counts, test_class_counts):
    """
    Perform exploratory data analysis on the dataset
    """
    print("\n=== Exploratory Data Analysis ===")

    print(f"Training dataset shape: {X_train.shape}")
    print(f"Number of classes: {len(class_mapping)}")
    print(f"Image dimensions: {X_train[0].shape}")
    print(f"Min pixel value: {X_train.min()}, Max pixel value: {X_train.max()}")

    plt.figure(figsize=(15, 6))

    sorted_classes = sorted(class_counts.items(), key=lambda x: x[0])
    class_names = [x[0] for x in sorted_classes]
    class_samples = [x[1] for x in sorted_classes]

    plt.bar(class_names, class_samples)
    plt.xticks(rotation=90, fontsize=10)
    plt.title('Number of Training Samples per Class')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.tight_layout()
    plt.savefig('class_distribution_train.png')
    plt.show()

    plt.figure(figsize=(15, 6))
    sorted_test_classes = sorted(test_class_counts.items(), key=lambda x: x[0])
    test_class_names = [x[0] for x in sorted_test_classes]
    test_class_samples = [x[1] for x in sorted_test_classes]

    plt.bar(test_class_names, test_class_samples)
    plt.xticks(rotation=90, fontsize=10)
    plt.title('Number of Test Samples per Class')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.tight_layout()
    plt.savefig('class_distribution_test.png')
    plt.show()

    plt.figure(figsize=(12, 12))
    for i in range(min(25, len(class_mapping))):
        indices = np.where(y_train == i)[0]
        if len(indices) > 0:
            idx = np.random.choice(indices)
            plt.subplot(5, 5, i+1)
            plt.imshow(X_train[idx].reshape(32, 32), cmap='gray')
            plt.title(class_mapping[i])
            plt.axis('off')
    plt.tight_layout()
    plt.savefig('sample_images.png')
    plt.show()

    plt.figure(figsize=(10, 6))
    sample_images = X_train[:1000].flatten()
    plt.hist(sample_images, bins=50)
    plt.title('Pixel Intensity Distribution')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.savefig('pixel_distribution.png')
    plt.show()

    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.1
    )

    idx = np.random.choice(len(X_train))
    img = X_train[idx].reshape(1, 32, 32, 1)

    plt.figure(figsize=(12, 8))
    plt.subplot(3, 3, 1)
    plt.imshow(img.reshape(32, 32), cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    i = 2
    for batch in datagen.flow(img, batch_size=1):
        plt.subplot(3, 3, i)
        plt.imshow(batch[0].reshape(32, 32), cmap='gray')
        plt.title(f'Augmented #{i-1}')
        plt.axis('off')
        i += 1
        if i > 9:
            break

    plt.tight_layout()
    plt.savefig('augmented_samples.png')
    plt.show()

def build_model(num_classes):
    """
    Build and compile the CNN model
    """
    model = Sequential([
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 1)),
        BatchNormalization(),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def plot_history(history):
    """
    Plot training and validation accuracy/loss curves
    """
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

    val_acc = max(history.history['val_accuracy'])
    val_loss = min(history.history['val_loss'])
    print(f"Best validation accuracy: {val_acc:.4f}")
    print(f"Best validation loss: {val_loss:.4f}")
    print(f"Number of epochs trained: {len(history.history['val_accuracy'])}")

def evaluate_performance(model, X_test, y_test_categorical, y_test, class_mapping):
    """
    Evaluate model performance with various metrics and visualizations
    """
    print("\n=== Model Performance Evaluation ===")

    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = y_test

    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    f1_macro = f1_score(y_true, y_pred, average='macro')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (macro): {precision_macro:.4f}")
    print(f"Recall (macro): {recall_macro:.4f}")
    print(f"F1 Score (macro): {f1_macro:.4f}")

    class_names = [class_mapping[i] for i in range(len(class_mapping))]
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    conf_matrix = confusion_matrix(y_true, y_pred)

    cm_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(14, 12))
    sns.heatmap(conf_matrix, annot=False, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

    confusion_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and conf_matrix[i, j] > 0:
                confusion_pairs.append((i, j, conf_matrix[i, j]))

    confusion_pairs.sort(key=lambda x: x[2], reverse=True)
    print("\nTop-5 most confused classes:")
    for i, j, count in confusion_pairs[:5]:
        print(f"True: {class_mapping[i]}, Predicted: {class_mapping[j]}, Count: {count}")

    per_class_accuracy = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)

    accuracy_pairs = [(class_mapping[i], acc) for i, acc in enumerate(per_class_accuracy)]
    accuracy_pairs.sort(key=lambda x: x[1])

    plt.figure(figsize=(15, 8))
    classes = [x[0] for x in accuracy_pairs]
    accuracies = [x[1] for x in accuracy_pairs]
    plt.barh(classes, accuracies)
    plt.xlabel('Accuracy')
    plt.title('Per-Class Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('per_class_accuracy.png')
    plt.show()

    print("\nBest performing classes:")
    for class_name, acc in accuracy_pairs[-5:]:
        print(f"{class_name}: {acc:.4f}")

    print("\nWorst performing classes:")
    for class_name, acc in accuracy_pairs[:5]:
        print(f"{class_name}: {acc:.4f}")

    return y_pred, y_pred_prob

def visualize_image_processing_steps(X_samples, class_name, title):
    """
    Visualize the image processing steps applied to samples
    """
    plt.figure(figsize=(12, 8))
    plt.suptitle(f"{title} - {class_name}", fontsize=16)

    for i in range(min(10, len(X_samples))):
        plt.subplot(2, 5, i+1)
        plt.imshow(X_samples[i].reshape(32, 32), cmap='gray')
        plt.title(f"Sample {i+1}")
        plt.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'processed_samples_{title.replace(" ", "_")}.png')
    plt.show()

def visualize_predictions(X_test, y_true, y_pred, y_pred_prob, class_mapping, num_samples=10):
    """
    Visualize model predictions with confidence scores
    """
    correct_indices = np.where(y_true == y_pred)[0]
    incorrect_indices = np.where(y_true != y_pred)[0]

    n_correct = min(num_samples // 2, len(correct_indices))
    n_incorrect = min(num_samples - n_correct, len(incorrect_indices))

    selected_correct = np.random.choice(correct_indices, n_correct, replace=False)
    selected_incorrect = np.random.choice(incorrect_indices, n_incorrect, replace=False)

    indices = np.concatenate([selected_correct, selected_incorrect])
    np.random.shuffle(indices)

    plt.figure(figsize=(15, 8))
    for i, idx in enumerate(indices[:num_samples]):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X_test[idx].reshape(32, 32), cmap='gray')

        true_class = class_mapping[y_true[idx]]
        pred_class = class_mapping[y_pred[idx]]
        confidence = np.max(y_pred_prob[idx]) * 100

        title = f"True: {true_class}\nPred: {pred_class}\nConf: {confidence:.1f}%"
        plt.title(title, color='green' if true_class == pred_class else 'red', fontsize=9)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('prediction_samples.png')
    plt.show()

def visualize_misclassifications(X_test, y_true, y_pred, class_mapping):
    """
    Visualize examples of misclassified samples
    """
    misclassified = np.where(y_true != y_pred)[0]

    if len(misclassified) == 0:
        print("No misclassifications found!")
        return

    misclass_groups = {}
    for idx in misclassified:
        true_label = y_true[idx]
        pred_label = y_pred[idx]
        key = (true_label, pred_label)

        if key not in misclass_groups:
            misclass_groups[key] = []

        misclass_groups[key].append(idx)

    top_pairs = sorted(misclass_groups.items(), key=lambda x: len(x[1]), reverse=True)[:5]

    for pair, indices in top_pairs:
        true_label, pred_label = pair
        true_class = class_mapping[true_label]
        pred_class = class_mapping[pred_label]

        plt.figure(figsize=(12, 4))
        plt.suptitle(f"True: {true_class}, Predicted: {pred_class}", fontsize=14)

        for i, idx in enumerate(indices[:6]):  # Show up to 6 examples
            plt.subplot(1, 6, i+1)
            plt.imshow(X_test[idx].reshape(32, 32), cmap='gray')
            plt.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.9])  # Adjust for title
        plt.savefig(f'misclassification_{true_class}_as_{pred_class}.png')
        plt.show()

def main():
    print("Setting up Devanagari Character and Digit Recognition System...")

    data_dir = download_dataset()

    X_train, y_train_categorical, X_test, y_test_categorical, y_train, y_test, class_mapping, num_classes, class_counts, test_class_counts = load_data(data_dir)

    explore_data(X_train, y_train, class_mapping, class_counts, test_class_counts)

    X_train, X_val, y_train_cat, y_val_cat = train_test_split(X_train, y_train_categorical, test_size=0.1, random_state=42)

    print(f"\nTraining data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Number of classes: {num_classes}")

    model = build_model(num_classes)
    model.summary()

    checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001, verbose=1)

    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.1
    )
    datagen.fit(X_train)

    print("\nTraining the model...")
    history = model.fit(
        datagen.flow(X_train, y_train_cat, batch_size=64),
        epochs=15,
        validation_data=(X_val, y_val_cat),
        callbacks=[early_stopping, reduce_lr, checkpoint],
        verbose=1
    )

    plot_history(history)

    print("\nEvaluating the model on test data...")
    test_loss, test_acc = model.evaluate(X_test, y_test_categorical, verbose=1)
    print(f"Test accuracy: {test_acc*100:.2f}%")

    y_pred, y_pred_prob = evaluate_performance(model, X_test, y_test_categorical, y_test, class_mapping)

    visualize_predictions(X_test, y_test, y_pred, y_pred_prob, class_mapping, num_samples=10)

    visualize_misclassifications(X_test, y_test, y_pred, class_mapping)

    model.save('devanagari_recognition_model.h5')
    print("Model saved as 'devanagari_recognition_model.h5'")

    with open('class_mapping.txt', 'w') as f:
        for i, class_name in class_mapping.items():
            f.write(f"{i}: {class_name}\n")
    print("Class mapping saved as 'class_mapping.txt'")

    performance = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_macro': precision_score(y_test, y_pred, average='macro'),
        'recall_macro': recall_score(y_test, y_pred, average='macro'),
        'f1_macro': f1_score(y_test, y_pred, average='macro')
    }

    with open('performance_metrics.txt', 'w') as f:
        for metric, value in performance.items():
            f.write(f"{metric}: {value:.4f}\n")
    print("Performance metrics saved as 'performance_metrics.txt'")

if __name__ == "__main__":
    main()