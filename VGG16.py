import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Reproducibility
def set_seed(seed=67):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)


set_seed(67)


# Data loading and preprocessing
def load_data(base_path, target_size=(224, 224)):
    """
        Load EEG spectrograms and prepare them for CNN classification.

        Each spectrogram inherits the label of its class directory.
        Patient IDs are stored separately to enable patient-level
        cross-validation and prevent data leakage between folds.

        Returns:
            X: Array of preprocessed spectrograms.
            y: Array of binary class labels.
            groups: Array of patient IDs used for grouped cross-validation.
    """
    X, y, groups = [], [], []

    # Iterate through class directories
    for class_folder in os.listdir(base_path):
        class_path = os.path.join(base_path, class_folder)
        if not os.path.isdir(class_path):
            continue

        try:
            label = int(class_folder)
        except ValueError:
            continue

        # Select the recording condition used in the experiment
        # 'EO' - eyes-open or 'EC' - eyes-closed condition
        for condition_folder in os.listdir(class_path):
            if condition_folder != 'EC':
                continue

            condition_path = os.path.join(class_path, condition_folder)

            for patient_folder in os.listdir(condition_path):
                patient_path = os.path.join(condition_path, patient_folder)

                if not os.path.isdir(patient_path):
                    continue

                # All segments belonging to one patient share the same ID
                patient_id = patient_folder

                for file_name in os.listdir(patient_path):
                    if not file_name.lower().endswith(('.png')):
                        continue

                    file_path = os.path.join(patient_path, file_name)

                    # Resize each image and convert it to a three-channel array
                    img = load_img(file_path, target_size=target_size, color_mode='rgb')
                    img_array = img_to_array(img)

                    # Apply logarithmic transformation followed by per-image standardization
                    img_array = preprocess_input(img_array)
                    img_array = (img_array - np.mean(img_array)) / (np.std(img_array) + 1e-8)

                    X.append(img_array)
                    y.append(label)
                    groups.append(patient_id)

    return np.array(X), np.array(y), np.array(groups)


# VGG16
def build_vgg(input_shape=(224, 224, 3)):
    """
    Build a VGG16-based convolutional neural network.

    The network uses a pretrained VGG16 architecture for
    hierarchical feature extraction through stacked convolutional
    and pooling layers, followed by global average pooling,
    dropout, and a fully connected classification layer.

    The sigmoid output produces a probability for binary
    classification.
    """
    # Load pretrained model without its original classification layer
    # Use ImageNet weights to initialize the network
    base = VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    # Allow pretrained InceptionV3 layers to be updated during training
    base.trainable = True

    # Build final classification model
    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.6),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    return model


# Model training and cross-validation
if __name__ == "__main__":

    # Path to the directory containing dataset
    DATA_DIR = "/mnt/c/Users/Enrique/Desktop/Studia/magisterka/praca magisterska/10s SEGMENTS 4 kanały/spektrogramy"

    # Load spectrograms, binary labels, and patient group identifiers
    X, y, groups = load_data(DATA_DIR)

    # Five-fold cross-validation with patient-level separation
    gkf = GroupKFold(n_splits=5)

    all_true = []
    all_pred = []
    fold_accs = []

    # Collect the final prediction results
    global_patient_results = {}


    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):

        print(f"\n========== FOLD {fold} ==========")

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        groups_test = groups[test_idx]

        # Create a new model
        model = build_vgg()

        # Using Adam optimizer
        model.compile(
            optimizer=Adam(1e-5),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # Train CNN
        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50,
            batch_size=16,
            callbacks=[
                # Stop training when validation loss stops improving
                EarlyStopping(patience=6, restore_best_weights=True),
                # Reduce the learning rate when validation loss plateaus
                ReduceLROnPlateau(patience=3, factor=0.5)
            ],
            verbose=1
        )

        # Obtain predicted class probabilities for individual segments
        probs = model.predict(X_test).flatten()

        # Aggregate segment predictions
        patient_probs = defaultdict(list)
        patient_labels = {}

        # Collect all segment probabilities belonging to each patient
        for i, pid in enumerate(groups_test):
            patient_probs[pid].append(probs[i])

            # Every segment from a patient should have the same label
            if pid not in patient_labels:
                patient_labels[pid] = y_test[i]

        y_true_patients = []
        y_pred_patients = []


        for pid in patient_probs:
            # Use the mean segment probability as the patient-level score
            mean_prob = np.mean(patient_probs[pid])
            pred = int(mean_prob > 0.5)

            y_true_patients.append(patient_labels[pid])
            y_pred_patients.append(pred)

            # Calculate segment-level accuracy for this patient
            segment_preds = [int(p > 0.5) for p in patient_probs[pid]]
            true_label = patient_labels[pid]
            correct_segments = sum(1 for p_seg in segment_preds if p_seg == true_label)
            total_segments = len(segment_preds)

            patient_acc = correct_segments / total_segments

            # Save the patient's ground-truth label, aggregated prediction and proportion of correctly classified segments
            global_patient_results[pid] = {
                'true_label': true_label,
                'predicted_label': pred,
                'accuracy': patient_acc,
                'correct_segments': correct_segments,
                'total_segments': total_segments
            }

        # Evaluate the current fold
        # Confusion matrix layout: [[TN, FP], [FN, TP]]
        cm = confusion_matrix(y_true_patients, y_pred_patients, labels=[0, 1])

        TN, FP, FN, TP = cm.ravel()

        accuracy = (TP + TN) / (TP + TN + FP + FN)

        specificity = TN / (TN + FP)
        precision = precision_score(y_true_patients, y_pred_patients, zero_division=0)
        recall = recall_score(y_true_patients, y_pred_patients, zero_division=0)
        f1 = f1_score(y_true_patients, y_pred_patients, zero_division=0)

        fold_accs.append(accuracy)
        # Accumulate patient-level results for the overall evaluation
        all_true.extend(y_true_patients)
        all_pred.extend(y_pred_patients)

        print(f"Accuracy:   {accuracy:.4f}")
        print(f"Specificity:{specificity:.4f}")
        print(f"Precision:  {precision:.4f}")
        print(f"Recall:     {recall:.4f}")
        print(f"F1-score:   {f1:.4f}")

    # Final evaluation
    print("\n========== Results ==========")

    # Combine patient-level predictions from all folds
    cm = confusion_matrix(all_true, all_pred, labels=[0, 1])

    TN, FP, FN, TP = cm.ravel()

    accuracy = (TP + TN) / (TP + TN + FP + FN)

    specificity = TN / (TN + FP)
    precision = precision_score(all_true, all_pred, zero_division=0)
    recall = recall_score(all_true, all_pred, zero_division=0)
    f1 = f1_score(all_true, all_pred, zero_division=0)

    print(f"\nAccuracy:   {accuracy:.4f}")
    print(f"Specificity:{specificity:.4f}")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"F1-score:   {f1:.4f}")

    print("\nRaport:")
    print(classification_report(all_true, all_pred, target_names=["Control", "ADHD"], zero_division=0))

    # Patient level results
    print("Patient-Level Results")
    print(f"{'Patient ID':<15} | {'True label':<15} | {'Predicted':<15} | {'Segment accuracy':<25}")

    for pid, data in sorted(global_patient_results.items()):
        true_name = "1" if data['true_label'] == 1 else "0"
        pred_name = "1" if data['predicted_label'] == 1 else "0"
        acc_pct = data['accuracy']

        print(f"{pid:<15} | {true_name:<15} | {pred_name:<15} | {acc_pct:>5.2f}")

    # Confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Control", "ADHD"],
                yticklabels=["Control", "ADHD"])
    plt.title("Confusion matrix- VGG16")
    plt.xlabel("Predicted")
    plt.ylabel("Real")
    plt.show()

    model.save("VGG16_4ch_EC.keras")