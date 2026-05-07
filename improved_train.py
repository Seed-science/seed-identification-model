
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = matrix.max() / 2 if matrix.size and matrix.max() > 0 else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if matrix[row, col] > threshold else "black"
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color=color)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


# =========================
# EVALUATION
# =========================
def evaluate(model, dataset, save_path):
    y_true = []
    y_pred = []
    confidences = []

    for images, labels in dataset:
        probabilities = model.predict(images, verbose=0)
        predictions = np.argmax(probabilities, axis=1)
        confidence = np.max(probabilities, axis=1)

        y_true.extend(labels.numpy().tolist())
        y_pred.extend(predictions.tolist())
        confidences.extend(confidence.tolist())

    labels = list(range(NUM_CLASSES))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True,
    )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(report_dict["macro avg"]["precision"]),
        "recall_macro": float(report_dict["macro avg"]["recall"]),
        "f1_macro": float(report_dict["macro avg"]["f1-score"]),
        "precision_weighted": float(report_dict["weighted avg"]["precision"]),
        "recall_weighted": float(report_dict["weighted avg"]["recall"]),
        "f1_weighted": float(report_dict["weighted avg"]["f1-score"]),
    }

    print("\nConfusion Matrix:\n", matrix)
    print("\nClassification Report:\n", report_text)

    with open(save_path / "metrics.json", "w") as file:
        json.dump(metrics, file, indent=2)

    with open(save_path / "classification_report.txt", "w") as file:
        file.write(report_text)

    with open(save_path / "classification_report.json", "w") as file:
        json.dump(report_dict, file, indent=2)

    with open(save_path / "classification_report.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["class", "precision", "recall", "f1_score", "support"])

        for key, value in report_dict.items():
            if isinstance(value, dict):
                writer.writerow(
                    [
                        key,
                        value.get("precision", ""),
                        value.get("recall", ""),
                        value.get("f1-score", ""),
                        value.get("support", ""),
                    ]
                )

    with open(save_path / "confusion_matrix.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["true/predicted", *CLASS_NAMES])
        for class_name, row in zip(CLASS_NAMES, matrix.tolist()):
            writer.writerow([class_name, *row])

    with open(save_path / "predictions.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["row", "true_label", "predicted_label", "confidence"])

        for index, (true_label, predicted_label, confidence) in enumerate(
            zip(y_true, y_pred, confidences),
            start=1,
        ):
            writer.writerow(
                [
                    index,
                    CLASS_NAMES[int(true_label)],
                    CLASS_NAMES[int(predicted_label)],
                    float(confidence),
                ]
            )

    save_confusion_matrix_plot(matrix, save_path / "confusion_matrix.png")
    return metrics


# =========================
# TRAINING HELPERS
# =========================
def make_callbacks(model_dir, stage):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_dir / f"best_{stage}.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-7,
        ),
        tf.keras.callbacks.CSVLogger(
            filename=str(model_dir / f"{stage}_training_log.csv"),
            append=False,
        ),
    ]


def freeze_batch_norm_layers(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False


def choose_best_checkpoint(model_dir):
    candidates = [
        model_dir / "best_initial.keras",
        model_dir / "best_fine_tuned.keras",
    ]

    best_path = None
    best_accuracy = -1.0

    for path in candidates:
        if not path.exists():
            continue

        candidate = tf.keras.models.load_model(path)
        loss, accuracy = candidate.evaluate(val_data, verbose=0)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_path = path

    if best_path is None:
        raise FileNotFoundError("No checkpoint was saved.")

    print(f"Best validation checkpoint: {best_path.name} with accuracy {best_accuracy:.4f}")
    return best_path


def train_one_model(model_name):
    print(f"\nTraining model: {model_name}")

    model_dir = OUTPUT_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    model, base_model = build_model(model_name)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("\nInitial training...")
    history_initial = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS_INITIAL,
        class_weight=class_weights,
        callbacks=make_callbacks(model_dir, "initial"),
    )

    plot_history(history_initial, model_dir / "initial_training.png", "Initial Training")

    print("\nFine-tuning...")

    base_model.trainable = True

    for layer in base_model.layers[:-FINE_TUNE_LAYERS]:
        layer.trainable = False

    freeze_batch_norm_layers(base_model)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history_fine = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS_FINE,
        class_weight=class_weights,
        callbacks=make_callbacks(model_dir, "fine_tuned"),
    )

    plot_history(history_fine, model_dir / "fine_tuning.png", "Fine Tuning")

    print("\nLoading best checkpoint...")
    best_checkpoint = choose_best_checkpoint(model_dir)
    best_model = tf.keras.models.load_model(best_checkpoint)

    print("\nEvaluating final model...")
    metrics = evaluate(best_model, test_data, model_dir)

    best_model.save(model_dir / "final_model.keras")

    summary = {
        "model_name": model_name,
        "classes": CLASS_NAMES,
        "best_checkpoint": best_checkpoint.name,
        "metrics": metrics,
    }

    with open(model_dir / "summary.json", "w") as file:
        json.dump(summary, file, indent=2)

    print(f"\nResults for {model_name}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision weighted: {metrics['precision_weighted']:.4f}")