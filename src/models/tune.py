import keras_tuner as kt
import keras

from src.utils import load_yaml, save_yaml
from src.models.multimodal_cnn import MultiModalModel


def configure_model_tuner(
    cfg_path,
    tabular_input_size,
    train_dataset,
    val_dataset,
    epochs,
):
    cfg = load_yaml(cfg_path)

    if cfg['training']['early_stopping']:
        # define early stopping class
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            verbose=1,
            patience=5,
            mode='max',
            restore_best_weights=True
        )
    else:
        early_stopping=None
        
    def build_model(hp):
        
        model_wrapper = MultiModalModel(
            num_classes=cfg["training"]["num_classes"],
            learning_rates=cfg["training"]["learning_rates"],

            img_num_layers=cfg["image_branch"]["num_layers"],
            img_num_filters=cfg["image_branch"]["filters"],
            img_kernel_size=cfg["image_branch"]["kernel_size"],
            img_dropout_rates=cfg["image_branch"]["dropout_rates"],
            img_activation=cfg["image_branch"]["activation"],

            tab_num_layers=cfg["tabular_branch"]["num_layers"],
            tab_hidden_layer_size=cfg["tabular_branch"]["hidden_units"],
            tab_dropout_rates=cfg["tabular_branch"]["dropout_rates"],
            tab_activation=cfg["tabular_branch"]["activation"],

            multi_num_layers=cfg["multimodal_branch"]["num_layers"],
            multi_hidden_layer_size=cfg["multimodal_branch"]["hidden_units"],
            multi_dropout_rates=cfg["multimodal_branch"]["dropout_rates"],
            multi_activation=cfg["multimodal_branch"]["activation"],
            early_stopping=cfg['training']['early_stopping']
        )

        return model_wrapper.build_model(hp)

    tuner = kt.Hyperband(
        hypermodel=build_model,
        objective=kt.Objective("val_accuracy", direction="max"),
        max_epochs=epochs,
        factor=3,
        hyperband_iterations=2,   # repeats the full process
        directory="tuning",
        project_name="hyperband_fungi",
        overwrite=True,
    )

    tuner.search(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[early_stopping],
    )

    return tuner


def extract_best_config(tuner, input_cfg_path, output_cfg_path="best_model.yaml"):
    """
    Build a full config YAML from scratch using the best hyperparameters
    found by a Keras Tuner object.
    """
    base_cfg = load_yaml(input_cfg_path)
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

    # ---------------- Image branch ----------------
    img_layers = base_cfg["image_branch"]["num_layers"]
    img_dropout = best_hp.get("img_dropout")
    img_base_filters = best_hp.get("base_filters")

    image_branch = {
        "num_layers": img_layers,
        "filters": [
            img_base_filters * (i + 1) for i in range(img_layers)
        ],
        "kernel_size": base_cfg["image_branch"]["kernel_size"],
        "dropout_rates": [img_dropout] * img_layers,
        "activation": base_cfg["image_branch"]["activation"],
    }

    # ---------------- Tabular branch ----------------
    tab_layers = base_cfg["tabular_branch"]["num_layers"]
    tab_dropout = best_hp.get("tab_dropout")
    tab_base_units = best_hp.get("base_hidden_layer_size")

    tabular_branch = {
        "num_layers": tab_layers,
        "hidden_units": [
            tab_base_units * (i + 1) for i in range(tab_layers, 0, -1)
        ],
        "dropout_rates": [tab_dropout] * tab_layers,
        "activation": base_cfg["tabular_branch"]["activation"],
        "l2": best_hp.get("tab_l2"),
    }

    # ---------------- Multimodal branch ----------------
    multi_layers = base_cfg["multimodal_branch"]["num_layers"]
    multi_dropout = best_hp.get("multi_dropout")
    multi_base_units = best_hp.get("combined_dense1")

    multimodal_branch = {
        "num_layers": multi_layers,
        "hidden_units": [
            multi_base_units * (i + 1) for i in range(multi_layers, 0, -1)
        ],
        "dropout_rates": [multi_dropout] * multi_layers,
        "activation": base_cfg["multimodal_branch"]["activation"],
        "l2": best_hp.get("multi_l2"),
    }

    # ---------------- Training ----------------
    training = {
        "num_classes": base_cfg["training"]["num_classes"],
        "learning_rate": best_hp.get("lr"),
        "batch_size": base_cfg["training"]["batch_size"],
        "optimizer": "adam",
    }

    cfg = {
        "image_branch": image_branch,
        "tabular_branch": tabular_branch,
        "multimodal_branch": multimodal_branch,
        "training": training,
    }

    save_yaml(cfg, output_cfg_path)
    return cfg