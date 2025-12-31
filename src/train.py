import argparse
import sys
from pathlib import Path

from sklearn.preprocessing import OrdinalEncoder

# Add the project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.data_processing.img_processing import (
    read_img_paths_to_train_val_set,
)

from src.data_processing.tabular_processing import (
    pre_process_metadata
)

from src.data_processing.multimodal_processing import (
    get_top_labels,
    filter_by_labels_1,
    make_supervised_ds
)

from src.models.tune import configure_model_tuner, extract_best_config


def main(args):
    
    # model_cfg_path = args.model_config
    # train_cfg_path = args.training_config
    data_cfg_path  = args.data_config

    # Read in TRAIN and VAL image paths and labels
    # --------------------------------------------------------------
    train_img_pths, train_img_labels, val_img_pths, val_img_labels = read_img_paths_to_train_val_set(
        cfg_path=data_cfg_path
    )
    
    top_labels = get_top_labels(labels=train_img_labels, top_n=6)
    
    # Read in TRAIN and VAL tab. metadata
    # --------------------------------------------------------------
    train_metadata_df, val_metadata_df = pre_process_metadata(
        cfg_path=data_cfg_path
    )
    
    str_cols = train_metadata_df.select_dtypes(include=["object", "string"]).columns.tolist()
    
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        encoded_missing_value=-1,
        unknown_value=-1,
        dtype=int,
    )

    train_metadata_np = train_metadata_df.copy()
    len_metadata_input = len(train_metadata_np)
    print(f"{len_metadata_input=}")
    train_metadata_np[str_cols] = encoder.fit_transform(train_metadata_np[str_cols])

    val_metadata_np = val_metadata_df.copy()
    val_metadata_np[str_cols] = encoder.transform(val_metadata_np[str_cols])

    print(f"{val_metadata_np.shape=}")
    print(f"{train_metadata_np.shape=}")
    print(f"{len(train_img_pths)=}")
    print(f"{len(val_img_pths)=}")
    
    train_filtered_paths, train_filtered_meta_dataset, train_filtered_labels = filter_by_labels_1(
        paths=train_img_pths,
        labels=train_img_labels,
        metadata=train_metadata_np,
        allowed_labels=top_labels
    )

    val_filtered_paths, val_filtered_meta_dataset, val_filtered_labels = filter_by_labels_1(
        paths=val_img_pths,
        labels=val_img_labels,
        metadata=val_metadata_np,
        allowed_labels=top_labels
    )
    
    # BUILD TRAIN and VAL sets of combined img and tabular data
    # --------------------------------------------------------------
    train_ds = make_supervised_ds(
        paths=train_filtered_paths,
        metadata=train_filtered_meta_dataset,
        ref_labels=top_labels,
        set_labels=train_filtered_labels,
        training=True,
    )

    val_ds = make_supervised_ds(
        paths=val_filtered_paths,
        metadata=val_filtered_meta_dataset,
        ref_labels=top_labels,
        set_labels=val_filtered_labels,
        training=False,
    )
    
    tuner_ = configure_model_tuner(
        cfg_path='configs/base_model.yaml',
        tabular_input_size=len_metadata_input,
        train_dataset=train_ds,
        val_dataset=val_ds,
        epochs=10,
    )
    best_cfg = extract_best_config(
        tuner=tuner_, 
        input_cfg_path='configs/base_model.yaml', 
        output_cfg_path='configs/best_model.yaml'
    )

    '''
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=train_cfg["training"]["optimizer"]["lr"]
    )

    model.compile(
        optimizer=optimizer,
        loss=train_cfg["training"]["loss"],
        metrics=["accuracy"]
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=train_cfg["training"]["epochs"]
    )

    model.save(args.output_dir)
    '''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model-config", required=True)
    # parser.add_argument("--training-config", required=True)
    parser.add_argument("--data-config", required=True)
    # parser.add_argument("--output-dir", default="artifacts/model")

    args = parser.parse_args()
    main(args)