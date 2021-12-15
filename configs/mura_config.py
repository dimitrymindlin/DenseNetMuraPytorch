mura_config = {
    "data": {
        "study_types": ['XR_WRIST', 'XR_HAND', 'XR_FOREARM'],
        "class_names": ["negative", "positive"],
        "image_dimension": (224, 224, 3),
        "image_height": 224,
        "image_width": 224,
        "image_channel": 3,
    },
    "train": {
        "augmentation": True,
        "batch_size": 1,
        "learn_rate": 0.0001,
        "epochs": 3,
        "patience_learning_rate": 2,
        "min_learning_rate": 1e-8,
        "early_stopping_patience": 8
    },
    "test": {
        "batch_size": 64,
        "F1_threshold": 0.5,
    },
    "model": {
        "pretrained": True,
        "pooling": "avg",
    }
}