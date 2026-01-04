import tensorflow as tf
import keras
import typing as t

tf.config.optimizer.set_jit(False)

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class MultiModalModel(tf.keras.Model):
    
    def __init__(
        self,
        num_classes: int,
        learning_rates: t.List[float],
        # img args
        img_num_layers: int,
        img_num_filters: t.List[int],
        img_kernel_size: int,
        img_dropout_rates: t.List[float],
        img_activation: str,
        # tabular args
        tab_num_layers: int,
        tab_hidden_layer_size: t.List[int],
        tab_dropout_rates: t.List[float],
        tab_activation: str,
        # multi args
        multi_num_layers: int,
        multi_hidden_layer_size: t.List[int],
        multi_dropout_rates: t.List[float],
        multi_activation: str,
        early_stopping: bool,
    ):
        '''
        if img_num_layers != img_dropout_rates:
            raise ValueError("Number of image layers and given number of dropout rates must match.")
        if tab_num_layers != tab_dropout_rates:
            raise ValueError("Number of tabular layers and given number of dropout rates must match.")
        if multi_num_layers != multi_dropout_rates:
            raise ValueError("Number of multimodal layers and given number of dropout rates must match.")
        '''
        self.num_classes = num_classes
        self.learning_rates = learning_rates
        
        self.img_num_layers = img_num_layers
        self.img_num_filters = img_num_filters
        self.img_kernel_size = img_kernel_size
        self.img_dropout_rates = img_dropout_rates
        self.img_activation = img_activation
        
        self.tab_num_layers = tab_num_layers
        self.tab_hidden_layer_size = tab_hidden_layer_size
        self.tab_dropout_rates = tab_dropout_rates
        self.tab_activation = tab_activation
        
        self.multi_num_layers = multi_num_layers
        self.multi_hidden_layer_size = multi_hidden_layer_size
        self.multi_dropout_rates = multi_dropout_rates
        self.multi_activation = multi_activation
        self.early_stopping = early_stopping   
        
        self.img_branch = None
        self.tab_branch = None
        self.compiled_model = None
        
    def build_image_branch(self, hp):
        image_input = keras.Input(shape=(224, 224, 3), name="image")
        x = image_input

        img_dropout = hp.Float("img_dropout", *self.img_dropout_rates, step=0.1)
        img_base_filters = hp.Choice("base_filters", self.img_num_filters)

        for layer_i in range(1, self.img_num_layers + 1):
            x = keras.layers.Conv2D(
                filters=min(img_base_filters * layer_i, 256),
                kernel_size=(self.img_kernel_size, self.img_kernel_size),
                padding="same",
                activation=self.img_activation
            )(x)
            x = keras.layers.Dropout(img_dropout)(x)
            x = keras.layers.MaxPooling2D()(x)

        x = keras.layers.GlobalAveragePooling2D()(x)

        self.image_input = image_input     # ✅ store input
        self.img_branch = x                # ✅ store output
    
    def build_tabular_branch(self, hp):
        metadata_input = keras.Input(
            shape=(19,),
            name="metadata"
        )
        x = metadata_input

        tab_dropout = hp.Float("tab_dropout", *self.tab_dropout_rates, step=0.1)
        tab_base_size = hp.Choice("base_hidden_layer_size", self.tab_hidden_layer_size)
        tab_l2_reg = tf.keras.regularizers.l2(
            hp.Float("tab_l2", 1e-5, 1e-3, sampling="log")
        )

        for layer_i in range(self.tab_num_layers, 0, -1):
            x = keras.layers.Dense(
                units=min(tab_base_size * layer_i, 512),
                activation=self.tab_activation,
                kernel_regularizer=tab_l2_reg,
            )(x)
            x = keras.layers.Dropout(tab_dropout)(x)

        self.metadata_input = metadata_input
        self.tab_branch = x
        
    def build_multimodal_branch(self, hp):
        combined = keras.layers.Concatenate()(
            [self.img_branch, self.tab_branch]
        )

        multi_dropout = hp.Float("multi_dropout", *self.multi_dropout_rates, step=0.1)
        multi_hidden_layers_size = hp.Choice("combined_dense", self.multi_hidden_layer_size)
        multi_l2_reg = tf.keras.regularizers.l2(
            hp.Float("multi_l2", 1e-5, 1e-3, sampling="log")
        )

        for layer_i in range(self.multi_num_layers, 0, -1):
            combined = keras.layers.Dense(
                units=multi_hidden_layers_size * layer_i,
                activation=self.multi_activation,
                kernel_regularizer=multi_l2_reg,
            )(combined)
            combined = keras.layers.Dropout(multi_dropout)(combined)

        output = keras.layers.Dense(
            self.num_classes,
            activation="softmax"
        )(combined)

        model = keras.Model(
            inputs={
                "image": self.image_input,
                "metadata": self.metadata_input,
            },
            outputs=output,
        )

        lr = hp.Float("lr", *self.learning_rates, sampling="log")
        # Use gradient clipping to prevent exploding gradients
        optimizer = keras.optimizers.Adam(lr, clipvalue=1.0)
        model.compile(
            optimizer=optimizer,
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=["accuracy"],
        )

        self.compiled_model = model
        
    def build_model(self, hp):
        
        if self.img_branch is None:
            self.build_image_branch(
                hp
            )
        if self.tab_branch is None:
            self.build_tabular_branch(
                hp
            )
        
        self.build_multimodal_branch(
            hp
        )
        
        return self.compiled_model