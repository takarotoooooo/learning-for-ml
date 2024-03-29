import tensorflow as tf


class MatrixFactorizationModel(tf.keras.Model):
    """
    MatrixFactorizationモデル

    Attributes
    ----------
    user_embeddings_layer : tf.keras.layers.Embedding
        ユーザーの潜在因子行列層
    item_embeddings_layer : tf.keras.layers.Embedding
        アイテムの潜在因子行列層
    """

    def __init__(self, user_size: int, item_size: int, embeddings_dim: int = 5):
        super(MatrixFactorizationModel, self).__init__()
        self.user_embeddings_layer = tf.keras.layers.Embedding(
            input_dim=user_size,
            output_dim=self.embeddings_dim,
            mask_zero=False,
            name='user_embeddings_layer',
            embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=0.01, seed=None))
        self.item_embeddings_layer = tf.keras.layers.Embedding(
            input_dim=item_size,
            output_dim=self.embeddings_dim,
            mask_zero=False,
            name='item_embeddings_layer',
            embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=0.01, seed=None))
        self.user_bias_layer = tf.keras.layers.Embedding(
            input_dim=user_size,
            output_dim=1,
            mask_zero=False,
            name='user_bias_layer',
            embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=0.01, seed=None))
        self.item_bias_layer = tf.keras.layers.Embedding(
            input_dim=item_size,
            output_dim=1,
            mask_zero=False,
            name='item_bias_layer',
            embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=0.01, seed=None))

    def call(self, inputs):
        user_embeddings = self.user_embeddings_layer(inputs[0])
        item_embeddings = self.item_embeddings_layer(inputs[1])
        user_bias = self.user_bias_layer(inputs[0])
        item_bias = self.item_bias_layer(inputs[1])

        x = tf.keras.layers.Dot(axes=2)([user_embeddings, item_embeddings])
        x = tf.keras.layers.Add()([x, user_bias, item_bias])
        return tf.keras.layers.Flatten()(x)


class MatrixFactorization():
    def __init__(self):
        pass

    def generate_model(
            self,
            user_num: int,
            item_num: int,
            dimension: int = 10) -> tf.keras.models.Model:
        user_input_layer = tf.keras.layers.Input(shape=(1, ))
        item_input_layer = tf.keras.layers.Input(shape=(1, ))
        user_embedding_layer = tf.keras.layers.Embedding(user_num, dimension)(user_input_layer)
        item_embedding_layer = tf.keras.layers.Embedding(item_num, dimension)(item_input_layer)
        user_bias_layer = tf.keras.layers.Embedding(user_num, 1)(user_input_layer)
        item_bias_layer = tf.keras.layers.Embedding(item_num, 1)(item_input_layer)

        x = tf.keras.layers.Dot(axes=2)([user_embedding_layer, item_embedding_layer])
        x = tf.keras.layers.Add()([x, user_bias_layer, item_bias_layer])
        x = tf.keras.layers.Flatten()(x)

        model = tf.keras.models.Model(inputs=[user_input_layer, item_input_layer], outputs=x)
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)

        model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=optimizer,
            metrics=[tf.keras.metrics.RootMeanSquaredError()])

        return model

    def train(
            self,
            model: tf.keras.models.Model,
            train_x: list,
            train_y: list,
            test_x: list,
            test_y: list) -> tf.keras.models.Model:
        # callback関数
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=0)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            'keras.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            mode='min',
            verbose=0)

        # 学習実行
        model.fit(
            x=train_x,
            y=train_y,
            epochs=200,
            batch_size=1024,
            validation_data=(test_x, test_y),
            callbacks=[early_stopping, checkpoint],
            verbose=1)

        return model

    def load_model(self, model_path: str) -> tf.keras.models.Model:
        return tf.keras.models.load_model(model_path, compile=False)

    def predict(
            self,
            model: tf.keras.models.Model,
            user_idx: list,
            item_ids: list):

        return model.predict([user_idx, item_ids], verbose=1)
