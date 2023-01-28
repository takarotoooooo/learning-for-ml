import json
import numpy as np
import pandas as pd
import tensorflow as tf
from functools import reduce
from pathlib import PosixPath
from tensorflow.keras.preprocessing.sequence import pad_sequences


class FactorizationMachinesEmbeddingsLayer(tf.keras.layers.Layer):
    """
    FactorizationMachinesのパラメータ用埋め込み層

    Attributes
    ----------
    fields : list
        FactorizationMachinesFieldの配列
    dimension : int
        潜在因子の深さ
    field_embeddings_layers : dict
        フィールド名をkey, 対応する埋め込み層をvalueとするdict
    """
    def __init__(self, fields: list, dimension: int, name_prefix: str = ''):
        super(FactorizationMachinesEmbeddingsLayer, self).__init__()

        self.fields = fields
        self.dimension = dimension

        self.field_embeddings_layers = {}
        for field in self.fields:
            self.field_embeddings_layers[field.name] = tf.keras.layers.Embedding(
                input_dim=field.max_index,
                output_dim=self.dimension,
                mask_zero=(not field.value_is_continuous),
                name=f'embeddings_{name_prefix}{field.name}',
                embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=0.01, seed=None))

    def generate_embeddings(self, input_value: list, field_name: str, value_is_continuous: bool):
        """
        埋め込み層を通して潜在因子を返却する

        Parameters
        ----------
        input_value : list
            入力値
        field_name : str
            入力値のフィールド名
        value_is_continuous : bool
            量的変数かカテゴリカル変数か

        Returns
        ----------
        embeddings : list
            入力値を埋め込み層に通した結果
            (入力値次元 x 潜在因子の深さ)の配列
        """
        if value_is_continuous:
            embeddings = tf.multiply(
                self.field_embeddings_layers[field_name](np.zeros(len(input_value))),
                input_value)
        else:
            layer_outputs = self.field_embeddings_layers[field_name](input_value)
            embeddings = tf.reshape(
                tf.reduce_sum(
                    layer_outputs,
                    axis=1,
                    keepdims=True),
                shape=(len(layer_outputs), self.dimension))

        return embeddings  # (len(inputs), 1 dim=5)

    def call(self, inputs):
        embeddings = [
            self.generate_embeddings(
                input_value=input_value,
                field_name=field.name,
                value_is_continuous=field.value_is_continuous)
            for input_value, field in zip(inputs, self.fields)
        ]
        return tf.reshape(
            tf.keras.layers.Concatenate(axis=1)(embeddings),
            shape=(len(embeddings[0]), len(embeddings), self.dimension))


class FactorizationMachinesInteractionLayer(tf.keras.layers.Layer):
    """
    FactorizationMachinesの交互作用部分を出力する層

    Attributes
    ----------
    embeddings_layer : FactorizationMachinesEmbeddingsLayer
        潜在因子用の埋め込み層
    """
    def __init__(self, fields: list, embeddings_dim: int = 5):
        super(FactorizationMachinesInteractionLayer, self).__init__()

        self.embeddings_layer = FactorizationMachinesEmbeddingsLayer(
            fields=fields,
            dimension=embeddings_dim,
            name_prefix='interaction_')

    def call(self, inputs):
        embeddings = self.embeddings_layer(inputs)  # (len(inputs), len(fields), dim=5)

        summed_square = tf.square(tf.reduce_sum(embeddings, axis=1))  # (len(inputs), dim=5)
        squared_sum = tf.reduce_sum(tf.square(embeddings), axis=1)  # (len(inputs), dim=5)
        sum_dimension = tf.reduce_sum(tf.subtract(summed_square, squared_sum), axis=1, keepdims=True)  # (len(inputs), 1)
        return tf.multiply(0.5, sum_dimension)  # (len(inputs), 1)


class FactorizationMachinesLinearLayer(tf.keras.layers.Layer):
    """
    FactorizationMachinesの単作用部分を出力する層

    Attributes
    ----------
    embeddings_layer : FactorizationMachinesEmbeddingsLayer
        潜在因子用の埋め込み層
    """
    def __init__(self, fields: list):
        super(FactorizationMachinesLinearLayer, self).__init__()

        self.embeddings_layer = FactorizationMachinesEmbeddingsLayer(
            fields=fields,
            dimension=1,
            name_prefix='linear_')

    def call(self, inputs):
        embeddings = self.embeddings_layer(inputs)  # (len(inputs), len(fields), dim=1)
        return tf.reduce_sum(embeddings, axis=1)  # (len(inputs), dim=1)


class FactorizationMachinesField():
    """
    FactorizationMachinesのフィールドの定義

    Attributes
    ----------
    name : str
        フィールドの名前
    value_is_continuous : bool
        量的変数かどうか
    vocabulary : list
        カテゴリ変数の場合の入り得る値のリスト
    vocabulary_mapping : dict
        カテゴリ変数の場合の入りうる値とインデックス値のdict
    max_index : int
        カテゴリ変数の場合の最大インデックス値
    multi_settable_values_num : int
        カテゴリ変数の場合の同時に入り得る値の数
        例）カテゴリなら1、タグなら5
    """
    def __init__(
            self,
            name: str,
            value_is_continuous: bool = False,
            vocabulary: list = [],
            multi_settable_values_num: int = 1) -> None:
        self.name = name
        self.value_is_continuous = value_is_continuous
        self.vocabulary = vocabulary
        self.vocabulary_mapping = {val: i + 1 for i, val in enumerate(self.vocabulary)}
        self.max_index = len(self.vocabulary) + 1
        self.multi_settable_values_num = multi_settable_values_num

    def fetch_index(self, value: str) -> int:
        """
        カテゴリ値に対するインデックス値を返却する
        未知のカテゴリ値の場合は0を返す

        Parameters
        ----------
        value : str
            カテゴリ値

        Returns
        ----------
        index : int
            カテゴリ値に対するインデックス値
        """
        return self.vocabulary_mapping.get(value, 0)

    def fetch_indices(self, values: list) -> list:
        return [self.fetch_index(value=value) for value in values]


class FactorizationMachinesModel(tf.keras.Model):
    """
    FactorizationMachinesモデル

    Attributes
    ----------
    linear_layer : FactorizationMachinesLinearLayer
        単作用の結果を表す層
    interaction_layer : FactorizationMachinesInteractionLayer
        交互作用の結果を表す層
    """

    def __init__(self, fields: list, embeddings_dim: int = 5):
        super(FactorizationMachinesModel, self).__init__()
        self.linear_layer = FactorizationMachinesLinearLayer(fields)
        self.interaction_layer = FactorizationMachinesInteractionLayer(fields, embeddings_dim=embeddings_dim)

    def call(self, inputs):
        linear_terms = self.linear_layer(inputs)
        interaction_terms = self.interaction_layer(inputs)
        output = tf.add(linear_terms, interaction_terms)
        return output


class FactorizationMachinesPredictor(FactorizationMachinesModel):
    """
    FactorizationMachines回帰モデル

    Attributes
    ----------
    linear_layer : FactorizationMachinesLinearLayer
        単作用の結果を表す層
    interaction_layer : FactorizationMachinesInteractionLayer
        交互作用の結果を表す層
    """
    def __init__(self, fields_map: dict, embeddings_dim: int = 5):
        super(FactorizationMachinesPredictor, self).__init__(
            fields=list(fields_map.values()), embeddings_dim=embeddings_dim
        )

    def call(self, inputs):
        super().call(inputs=inputs)


class FactorizationMachinesBinalyClassifier(FactorizationMachinesModel):
    """
    FactorizationMachines二値分類モデル

    Attributes
    ----------
    linear_layer : FactorizationMachinesLinearLayer
        単作用の結果を表す層
    interaction_layer : FactorizationMachinesInteractionLayer
        交互作用の結果を表す層
    """
    def __init__(self, fields_map: dict, embeddings_dim: int = 5):
        super(FactorizationMachinesBinalyClassifier, self).__init__(
            fields=list(fields_map.values()), embeddings_dim=embeddings_dim
        )

    def call(self, inputs):
        output = super().call(inputs=inputs)
        return tf.keras.activations.sigmoid(output)


class FactorizationMachines():
    """
    FactorizationMachinesの学習・推論機
    学習・推論にはpandasのDataFrameを入力にする

    Attributes
    ----------
    model : FactorizationMachinesModel
        FactorizationMachinesモデル
    fields_map : dict
        学習・推論に持ちいるデータのフィールド定義
    """
    def __init__(self):
        self.model = None
        self.fields_map = {}

    def generate_explanatory_variables(
            self,
            datasets: pd.core.frame.DataFrame,
            explanatory_variable_columns: list) -> list:
        """
        データから説明変数の要素を抽出し、学習・推論に適したフォーマットにして返却する

        Parameters
        ----------
        datasets : pandas.core.frame.DataFrame
            モデルの入力の元になるpandasのDataFrame
        explanatory_variable_columns : list
            説明変数になるカラム名のリスト

        Returns
        ----------
        matrix : list
            FactorizationMachinesモデルに入力する値
        """
        matrix = []
        for column in explanatory_variable_columns:
            field = self.fields_map[column]

            if field.value_is_continuous:
                matrix.append(datasets[column].values)
            else:
                matrix.append(
                    pad_sequences(
                        [
                            np.ravel(v)
                            for v in datasets[column].apply(
                                lambda x: field.fetch_indices(values=np.ravel(x))
                            ).values
                        ],
                        maxlen=field.multi_settable_values_num,
                        padding='post',
                        truncating='post'))

        return matrix

    def generate_train_datasets(
            self,
            datasets: pd.core.frame.DataFrame,
            explanatory_variable_columns: list,
            target_variable_column: str):
        """
        データから学習用のデータを整形する

        Parameters
        ----------
        datasets : pandas.core.frame.DataFrame
            モデルの入力の元になるpandasのDataFrame
        explanatory_variable_columns : list
            説明変数になるカラム名のリスト
        target_variable_column : str
            目的変数になるカラム名

        Returns
        ----------
        x_datasets : list
            説明変数の配列
        y_datasets : list
            目的変数の配列
        """
        x_datasets = self.generate_explanatory_variables(
            datasets=datasets,
            explanatory_variable_columns=explanatory_variable_columns
        )
        y_datasets = datasets[target_variable_column].values
        return (x_datasets, y_datasets)

    def __generate_vocabulary(self, series: pd.core.series.Series) -> list:
        """
        カテゴリ値のリストを作成する

        Parameters
        ----------
        series : pandas.core.series.Series
            カテゴリ値のリストを抽出したいpandasのSeries

        Returns
        ----------
        vocabulary : list
            重複を排除したカテゴリのリスト
        """
        if type(series.values[0]) == list:
            values = series[series.apply(lambda x: len(x) > 0)].values
            vocabulary = list(set(reduce(lambda a, b: np.concatenate([np.ravel(a), np.ravel(b)]), values)))
        else:
            vocabulary = list(series.unique())

        return vocabulary

    def __generate_fields_map(
            self,
            datasets: pd.core.frame.DataFrame,
            explanatory_variable_columns_info: dict) -> None:
        """
        フィールドのdictを作成する

        Parameters
        ----------
        datasets : pandas.core.frame.DataFrame
            説明変数のpandas DataFrame
        """
        for column_name, field_info in explanatory_variable_columns_info.items():
            value_type = field_info.get('value_type', 'categorical')
            max_values_num = field_info.get('max_values_num', 1)
            value_is_continuous = True
            vocabulary = []
            if value_type == 'categorical':
                value_is_continuous = False
                vocabulary = self.__generate_vocabulary(series=datasets[column_name])

            self.fields_map[column_name] = FactorizationMachinesField(
                name=column_name,
                value_is_continuous=value_is_continuous,
                vocabulary=vocabulary,
                multi_settable_values_num=max_values_num
            )

    def train_model(
            self,
            datasets: pd.core.frame.DataFrame,
            target_variable_column: str,
            explanatory_variable_columns_info: dict,
            split_train_rate: float = 0.8,
            predict_type: str = 'regression',
            embeddings_dim: int = 10,
            learning_rate: float = 0.01,
            epochs: int = 100,
            batch_size: int = 1000):
        """
        FactorizationMachinesモデルを学習する

        Parameters
        ----------
        datasets : pandas.core.frame.DataFrame
            学習に利用するデータ
        target_variable_column : str
            目的変数のカラム名
        explanatory_variable_columns_info : dict
            説明変数のカラムと設定情報
        split_train_rate : float
            学習データと検証データの分割率
        predict_type : str
            回帰問題か二値分類問題か
        embeddings_dim : int
            潜在因子ベクトルの深さ
        learning_rate : float
            学習率
        epochs : int
            学習反復回数
        batch_size : int
            学習ミニバッチサイズ
        """

        # 学習用とテスト用にindexを分割する
        split_indices = np.random.permutation(len(datasets))
        train_num = int(np.floor(len(datasets) * split_train_rate))
        train_dataset_indices = split_indices[:train_num]
        test_dataset_indices = split_indices[train_num:]
        self.train_datasets = datasets.iloc[train_dataset_indices]
        self.test_datasets = datasets.iloc[test_dataset_indices]

        self.__generate_fields_map(
            datasets=self.train_datasets,
            explanatory_variable_columns_info=explanatory_variable_columns_info
        )
        explanatory_variable_columns = list(self.fields_map.keys())

        train_x, train_y = self.generate_train_datasets(
            datasets=self.train_datasets,
            explanatory_variable_columns=explanatory_variable_columns,
            target_variable_column=target_variable_column)
        test_x, test_y = self.generate_train_datasets(
            datasets=self.test_datasets,
            explanatory_variable_columns=explanatory_variable_columns,
            target_variable_column=target_variable_column)

        if predict_type == 'regression':
            self.model = FactorizationMachinesPredictor(
                fields_map=self.fields_map, embeddings_dim=embeddings_dim
            )
            self.model.compile(
                optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9),
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=[tf.keras.metrics.RootMeanSquaredError()])
        elif predict_type == 'binaly_classification':
            self.model = FactorizationMachinesBinalyClassifier(
                fields_map=self.fields_map, embeddings_dim=embeddings_dim
            )
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss='binary_crossentropy',
                metrics=[tf.keras.metrics.AUC(num_thresholds=1000)])

        # 学習
        self.model.fit(
            x=train_x,
            y=train_y,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(test_x, test_y),
            verbose=1)

    def save_model(self, dest_dir_path: PosixPath):
        """
        学習済みモデルを保存する

        Parameters
        ----------
        dest_dir_path : PosixPath
            保存先ディレクトリパス
        """
        tf.keras.models.save_model(self.model, dest_dir_path)

        data = {}
        for column_name, field in self.fields_map.items():
            data[column_name] = {
                'name': field.name,
                'value_is_continuous': field.value_is_continuous,
                'vocabulary': field.vocabulary,
                'multi_settable_values_num': field.multi_settable_values_num
            }

        with open(dest_dir_path.joinpath('fields_map.json'), 'w') as outfile:
            json.dump(data, outfile)

    def load_model(self, src_dir_path: PosixPath):
        """
        学習済みモデルを読み込む

        Parameters
        ----------
        src_dir_path : PosixPath
            学習済みモデルが保存されているディレクトリパス
        """
        self.model = tf.keras.models.load_model(src_dir_path)

        with open(src_dir_path.joinpath('fields_map.json'), 'r') as infile:
            data = json.loads(infile.read())
            for column_name, field_data in data.items():
                self.fields_map[column_name] = FactorizationMachinesField(
                    name=column_name,
                    value_is_continuous=field_data['value_is_continuous'],
                    vocabulary=field_data['vocabulary'],
                    multi_settable_values_num=field_data['multi_settable_values_num']
                )

    def predict(
            self,
            datasets: pd.core.frame.DataFrame):
        """
        学習済みモデルを使って推論する

        Parameters
        ----------
        datasets : pandas.core.frame.DataFrame
            推論に利用するデータセット

        Returns
        ----------
        matix : numpy.array
            推論結果の配列
        """
        x = self.generate_explanatory_variables(
            datasets=datasets,
            explanatory_variable_columns=list(self.fields_map.keys()))
        return self.model.predict(x, verbose=0)
