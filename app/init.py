import os
import sys
import json
import numpy   as np
import pandas  as pd
import pycaret as pc
from datetime import datetime
from typing   import Optional, List, Tuple
from pycaret.classification import *
from sklearn.preprocessing  import MultiLabelBinarizer

print(
    f"""
    ┌ Pandas  🐼: V.{pd.__version__}
    ├ Numpy   🔢: V.{np.__version__}
    └ PyCaret 🧠: V.{pc.__version__}
    """
)

# 📌 Clase para la carga y preprocesamiento de datos
class DatasetProcessor:
    def __init__(self, filename: str = 'Copia de MLA_100k_checked_v3', test_size: int = 10000) -> None:
        self.filename  = filename
        self.test_size = test_size
        self.setup     = None
    # --------------------------------------------------------------------------
    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """LECTURA DEL DATAFRAME DESDE LOS DATOS BRUTOS .JSON"""
        try:
            N    = -self.test_size
            data = [json.loads(x) for x in open(self.filename)]
            Train = data[:N]
            Test  = data[N:]
            Train = pd.json_normalize(Train, sep = '_')
            Test  = pd.json_normalize(Test , sep = '_')
            return Train, Test
        except FileNotFoundError:
            raise ValueError(f"El archivo '{self.filename}' no fue encontrado.")
        except Exception as e:
            raise RuntimeError(f"Error al cargar el dataset: {e}")
    # --------------------------------------------------------------------------
    def count_list_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = ['non_mercado_pago_payment_methods', 'pictures']
        dict = {}
        for i in cols:
            dict[i] = df[i].apply(
                lambda x: np.nan if len(x) == 0 else len(x)
            )
        df_new = pd.DataFrame(data = dict)
        df_new.rename(
            columns = {
                'pictures': 'pictures_count',
                'non_mercado_pago_payment_methods': 'non_mercado_pago_payment_methods_count'
            },
            inplace = True
        )
        df_new['pictures_count'].fillna(0, inplace = True)
        df_new['non_mercado_pago_payment_methods_count'].fillna(0, inplace = True)
        return df_new
    # --------------------------------------------------------------------------
    def transform_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['date_created_dt'] = pd.to_datetime(df['date_created'])
        df['last_updated_dt'] = pd.to_datetime(df['last_updated'])

        df['age_days'] = (
                ( df['last_updated_dt'] - df['date_created_dt'] )
            ).apply(lambda x: x.total_seconds() / (60 * 60 * 24))
        df['duration_days'] = (
                ( df['stop_time'].apply(lambda x: datetime.fromtimestamp(x / 1000)) ) - \
                ( df['start_time'].apply(lambda x: datetime.fromtimestamp(x / 1000)) )
            ).apply(lambda x: x.total_seconds() / (60 * 60 * 24))
        return df[['date_created_dt', 'last_updated_dt', 'age_days', 'duration_days']]
    # --------------------------------------------------------------------------
    def list2text(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        df = df.copy()
        for i in columns:
            if i in df.columns:
                df[i] = df[i].apply(lambda x: ', '.join(x) if isinstance(x, list) and x else 'NAN')
        return df
    # --------------------------------------------------------------------------
    def onehot_payments(self, df: pd.DataFrame, paymentsDict: dict) -> pd.DataFrame:
        paymentSeries = df['non_mercado_pago_payment_methods'].apply(
            lambda x: [d['description'] for d in x if isinstance(x, list) and 'description' in d]
        )
        MLB    = MultiLabelBinarizer(classes = sorted(paymentsDict))
        oneHot = MLB.fit_transform(paymentSeries)
        v_replace = np.vectorize(lambda x: x.replace(' ', '_').upper())
        return pd.DataFrame(oneHot, columns = v_replace(MLB.classes_), index = df.index)
    # --------------------------------------------------------------------------
    def cleaned_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        Train, Test = self.load_raw_data()
        NewCondition = {'used': 0, 'new': 1}
        Train['condition'] = Train['condition'].map(NewCondition)
        Test['condition']  = Test['condition'].map(NewCondition)

        paymentsDict = {
            'Acordar con el comprador', 'American Express', 'Cheque certificado'
            ,'Contra reembolso', 'Diners', 'Efectivo', 'Giro postal', 'MasterCard'
            , 'Mastercard Maestro', 'MercadoPago', 'Tarjeta de crédito', 'Transferencia bancaria'
            , 'Visa', 'Visa Electron'
        }
        Train = pd.concat([Train, self.count_list_fields(Train)] , axis = 1)
        Train = pd.concat([Train, self.transform_datetime(Train)], axis = 1)
        Train = pd.concat([Train, self.onehot_payments(Train, paymentsDict)], axis = 1)
        Test  = pd.concat([Test , self.count_list_fields(Test)]  , axis = 1)
        Test  = pd.concat([Test , self.transform_datetime(Test)] , axis = 1)
        Test  = pd.concat([Test , self.onehot_payments(Test, paymentsDict)], axis = 1)

        RemoveCols = [
            'attributes', 'category_id', 'coverage_areas', 'date_created', 'descriptions'
            , 'differential_pricing', 'international_delivery_mode', 'last_updated'
            , 'listing_source', 'non_mercado_pago_payment_methods', 'official_store_id'
            , 'original_price', 'parent_item_id', 'permalink', 'pictures', 'secure_thumbnail'
            , 'seller_address_city_id', 'seller_address_country_id', 'seller_address_country_name'
            , 'seller_address_state_id', 'shipping_dimensions', 'shipping_free_methods'
            , 'shipping_methods', 'shipping_tags', 'shipping_tagscatalog_product_id', 'site_id'
            , 'start_time', 'stop_time', 'subtitle', 'thumbnail', 'variations', 'video_id', 'warranty'
        ]
        Train.drop(RemoveCols, axis = 1, errors = 'ignore', inplace = True)
        Test.drop(RemoveCols , axis = 1, errors = 'ignore', inplace = True)

        GetTextCols = ['deal_ids', 'sub_status', 'tags']
        Train = self.list2text(Train, GetTextCols)
        Test  = self.list2text(Test , GetTextCols)

        Train.set_index('id', inplace = True); Test.set_index('id', inplace = True)
        SCHEMA = {
            'accepts_mercadopago'                     : 'bool'
            , 'automatic_relist'                      : 'int64'
            , 'available_quantity'                    : 'int64'
            , 'base_price'                            : 'float64'
            , 'buying_mode'                           : 'str'
            , 'catalog_product_id'                    : 'str'
            , 'condition'                             : 'bool'
            , 'currency_id'                           : 'str'
            , 'deal_ids'                              : 'str'
            , 'duration_days'                         : 'float64'
            , 'initial_quantity'                      : 'int64'
            , 'listing_type_id'                       : 'str'
            , 'non_mercado_pago_payment_methods_count': 'int64'
            , 'pictures_count'                        : 'int64'
            , 'price'                                 : 'float64'
            , 'seller_address_city_name'              : 'str'
            , 'seller_address_state_name'             : 'str'
            , 'seller_id'                             : 'float64'
            , 'shipping_free_shipping'                : 'int64'
            , 'shipping_local_pick_up'                : 'int64'
            , 'shipping_mode'                         : 'str'
            , 'sold_quantity'                         : 'int64'
            , 'status'                                : 'str'
            , 'sub_status'                            : 'str'
            , 'tags'                                  : 'str'
            , 'title'                                 : 'str'
            , 'ACORDAR_CON_EL_COMPRADOR'              : 'bool'
            , 'AMERICAN_EXPRESS'                      : 'bool'
            , 'CHEQUE_CERTIFICADO'                    : 'bool'
            , 'CONTRA_REEMBOLSO'                      : 'bool'
            , 'DINERS'                                : 'bool'
            , 'EFECTIVO'                              : 'bool'
            , 'GIRO_POSTAL'                           : 'bool'
            , 'MASTERCARD'                            : 'bool'
            , 'MASTERCARD_MAESTRO'                    : 'bool'
            , 'MERCADOPAGO'                           : 'bool'
            , 'TARJETA_DE_CRÉDITO'                    : 'bool'
            , 'TRANSFERENCIA_BANCARIA'                : 'bool'
            , 'VISA'                                  : 'bool'
            , 'VISA_ELECTRON'                         : 'bool'
        }
        Train = Train.astype(SCHEMA); Test = Test.astype(SCHEMA)

        Train = Train.applymap(lambda x: x.upper() if type(x) == str else x)
        Test  = Test.applymap(lambda x: x.upper() if type(x) == str else x)
        return Train, Test

# 📌 Clase para el modelo de clasificación binaria
class BinaryClassifier:
    def __init__(self, **kwargs) -> None:
        self.params = kwargs

    def setup_training_pipeline(self, X_train: pd.DataFrame, X_test: pd.DataFrame, target: str) -> None:
        return setup(data = X_train, target = 'condition', test_data = X_test, **self.params)

    def save_transformed(self, path: str) -> None:
        Data_transformed = get_config('dataset_transformed')
        X_train_transformed = get_config('X_train_transformed'); y_train = get_config('y_train')
        X_test_transformed  = get_config('X_test_transformed') ; y_test  = get_config('y_test')

        Data_transformed.to_parquet(path + 'Data_transformed.parquet.gzip', compression = 'brotli')
        X_train_transformed.to_parquet(path + 'X_train_transformed.parquet.gzip', compression = 'brotli')
        X_test_transformed.to_parquet(path  + 'X_test_transformed.parquet.gzip' , compression = 'brotli')
        y_train.to_frame().to_parquet(path + 'y_train.parquet.gzip', compression = 'brotli')
        y_test.to_frame().to_parquet(path  + 'y_test.parquet.gzip' , compression = 'brotli')
        return None

# 📌 Ejecución del flujo completo
if __name__ == "__main__":
    # ------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------
    # Procesamiento de Datos
    Builder = DatasetProcessor('/lakehouse/default/Files/Copia de MLA_100k_checked_v3.jsonlines')
    X_train, X_test = Builder.cleaned_dataset()
    # ------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------
    # Entrenamiento del modelo
    catVars = [
        'buying_mode', 'catalog_product_id', 'currency_id', 'deal_ids', 'listing_type_id'
        , 'seller_address_city_name', 'seller_address_state_name', 'shipping_mode'
        , 'status', 'sub_status', 'tags', 'title'
        , 'ACORDAR_CON_EL_COMPRADOR', 'AMERICAN_EXPRESS', 'CHEQUE_CERTIFICADO', 'CONTRA_REEMBOLSO'
        , 'DINERS', 'EFECTIVO', 'GIRO_POSTAL', 'MASTERCARD', 'MASTERCARD_MAESTRO', 'MERCADOPAGO'
        , 'TARJETA_DE_CRÉDITO', 'TRANSFERENCIA_BANCARIA', 'VISA', 'VISA_ELECTRON'
    ]
    numVars = [
        'age_days', 'automatic_relist', 'available_quantity', 'base_price', 'duration_days'
        , 'initial_quantity', 'non_mercado_pago_payment_methods_count', 'pictures_count'
        , 'price', 'seller_id', 'shipping_free_shipping', 'shipping_local_pick_up', 'sold_quantity'
    ]
    # .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . .
    Classifier = BinaryClassifier(
        # , ordinal_features = None, numeric_features = numVars, categorical_features = catVars
        date_features = ['date_created_dt', 'last_updated_dt'], create_date_columns = ['day', 'month', 'year']
        , imputation_type = 'simple', iterative_imputation_iters = 10
        , numeric_imputation = 0, categorical_imputation = 'mode'
        , max_encoding_ohe = 200, fix_imbalance = False, fix_imbalance_method = 'SMOTE'
        , normalize = True, normalize_method = 'minmax'                       # zscore | maxabs | robust
        , fold = 20, fold_strategy = 'stratifiedkfold', fold_shuffle = False  # kfold  | groupkfold
        , n_jobs = -1, use_gpu = False, session_id = 2025
        # , log_experiment = 'mlflow', experiment_name = 'MercadoLibre_BinaryClass'
    )
    Classifier.setup_training_pipeline(X_train, X_test, 'condition')
    Classifier.save_transformed(path = '')
    Top3_Models = compare_models(
        include = ['lr', 'knn', 'dt', 'rf', 'et', 'xgboost', 'lightgbm', 'svm', 'ridge', 'gbc']
        , cross_validation = True, sort = 'Accuracy', n_select = 3, probability_threshold = 0.5
    )
    Blender_Weighted = blend_models(
        estimator_list = Top3_Models, choose_better = True, optimize = 'Accuracy', method = 'auto', weights = [0.5, 0.3, 0.2]
    )
    Tuned_Blender_Weighted = tune_model(estimator = Blender_Weighted, fold = 3, n_iter = 7, optimize = 'Accuracy', choose_better = True)
    # ------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------
    # Evaluación y salva del modelo
    ModelFinal = finalize_model(Tuned_Blender_Weighted)
    save_model(ModelFinal, 'Final_Model_MercadoLibre')