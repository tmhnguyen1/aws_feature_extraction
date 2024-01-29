from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
import joblib

import xgboost as xgb
import numpy as np
import pandas as pd
import tensorflow as tf
import json, os
import matplotlib.pyplot as plt

from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall
from tensorflow.keras.layers import Input, Dense, Dropout, Normalization
from tensorflow.keras.callbacks import EarlyStopping

import gc

# enable tensorflow memory flow
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def classify(row):
    true, pred = row.true, row.pred
    if true == pred and true == 0:
        return 'True Negative'
    elif true == pred and true == 1:
        return 'True Positive'
    elif true != pred and true == 0:
        return 'False Positive'
    else:
        return 'False Negative'
    
def create_model(xtrain, input_shape=500):
    inputs = Input(shape=input_shape)
    scaler = Normalization()
    scaler.adapt(xtrain)
    scaled_inputs = scaler(inputs)
    x = Dense(500, activation='relu')(scaled_inputs)
    x = Dropout(0.3)(x)
    x = Dense(100, activation='relu')(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(100, activation='relu')(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(50, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=output)
    return model
    

basedir = r"C:\Users\tmhnguyen\Documents\lalamove\lalamove\data\Clean_extracted_240115_uncal\train"
labels = [6, 7]
model_names = ['ann'] 
synthetic_percent_list = [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1]
model_performance = []

seed_value = 123
tf.random.set_seed(seed_value)
np.random.seed(seed_value)


for label in labels:
    print('\n\n\nLabel', label)
    y = pd.read_csv(basedir + f'/{label}/train_label_{label}.csv')
    X = []
    step = 30_000
    for i in range(np.ceil(len(y)/30_000).astype(int)):
        temp = pd.read_csv(basedir + f'/{label}/extract_features_{label}_{i}.csv', index_col=0)
        X.append(temp)
    X = pd.concat(X)
    assert len(X) == len(y), f"Length mismatch {len(X)}, {len(y)}"

    for model_name in model_names:
        dates = y.date.unique()
        for chosen in dates:
            for synthetic_percent in synthetic_percent_list:
                print('\n\nmodel_name: ', model_name)
                print('\ntest_date ', chosen)
                print('\nSynthetic percent: ', synthetic_percent)
                # chosen = dates[-3]
                test_idx = y[(y.date == chosen) & (y.type == 0)].index
                train_idx = y[(y.date != chosen) & (y.type == 0)]
                train_idx_add = y[(y.date != chosen) & (y.type == 1)].sample(frac=synthetic_percent)
                train_idx = pd.concat([train_idx, train_idx_add]).index
                # train_idx = y[y.date != chosen].index

                X_train, X_test = X.iloc[train_idx].to_numpy(), X.iloc[test_idx].to_numpy()
                y_train, y_test = y.iloc[train_idx].label.to_numpy(), y.iloc[test_idx].label.to_numpy()

                tf.keras.backend.clear_session() # release resource associated with previous model
                model = create_model(X_train, input_shape=X_train.shape[1])
                print('Normalization layer adapted: ', model.layers[1].is_adapted)

                model.compile(optimizer=Adam(learning_rate=2e-5),
                        loss='binary_crossentropy',
                        metrics=[BinaryAccuracy(name='acc'),
                                Precision(name='precision'),
                                Recall(name='recall')])

                history = model.fit(X_train, y_train, batch_size=256, epochs=50, validation_data=(X_test, y_test),
                                    callbacks=[EarlyStopping(patience=20, min_delta=0.00005, restore_best_weights=True)])

                os.makedirs(basedir + f'/../model_{model_name}/{label}/', exist_ok=True)
                if model_name == 'xgb':
                    model.save_model(basedir + f'/../model_{model_name}/{label}/model_{label}_test_date_{chosen}_spercent_{synthetic_percent}.json')
                elif model_name == 'ann':
                    model.save(basedir + f'/../model_{model_name}/{label}/model_{label}_test_date_{chosen}_spercent_{synthetic_percent}.hdf5')
                else:
                    joblib.dump(model, basedir + f'/../model_{model_name}/{label}/model_{label}_test_date_{chosen}_spercent_{synthetic_percent}.joblib')

                print('test date', chosen)
                predictions_train = (model.predict(X_train) >= 0.5).astype(int)
                accuracy_train = accuracy_score(y_train, predictions_train)
                precision_train = precision_score(y_train, predictions_train)
                recall_train = recall_score(y_train, predictions_train)
                print('train\n', accuracy_train, precision_train, recall_train)

                predictions = (model.predict(X_test) >= 0.5).astype(int)
                accuracy = accuracy_score(y_test, predictions)
                precision = precision_score(y_test, predictions)
                recall = recall_score(y_test, predictions)
                print('test\n', accuracy, precision, recall)

                w = 5
                pred = np.convolve(predictions.flatten(), np.ones(w), mode='same') / w >= 0.5
                accuracy_avg = accuracy_score(y_test, pred)
                precision_avg = precision_score(y_test, pred)
                recall_avg = recall_score(y_test, pred)

                model_performance.append([label, model_name, chosen, synthetic_percent, accuracy_train, recall_train, precision_train, accuracy, recall, precision, accuracy_avg, recall_avg, precision_avg])
                pd.DataFrame(model_performance, columns=['label', 'model_name', 'test_date', 'synthetic_%', 'train_accuracy', 'train_recall', 'train_precision', 'test_accuracy', 'test_recall', 'test_precision', 'test_accuracy_avg', 'test_recall_avg', 'test_precision_avg']).to_csv(basedir + '/../model_performance.csv')

                # ##################################
                pred = model.predict(X_test).flatten() >= 0.5
                print(pred.shape, y_test.shape)

                df = pd.DataFrame(np.stack((y_test, pred)).T, columns=['true', 'pred'])
                df.pred = df.pred.astype(int)

                df['type'] = df.apply(lambda x: classify(x), axis=1)
                types = df.type.value_counts().sort_index()[::-1]
                print(types)

                fig, ax = plt.subplots(figsize=(20, 2.5))
                i = 0
                colors = ['skyblue', 'blue', 'green', 'red']
                types_ = ['True Negative', 'True Positive', 'False Negative', 'False Positive']

                for j, t in enumerate(types_):
                    try:
                        ax.scatter(df[df.type==t].index, [i]*types[t], label=t, c=colors[j])
                    except KeyError:
                        print(f'There is no {t}')
                    i += 0.1

                ax.legend()
                ax.set_ylim(0, 2)
                ax.set_xlabel('Seconds')
                ax.get_yaxis().set_visible(False)
                ax.set_title(f'{label}')
                fig.savefig(basedir + f'/../model_{model_name}/{label}/prediction_raw_test_date_{chosen}_spercent_{synthetic_percent}.png')
                plt.close()

                pred = model.predict(X_test).flatten() 
                print(pred.shape, y_test.shape)
                w = 5 # window in seconds
                pred = np.convolve(pred, np.ones(w), mode='same') / w >= 0.5

                print(pred.shape, y_test.shape)
                df = pd.DataFrame(np.stack((y_test, pred)).T, columns=['true', 'pred'])
                df.pred = df.pred.astype(int)
                    
                df['type'] = df.apply(lambda x: classify(x), axis=1)
                types = df.type.value_counts().sort_index()[::-1]
                print(types)

                fig, ax = plt.subplots(figsize=(20, 2.5))
                i = 0
                colors = ['skyblue', 'blue', 'green', 'red']
                types_ = ['True Negative', 'True Positive', 'False Negative', 'False Positive']

                for j, t in enumerate(types_):
                    try:
                        ax.scatter(df[df.type==t].index, [i]*types[t], label=t, c=colors[j])
                    except KeyError:
                        print(f'There is no {t}')
                    i += 0.1

                ax.legend()
                ax.set_ylim(0, 2)
                ax.set_xlabel('Seconds')
                ax.get_yaxis().set_visible(False)
                ax.set_title(f'{label}')
                fig.savefig(basedir + f'/../model_{model_name}/{label}/prediction_avg_{w}s_test_date_{chosen}_spercent_{synthetic_percent}.png')
                plt.close()

                del model
                gc.collect()