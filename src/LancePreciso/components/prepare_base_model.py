import os
import urllib.request as request
import numpy as np
#from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from LancePreciso.entity.config_entity import PrepareBaseModelConfig

                                                




class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    
    def get_base_model(self):
        self.model = tf.keras.Sequential()

        #normalizer = tf.keras.layers.Normalization(axis=-1)
        #normalizer.adapt(np.array(self.config.params_lenght_train_dataset))
        #self.model = tf.keras.Sequential([normalizer,
        #                            tf.keras.layers.Dense(units=self.config.params_lenght_train_dataset, activation='relu'),
        #                            tf.keras.layers.Dense(units=16, activation='relu'),
        #                            tf.keras.layers.Dense(units=1),])
        #self.model.compile(
        #optimizer=tf.optimizers.Adam(learning_rate=self.config.params_learning_rate),
        #loss='mean_absolute_error',
        #metrics=['mae', 'mse'])
        self.model.add(tf.keras.Input(shape=(20,1)))
        self.model.add(tf.keras.layers.Dense(units= self.config.params_lenght_train_dataset, activation='relu'))
        self.model.add(tf.keras.layers.Dense(units=16, activation='relu'))
        self.model.add(tf.keras.layers.Dense(units=1))

        self.save_model(path=self.config.base_model_path, model=self.model) 
    
    @staticmethod
    def _prepare_full_model(model, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for layer in model.layers:
                model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                model.trainable = False
        
        model.compile( optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                        loss='mean_absolute_error',
                        metrics=['mae', 'mse'])
        

        model.summary()
        return model
    
    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            #classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)