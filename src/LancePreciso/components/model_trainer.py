import os
import tensorflow as tf
import numpy as np
import pandas as pd
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from LancePreciso.entity.config_entity import TrainingConfig, DataIngestionConfig
from LancePreciso.config.configuration import ConfigurationManager
from pathlib import Path



class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.base_model_path
        )

    def train_valid_generator(self):
        dataIngestion = ConfigurationManager()
        df_carros_lances = pd.read_csv(dataIngestion.get_data_ingestion_config().local_data_file)
        df_carros_lances2 = df_carros_lances.copy()
        df_carros_lances2['marca']=df_carros_lances2['marca'].map({'HYUNDAI':1, 'VOLKSWAGEN':2,'HONDA':3, 'CHEVROLET':4, 'TOYOTA':5,
                                                         'RENAULT':6,'FIAT':7,'MITSUBISHI':8,'JEEP':9,'CHERY':10,'KIA':11,
                                                         'MINI':12, 'CITROEN':13,'MERCEDES':14, 'FORD':15, 'NISSAN':16,
                                                         'Peugeot':17, 'VOLVO':18, 'LAND ROVER':19})
        df_carros_lances2['tipo_de_monta']=df_carros_lances2['tipo_de_monta'].map({'pequena':0,'media':1,'locadora':2})
        df_carros_lances2['airbag_estourado'] = df_carros_lances2['airbag_estourado'].map({'não':0,'sim':1})
        df_carros_lances2['batida_frontal'] = df_carros_lances2['batida_frontal'].map({'não':0,'sim':1})
        df_carros_lances2['lateral_esq'] = df_carros_lances2['lateral_esq'].map({'não':0,'sim':1})
        df_carros_lances2['lateral_dir'] = df_carros_lances2['lateral_dir'].map({'não':0,'sim':1})
        df_carros_lances2['batida_traseira'] = df_carros_lances2['batida_traseira'].map({'não':0,'sim':1})
        df_carros_lances2['batida_baixo'] = df_carros_lances2['batida_baixo'].map({'não':0,'sim':1})
        df_carros_lances2['teto_amassado'] = df_carros_lances2['teto_amassado'].map({'não':0,'sim':1})
        df_carros_lances2['enchente'] = df_carros_lances2['enchente'].map({'não':0,'sim':1})
        df_carros_lances2['roubo'] = df_carros_lances2['roubo'].map({'não':0,'sim':1})
        df_carros_lances2['remarcardo'] = df_carros_lances2['remarcardo'].map({'não':0,'sim':1})
        df_carros_lances2['ausencia_de_pecas'] = df_carros_lances2['ausencia_de_pecas'].map({'não':0,'sim':1})
        df_carros_lances2['mostra_km'] = df_carros_lances2['mostra_km'].map({'não':0,'sim':1})
        
        df_carros_lances2['veiculo'] = df_carros_lances2['veiculo'].map({'HB20S' : 1, 'FOX' : 2, 'VIRTUS' : 3, 'CIVIC' : 4, 
                                                                   'TRACKER' : 5,'HB20X' : 6,'HILUX' : 7,'DUSTER' : 8,
                                                                   'WR-V' : 9,'PRISMA' : 10,'CRUZE' : 11,'MOBI' : 12,
                                                                   'YARIS' : 13,'FIT' : 14,'ONIX' : 15,'CITY' : 16,
                                                                   'COROLLA' : 17,'VOYAGE' : 18,'STRADA' : 19,'CRETA' : 20,
                                                                   'HR-V' : 21,'ETIOS' : 22,'COBALT' : 23,'CIVC' : 24,
                                                                   'ARGO' : 25,'SIENA' : 26,'HB20' : 27,'KWID' : 28,'POLO' : 29,
                                                                   'L200' : 30,'SANDERO' : 31,'UP' : 32,'LOGAN' : 33,
                                                                   'T-CROSS' : 34,'RENEGADE' : 35,'TIGGO' : 36,'GOL' : 37,
                                                                   'TIGUAN' : 38,'OUTLANDER' : 39,'CRONOS' : 40,'S10' : 41,
                                                                   'COMPASS' : 42,'SORENTO' : 43,'MINI COOPER' : 44,
                                                                   'C4' : 45,'CRV' : 46,'CLS' : 47,'C250' : 48,'ECOSPORT' : 49,
                                                                   'TAOS' : 50,'VERSA' : 51,'CHEROKEE' : 52,'FIESTA' : 53,
                                                                   '208' : 54,'XC60' : 55,'UNO' : 56,'GLA' : 57,'DISCOVERY' : 58,
                                                                   'CERATO' : 59,'PULSE' : 60,'AMAROK' : 61,'TORO' : 62,'SW4' : 63,
                                                                   'JETTA' : 64})
        df_carros_lances2.pop('modelo')
        train_dataset = df_carros_lances2.sample(frac=0.8,random_state=0)
        test_dataset=df_carros_lances2.drop(train_dataset.index)
        train_features=train_dataset.copy()
        test_features=test_dataset.copy()

        train_labels = train_features.pop('lance_recomendado')
        test_labels=test_features.pop('lance_recomendado')
        
        self.train_features = train_features
        self.train_labels = train_labels
        self.test_features = test_features
        self.test_labels = test_labels

    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)



    
    def train(self):
        #normalizer = tf.keras.layers.Normalization(axis=-1)
        #normalizer.adapt(np.array(self.train_dataset))
        #self.model = tf.keras.Sequential([normalizer, *self.model.layers])

        self.model.fit(self.train_features,
                       self.train_labels,
                       epochs=self.config.params_epochs,
                       validation_split=self.config.params_validation_split)
       

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )