I. Description of model
------------
I used pretrained Xception Keras Application. Training and validation of model were made in Kaggle Notebook(pretrainde-model-xception-transfer-learning.ipynb).  
1. model.json - Contain parameters of Xception model.


2. model_weights.hdf5 - File with weights which give the best performance of model.


II. Structure
----------

```shell
base_model = Xception(weights=weights, include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)
```
Evaluation of model gives such loss and accuracy respectively:
```shell
0.7365152887533414 0.8488806
```


