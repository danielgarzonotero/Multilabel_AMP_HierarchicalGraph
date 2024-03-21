#%%
import time
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.explain import CaptumExplainer, Explainer

import pandas as pd
from src.device import device_info
from src.data import GeoDataset_1,  GeoDataset_3
from src.model import GCN_Geo
from src.process import train, validation, predict_test
from src.evaluation_metrics import evaluate_model

device_information = device_info()
print(device_information)
device = device_information.device

start_time = time.time()

## SET UP DATALOADERS: 

# Build starting dataset: 
datasets = {
            'training_validation_dataset': GeoDataset_1(root='data'),
            'testing_dataset': GeoDataset_3(root='data'),
            }


dataset = datasets['training_validation_dataset']
testing_datataset = datasets['testing_dataset']

#  Number of datapoints in each dataset:
size_training_dataset = 0.80
n_training = int(len(dataset) * size_training_dataset)
n_validation = len(dataset) - n_training

#  Define pytorch training and validation set objects:
training_set, validation_set = torch.utils.data.random_split(dataset, [n_training, n_validation], generator=torch.Generator().manual_seed(24))

print('Number of NODES features: ', dataset.num_features)
print('Number of EDGES features: ', dataset.num_edge_features)

finish_time_preprocessing = time.time()
time_preprocessing = (finish_time_preprocessing - start_time) / 60 

# Define dataloaders para conjuntos de entrenamiento, validación y prueba:
batch_size = 100  
train_dataloader = DataLoader(training_set, batch_size, shuffle=True)
val_dataloader = DataLoader(validation_set, batch_size, shuffle=True)
test_dataloader = DataLoader(testing_datataset, batch_size, shuffle=True)

## RUN TRAINING LOOP: 

# Train with a random seed to initialize weights:
torch.manual_seed(24)

# Set up model:
# Initial Inputs
initial_dim_gcn = dataset.num_features
edge_dim_feature = dataset.num_edge_features

hidden_dim_nn_1 = 20
hidden_dim_nn_2 = 10

hidden_dim_gat_0 = 15

hidden_dim_fcn_1 = 10
hidden_dim_fcn_2 = 5
hidden_dim_fcn_3 = 3 
dropout= 0.1

model = GCN_Geo(
                initial_dim_gcn,
                edge_dim_feature,
                hidden_dim_nn_1,
                hidden_dim_nn_2,

                hidden_dim_gat_0,
                
                hidden_dim_fcn_1,
                hidden_dim_fcn_2,
                hidden_dim_fcn_3,
                dropout
            ).to(device)



#/////////////////// Training /////////////////////////////
# Set up optimizer:
learning_rate = 1E-3 
weight_decay = 1E-5 
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# Definir el scheduler ReduceLROnPlateau
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold= 0.1, verbose= True, mode='max', patience=100, factor=0.1)


train_losses = []
val_losses = []

best_val_loss = float('inf')  # infinito

start_time_training = time.time()
number_of_epochs = 2

for epoch in range(1, number_of_epochs+1):
    train_loss = train(model, device, train_dataloader, optimizer, epoch, type_dataset='training_validation')
    train_losses.append(train_loss)

    val_loss = validation(model, device, val_dataloader, epoch, type_dataset='training_validation')
    val_losses.append(val_loss)

    # Programar el LR basado en la pérdida de validación
    #scheduler.step(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "weights/best_model_weights.pth")

finish_time_training = time.time()
time_training = (finish_time_training - start_time_training) / 60


#---------------------------------------//////// Losse curves ///////// ---------------------------------------------------------

plt.plot(train_losses, label='Training loss', color='darkorange') 
plt.plot(val_losses, label='Validation loss', color='seagreen')  

# Agregar texto para la mejor pérdida de validación
best_val_loss_epoch = val_losses.index(best_val_loss)  # Calcular el epoch correspondiente a la mejor pérdida de validación
best_val_loss = best_val_loss
# Añadir la época y el mejor valor de pérdida como subtítulo
plt.title('Training and Validation Loss\nAMP Dataset\nBest Validation Loss: Epoch {}, Value {:.4f}'.format(best_val_loss_epoch, best_val_loss), fontsize=17)
# Aumentar el tamaño de la fuente en la leyenda
plt.legend(fontsize=14) 
plt.xlabel('Epochs')
plt.ylabel('Loss')

# Guardar la figura en formato PNG con dpi 216
plt.savefig('results/Loss_curve.png', dpi=216)
plt.show()

# Testing:
weights_file = "weights/best_model_weights.pth"
threshold = 0.5
column_names = ['Antibacterial', 'MammalianCells', 'Antifungal', 'Antiviral', 'Anticancer', 'Antiparasitic']


# ------------------------------------////////// Training set /////////////---------------------------------------------------
start_time_predicting = time.time()
training_input, training_target, training_pred, training_pred_csv = predict_test(model, train_dataloader, device, weights_file, threshold, type_dataset='training_validation')

training_target_columns = training_target.cpu().numpy().T.tolist()
training_pred_columns = training_pred_csv.T.tolist()

# Crear un diccionario para el DataFrame
train_set_prediction = {
    'Sequence': training_input,
}

# Agregar cada columna de la matriz de objetivos al diccionario
for i, column in enumerate(training_target_columns):
    train_set_prediction[column_names[i]] = column

# Agregar cada columna de la matriz de predicciones al diccionario
for i, column in enumerate(training_pred_columns):
    train_set_prediction[f'Pred._{column_names[i]}'] = column

# Crear el DataFrame
df = pd.DataFrame(train_set_prediction)

# Guardar el DataFrame en un archivo Excel
df.to_excel('results/Training_prediction.xlsx', index=False)
# Evaluation metrics:

evaluate_model(prediction=training_pred,
               target=training_target,
               dataset_type='Training',
               threshold=threshold,
               device=device)


#-------------------------------------------- ////////// Validation Set //////////-------------------------------------------------
validation_input, validation_target, validation_pred, validation_pred_csv = predict_test(model, val_dataloader, device, weights_file, threshold, type_dataset='training_validation')

validation_target_columns = validation_target.cpu().numpy().T.tolist()
validation_pred_columns= validation_pred_csv.T.tolist()

# Crear un diccionario para el DataFrame
validation_set_prediction = {
    'Sequence': validation_input,
}

# Agregar cada columna de la matriz de objetivos al diccionario
for i, column in enumerate(validation_target_columns):
    validation_set_prediction[column_names[i]] = column

# Agregar cada columna de la matriz de predicciones al diccionario
for i, column in enumerate(validation_pred_columns):
    validation_set_prediction[f'Pred._{column_names[i]}'] = column

# Crear el DataFrame
df = pd.DataFrame(validation_set_prediction)

# Guardar el DataFrame en un archivo Excel
df.to_excel('results/Validation_prediction.xlsx', index=False)

# Evaluation metrics:

evaluate_model(prediction = validation_pred,
               target = validation_target,
               dataset_type = 'Validation',
               threshold = threshold,
               device = device)

# --------------------------------------------////////// Test Set //////////---------------------------------------------------

test_input, test_target, test_pred, test_pred_csv = predict_test(model, test_dataloader, device, weights_file,threshold, type_dataset='testing')

finish_time_testing = time.time()

test_target_columns = test_target.cpu().numpy().T.tolist()
test_pred_columns= test_pred_csv.T.tolist()

# Crear un diccionario para el DataFrame
test_set_prediction = {
    'Sequence': test_input,
}

# Agregar cada columna de la matriz de objetivos al diccionario
for i, column in enumerate(test_target_columns):
    test_set_prediction[column_names[i]] = column

# Agregar cada columna de la matriz de predicciones al diccionario
for i, column in enumerate(test_pred_columns):
    test_set_prediction[f'Pred._{column_names[i]}'] = column

# Crear el DataFrame
df = pd.DataFrame(test_set_prediction)

# Guardar el DataFrame en un archivo Excel
df.to_excel('results/Test_prediction.xlsx', index=False)

# Evaluation metrics:

evaluate_model(prediction=test_pred,
               target = test_target,
               dataset_type = 'Testing',
               threshold = threshold,
               device = device) 


finish_time_predicting = time.time()
time_prediction = (finish_time_predicting - start_time_predicting) / 60
total_time = (finish_time_predicting - start_time) / 60

#--------------------------------///////////Result DataFrame////////////---------------------------------------

data = {
     "Metric": [
    "node_features",
    "edge_features",
    "initial_dim_gcn",
    "edge_dim_feature",
    "hidden_dim_nn_1",
    "hidden_dim_nn_2",
    "hidden_dim_gat_0",
    "hidden_dim_fcn_1",
    "hidden_dim_fcn_2",
    "hidden_dim_fcn_3",
    "dropout",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "number_of_epochs",
    "threshold",
    "time_preprocessing",
    "time_training",
    "time_prediction",
    "total_time"
    ],
    "Value": [
        dataset.num_features,
        dataset.num_edge_features,
        initial_dim_gcn,
        edge_dim_feature ,
        hidden_dim_nn_1 ,
        hidden_dim_nn_2 ,
        hidden_dim_gat_0,
        hidden_dim_fcn_1 ,
        hidden_dim_fcn_2 ,
        hidden_dim_fcn_3 ,
        dropout,
        batch_size,
        learning_rate,
        weight_decay,
        number_of_epochs,
        threshold,
        time_preprocessing, 
        time_training,
        time_prediction,
        total_time
    ],
    
}


df = pd.DataFrame(data)
df.to_csv('results/Model_hyperparameters.csv', index=False)

#-------------------------------------///////// Explainer ////// -------------------------------------------

''' explainer = Explainer(
                    model=model,
                    algorithm=CaptumExplainer('IntegratedGradients'),
                    explanation_type='model', #Explains the model prediction.
                    model_config=dict(
                        mode='regression', #or 
                        task_level='node', #ok
                        return_type='raw', #ok
                    ),
                    node_mask_type='attributes', #"attributes": Will mask each feature across all nodes
                    edge_mask_type=None,
                    threshold_config=dict(
                        threshold_type='hard', #The type of threshold to apply. 
                        value = 0  #The value to use when thresholding.
                    ),
                    )


# Generar explicaciones para cada nodo en cada lote del DataLoader
aminoacids_features_dict = torch.load('data/dataset/dictionaries/training_validation/aminoacids_features_dict.pt', map_location=device)
blosum62_dict = torch.load('data//dataset/dictionaries/training_validation/blosum62_dict.pt', map_location=device)

batch_size = len(training_set)
train_dataloader = DataLoader(training_set, batch_size, shuffle=True)

start_time = time.time()

for batch in train_dataloader:
    batch = batch.to(device)
    x, target, edge_index,  edge_attr, idx_batch, cc, monomer_labels, amino = batch.x, batch.y, batch.edge_index, batch.edge_attr, batch.batch, batch.cc, batch.monomer_labels, batch.aminoacids_features
    
    node_features_dim = x.size(-1)
    print("Número de características de los nodos:", node_features_dim)
    
    explanation = explainer(
                            x=x, 
                            
                            edge_index=edge_index,
                            edge_attr=edge_attr,
                            aminoacids_features_dict=aminoacids_features_dict,
                            blosum62_dict=blosum62_dict,
                            idx_batch=idx_batch,
                            cc=cc,
                            monomer_labels=monomer_labels
                        )
    
    path = 'results/IntegratedGradients_feature_importance.png'
    explanation.visualize_feature_importance(path, top_k = node_features_dim) 
    finish_time = time.time()
    time_prediction = (finish_time- start_time) / 60
    print('\nTime Feature Importance:',time_prediction , 'min') '''

print("//////////// READY //////////////")
    # %%
