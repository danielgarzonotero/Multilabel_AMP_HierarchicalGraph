
import torch
import torch.nn.functional as F
import numpy as np


def train(model, device, dataloader, optim, epoch):
    model.train()
    
    weights = torch.tensor([0.41, 0.22, 0.18, 0.09, 0.07, 0.04], device = device)  #pesos personalizados
    loss_func = torch.nn.MultiLabelSoftMarginLoss(weight=weights)
    #loss_func = torch.nn.BCEWithLogitsLoss() 
    
    loss_collect = 0
    
    # Looping over the dataloader allows us to pull out input/output data:
    for batch in dataloader:
        # Zero out the optimizer:        
        optim.zero_grad()
        batch = batch.to(device)
        x, edge_index,  edge_attr, idx_batch, cc, monomer_labels, aminoacids_features = batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.cc, batch.monomer_labels, batch.aminoacids_features

        # Make a prediction:
        pred = model(
                    x,
                    edge_index,
                    edge_attr,
                    idx_batch,
                    cc,
                    monomer_labels, 
                    aminoacids_features
        )
        
        #loss = 0
        #for i in range(pred.size(1)):  # Iterar sobre cada etiqueta
        #    loss += loss_func(pred[:, i], batch.y[:, i].float())  # Calcular la pérdida para cada etiqueta individualmente
        
        #loss /= pred.size(1)  # Promediar la pérdida sobre todas las etiquetas
        
        loss = loss_func(pred, batch.y.float()) 
        
        # Backpropagation:
        loss.backward()
        optim.step()

        # Calculate the loss and add it to our total loss
        loss_collect += loss.item()  # loss summed across the batch

    # Return our normalized losses so we can analyze them later:
    loss_collect /= len(dataloader.dataset)
    
    print(
        "Epoch:{}   Training dataset:   Loss per Datapoint: {:.7f}%".format(
            epoch, loss_collect * 100
        )
    ) 
    return loss_collect    

def validation(model, device, dataloader, epoch):

    model.eval()
    loss_collect = 0
    weights = torch.tensor([0.41, 0.22, 0.18, 0.09, 0.07, 0.04], device = device)  #pesos personalizados
    loss_func = torch.nn.MultiLabelSoftMarginLoss(weight=weights)
    #loss_func = torch.nn.BCEWithLogitsLoss() 
    
    # Remove gradients:
    with torch.no_grad():

        for batch in dataloader:
            
            batch = batch.to(device)
            x, edge_index,  edge_attr, idx_batch, cc, monomer_labels, aminoacids_features = batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.cc, batch.monomer_labels, batch.aminoacids_features
            
            # Make a prediction:
            pred = model(
                        x,
                        edge_index,
                        edge_attr,
                        idx_batch,
                        cc,
                        monomer_labels, 
                        aminoacids_features
            )
                
            # Calculate the loss:
            #loss = 0
            #for i in range(pred.size(1)):  # Iterar sobre cada etiqueta
            #    loss += loss_func(pred[:, i], batch.y[:, i].float())  # Calcular la pérdida para cada etiqueta individualmente
            
            #loss /= pred.size(1)  # Promediar la pérdida sobre todas las etiquetas

            loss = loss_func(pred, batch.y.float()) 
            
            # Calculate the loss and add it to our total loss
            loss_collect += loss.item()  # loss summed across the batch

    loss_collect /= len(dataloader.dataset)
    
    # Print out our test loss so we know how things are going
    print(
        "Epoch:{}   Validation dataset: Loss per Datapoint: {:.4f}%".format(
            epoch, loss_collect * 100
        )
    )  
    print('---------------------------------------')     
    # Return our normalized losses so we can analyze them later:
    return loss_collect


def predict_test(model, dataloader, device, weights_file, threshold):
    
    model.eval()
    model.load_state_dict(torch.load(weights_file))
    
    x_all = []
    y_all = []
    pred_all = []
    
    pred_all_csv = []
    
    # Remove gradients:
    with torch.no_grad():

        # Looping over the dataloader allows us to pull out input/output data:
        for batch in dataloader:
            
            batch = batch.to(device)
            x, edge_index,  edge_attr, idx_batch, cc, monomer_labels, aminoacids_features, sequence = batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.cc, batch.monomer_labels, batch.aminoacids_features, batch.sequence
            
            # Make a prediction:
            pred = model(
                        x,
                        edge_index,
                        edge_attr,
                        idx_batch,
                        cc,
                        monomer_labels, 
                        aminoacids_features
            )
            
            pred_sigmoid = torch.sigmoid(pred) #to be able to round and saving in a csv file as prediction results
            
            x_all.extend(sequence)
            y_all.append(batch.y.double())
            pred_all.append(pred)
            pred_all_csv.append(pred_sigmoid)
            
            

    # Concatenate the lists of tensors into a single tensor
    y_all = torch.cat(y_all, dim=0)
    pred_all = torch.cat(pred_all, dim=0)
    
    #This is to export the prediction rounded based on the threshold
    pred_all_csv = torch.cat(pred_all_csv, dim=0)
    pred_all_csv = [apply_custom_round(pred, threshold) for pred in pred_all_csv]
    pred_all_csv = np.stack(pred_all_csv)
    
    return x_all, y_all, pred_all, pred_all_csv


def apply_custom_round(pred_all_csv, threshold):
    # Aplica round para redondear cada elemento del tensor a 0 o 1
    pred_all_csv_rounded = torch.round(pred_all_csv)
    
    # Ajusta los valores redondeados según el umbral
    pred_all_csv_rounded[pred_all_csv < threshold] = 0
    pred_all_csv_rounded[pred_all_csv >= threshold] = 1

    return pred_all_csv_rounded.cpu().numpy()
