import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from torchmetrics.classification import BinaryConfusionMatrix
from torchmetrics.classification import MultilabelConfusionMatrix
import torch 


import pandas as pd

def evaluate_model(prediction, target, dataset_type, threshold, device):
    # Crear la matriz de confusión binaria
    target = target.to(torch.int)
    bcm = MultilabelConfusionMatrix(num_labels=6, threshold=threshold).to(device) 
    confusion_matrix = bcm(prediction, target)
    confusion_matrix_np = confusion_matrix.detach().cpu().numpy()
    
    num_labels = confusion_matrix_np.shape[0]
    
    TP_list = []
    TN_list = []
    FP_list = []
    FN_list = []
    ACC_list = []
    PR_list = []
    SN_list = []
    SP_list = []
    F1_list = []
    MCC_list = []
    AUC_list = []
    
    class_names = ['Antibacterial', 'MammalianCells', 'Antifungal', 'Antiviral', 'Anticancer', 'Antiparasitic']
    metric_names = ['TP', 'TN', 'FP', 'FN', 'ACC', 'PR', 'SN', 'SP', 'F1', 'MCC', 'AUC']

    for i in range(num_labels):
        TP = confusion_matrix[i][1, 1].cpu().numpy()
        TN = confusion_matrix[i][0, 0].cpu().numpy()
        FP = confusion_matrix[i][0, 1].cpu().numpy()
        FN = confusion_matrix[i][1, 0].cpu().numpy()
        
        # Calcular ACC, PR, SN, SP, F1, MCC
        ACC = (TP + TN) / (TP + TN + FP + FN)
        PR = TP / (TP + FP) if (TP + FP) > 0 else 0
        SN = TP / (TP + FN) if (TP + FN) > 0 else 0
        SP = TN / (TN + FP) if (TN + FP) > 0 else 0
        F1 = 2 * (PR * SN) / (PR + SN) if (PR + SN) > 0 else 0
        MCC = (TP * TN - FP * FN) / (((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5) if ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) > 0 else 0
        
        # Agregar valores a las listas
        TP_list.append(TP)
        TN_list.append(TN)
        FP_list.append(FP)
        FN_list.append(FN)
        ACC_list.append(ACC)
        PR_list.append(PR)
        SN_list.append(SN)
        SP_list.append(SP)
        F1_list.append(F1)
        MCC_list.append(MCC)
        
        
        # Calcular la curva ROC y el área bajo la curva (AUC)
        fpr, tpr, thresholds = roc_curve(target[:, i].cpu().numpy(), prediction[:, i].cpu().numpy())
        roc_auc = auc(fpr, tpr)
        AUC_list.append(roc_auc)
    
    metrics_dict = {
        'Class': class_names,
        'Dataset Type': [dataset_type] * num_labels,
        'TP': TP_list,
        'TN': TN_list,
        'FP': FP_list,
        'FN': FN_list,
        'ACC': ACC_list,
        'PR': PR_list,
        'SN': SN_list,
        'SP': SP_list,
        'F1': F1_list,
        'MCC': MCC_list,
        'AUC': AUC_list
    }

    # Crear el DataFrame
    df = pd.DataFrame(metrics_dict)
    df.set_index('Class', inplace=True)
    df.to_csv(f'results/{dataset_type}_metrics_results.csv')
        
        
    # Graficar la curva ROC para todas las clases
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    
    for i in range(num_labels):
        fpr, tpr, _ = roc_curve(target[:, i].cpu().numpy(), prediction[:, i].cpu().numpy())
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
        
    plt.xlabel('False Positive Rate (FPR)', fontsize=14)
    plt.ylabel('True Positive Rate (TPR)', fontsize=14)
    plt.title(f'ROC Curve - {dataset_type} Set', fontsize=16)
    plt.legend(loc='lower right', fontsize=12)
    plt.savefig(f'results/{dataset_type}_ROC.png', dpi=216)
    plt.show()
    
    # Sumar todas las matrices de confusión por clase
    total_confusion_matrix = confusion_matrix_np.sum(axis=0)
    
    # Graficar matriz de confusión
    plt.figure(figsize=(8, 8))
    plt.imshow(total_confusion_matrix, cmap=plt.get_cmap('YlGn'))
    plt.title('Global Confusion Matrix Plot - {}'.format(dataset_type))
    plt.colorbar()
    for i in range(total_confusion_matrix.shape[0]):
        for j in range(total_confusion_matrix.shape[1]):
            plt.text(j, i, str(total_confusion_matrix[i, j]), ha='center', va='center', color='black', fontsize=18)
    
    # Etiquetar ejes
    plt.xlabel('Predicted Negative      Predicted Positive', fontsize=16)
    plt.ylabel('Target Positive        Target Negative  ', fontsize=16)
    plt.savefig('results/{}_global_cm.png'.format(dataset_type), dpi=216)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.show()

    return



