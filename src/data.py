import pandas as pd
import os
import torch
from torch_geometric.data import InMemoryDataset

from src.utils import sequences_geodata, get_features
from src.device import device_info
from src.aminoacids_features import get_aminoacid_features

#-------------------------------------- Dataset 1--------------------------------------------------

class GeoDataset_1(InMemoryDataset):
    def __init__(self, root='../data', raw_name='dataset/Chung_Xiao_validated_7332_validation_training.csv', transform=None, pre_transform=None):
        self.filename = os.path.join(root, raw_name) 
        
        self.df = pd.read_csv(self.filename)
        self.x = self.df[self.df.columns[0]].values
        self.y = self.df[self.df.columns[1:7]].values  
        
        # Change root and processed_dir names
        super(GeoDataset_1, self).__init__(root=os.path.join(root, f'{raw_name.split(".")[0]}_processed'), transform=transform, pre_transform=pre_transform)
        
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    def processed_file_names(self):
        return ['training.pt']
    
    def process(self):
        node_ft_dict, edge_ft_dict = get_features(self.x)

        data_list = []
        
        cc = 0 #This is an ID for each peptide in the dataset to then be able to call each dictionary
        
        aminoacids_ft_dict = get_aminoacid_features()
        
        for i, (x, y) in enumerate(zip(self.x, self.y)):
            device_info_instance = device_info()
            device = device_info_instance.device

            data_list.append(sequences_geodata(cc, x, y, aminoacids_ft_dict, node_ft_dict, edge_ft_dict, device))
            
            cc += 1
            
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        

     
#-------------------------------------- Dataset 2--------------------------------------------------
''' 
class GeoDataset_2(InMemoryDataset):
    def __init__(self, root='../data', raw_name='dataset/Xiao_validation.csv', transform=None, pre_transform=None):
        self.filename = os.path.join(root, raw_name) 
        
        self.df = pd.read_csv(self.filename)
        self.x = self.df[self.df.columns[0]].values
        self.y = self.df[self.df.columns[1:7]].values  
        
        # Change root and processed_dir names
        super(GeoDataset_2, self).__init__(root=os.path.join(root, f'{raw_name.split(".")[0]}_processed'), transform=transform, pre_transform=pre_transform)
        
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    def processed_file_names(self):
        return ['validation.pt']
    
    def process(self):
        node_ft_dict, edge_ft_dict = get_features(self.x)

        data_list = []
        
        cc = 0 #This is an ID for each peptide in the dataset to then be able to call each dictionary
        
        aminoacids_ft_dict = get_aminoacid_features()
        
        for i, (x, y) in enumerate(zip(self.x, self.y)):
            device_info_instance = device_info()
            device = device_info_instance.device

            data_list.append(sequences_geodata(cc, x, y, aminoacids_ft_dict, node_ft_dict, edge_ft_dict, device))
            
            cc += 1
            
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
         '''
        
#-------------------------------------- Dataset 3--------------------------------------------------

class GeoDataset_3(InMemoryDataset):
    def __init__(self, root='../data', raw_name='dataset/Chung_Xiao_validated_817_testing.csv', transform=None, pre_transform=None):
        self.filename = os.path.join(root, raw_name) 
        
        self.df = pd.read_csv(self.filename)
        self.x = self.df[self.df.columns[0]].values
        self.y = self.df[self.df.columns[1:7]].values  
        
        # Change root and processed_dir names
        super(GeoDataset_3, self).__init__(root=os.path.join(root, f'{raw_name.split(".")[0]}_processed'), transform=transform, pre_transform=pre_transform)
        
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    def processed_file_names(self):
        return ['testing.pt']
    
    def process(self):
        node_ft_dict, edge_ft_dict = get_features(self.x)

        data_list = []
        
        cc = 0 #This is an ID for each peptide in the dataset to then be able to call each dictionary
        
        aminoacids_ft_dict = get_aminoacid_features()
        
        for i, (x, y) in enumerate(zip(self.x, self.y)):
            device_info_instance = device_info()
            device = device_info_instance.device

            data_list.append(sequences_geodata(cc, x, y, aminoacids_ft_dict, node_ft_dict, edge_ft_dict, device))
            
            cc += 1
            
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])