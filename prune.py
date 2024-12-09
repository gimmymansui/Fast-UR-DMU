import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import copy

class ModelPruner:
    def __init__(self, model, prune_amount=0.2):
        self.model = model
        self.prune_amount = prune_amount
        self.initial_state_dict = copy.deepcopy(model.state_dict())
       
    def prune_model(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d):
               prune.l1_unstructured(module, name='weight', amount=self.prune_amount)
               
    def reset_weights(self):
        self.model.load_state_dict(self.initial_state_dict)
       
    def make_permanent(self):   
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d):
               prune.remove(module, 'weight')