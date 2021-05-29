import torch
from torch.nn import CrossEntropyLoss
from transformers import BertForSequenceClassification
import pytorch_lightning as pl

from .quantization import *


class IMDBSentimentClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-cased')
        self.loss = CrossEntropyLoss()

    def forward(self, x):
        mask = (x != 0).float()
        logits = self.model(x, mask)['logits']
        return logits

    def training_step(self, batch, batch_idx):
        y, x = batch['label'], batch['input_ids']
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        return {'loss': loss, 'log': {'train_loss': loss}}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer

def replace_layers(model, old, new):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_layers(module, old, new)
    
        if isinstance(module, old):
            ## simple module
            in_features, out_features = (int(x.split('=')[1]) for x in re.findall(r"\w*_features=\d*", module.extra_repr()))
            new_layer = new(in_features, out_features)
            new_layer.weight, new_layer.bias = module.weight, module.bias
            setattr(model, n, new_layer)

def bert_quan():
    model = torch.load("/Users/sarvasvarora/dev/SPARK/BFA/model.pt")
    replace_layers(model, torch.nn.Linear, quan_Linear)
    return model