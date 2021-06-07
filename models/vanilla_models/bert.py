import torch
from torch.nn import CrossEntropyLoss
from transformers import BertForSequenceClassification
import pytorch_lightning as pl


class BERTClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained('bert-base-cased')
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

def bert():
    model = BERTClassifier()
    model.load_state_dict(torch.load("/home/sarvasvarora/spark/data/bert_model.pt", map_location=torch.device('cpu')))
    return model