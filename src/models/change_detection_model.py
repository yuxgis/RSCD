# @Time    : 2020/9/2 15:10
# @Author  : yux
# @Content :
from collections import OrderedDict

from sklearn.metrics import accuracy_score
from changedetection_model.nestnet_model import NestUnet
from models import BaseModel

import torch

class ChangeDetectionModel(BaseModel):
    def __init__(self, configuration):
        """Initialize the model.
        """
        super().__init__(configuration)

        self.loss_names = ['changedetection']
        self.network_names = ['nestnet']

        self.netunet = NestUnet(3, 1)
        self.netunet = self.netunet.to(self.device)
        if self.is_train:  # only defined during training time
            self.criterion_loss = torch.nn.BCEWithLogitsLoss()
            self.optimizer = torch.optim.Adam(self.netunet.parameters(), lr=configuration['lr'])
            self.optimizers = [self.optimizer]

        # storing predictions and labels for validation
        self.val_predictions = []
        self.val_labels = []
        self.val_images = []


    def forward(self):
        """Run forward pass.
        """
        self.output = self.netunet(self.input1,self.input2)


    def backward(self):
        """Calculate losses; called in every training iteration.
        """
        self.loss_segmentation = self.criterion_loss(self.output, self.label)


    def optimize_parameters(self):
        """Calculate gradients and update network weights.
        """
        self.loss_segmentation.backward() # calculate gradients
        self.optimizer.step()
        self.optimizer.zero_grad()
        torch.cuda.empty_cache()


    def test(self):
        super().test() # run the forward pass

        # save predictions and labels as flat tensors
        self.val_images.append(self.input)
        self.val_predictions.append(self.output)
        self.val_labels.append(self.label)


    def post_epoch_callback(self, epoch, visualizer):
        self.val_predictions = torch.cat(self.val_predictions, dim=0)
        predictions = torch.argmax(self.val_predictions, dim=1)
        predictions = torch.flatten(predictions).cpu()

        self.val_labels = torch.cat(self.val_labels, dim=0)
        labels = torch.flatten(self.val_labels).cpu()

        self.val_images = torch.squeeze(torch.cat(self.val_images, dim=0)).cpu()

        # Calculate and show accuracy
        val_accuracy = accuracy_score(labels, predictions)

        metrics = OrderedDict()
        metrics['accuracy'] = val_accuracy

        visualizer.plot_current_validation_metrics(epoch, metrics)
        print('Validation accuracy: {0:.3f}'.format(val_accuracy))

        # Here you may do something else with the validation data such as
        # displaying the validation images or calculating the ROC curve

        self.val_images = []
        self.val_predictions = []
        self.val_labels = []









