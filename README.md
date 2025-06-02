,

################### model.py #########################
"""
Copyright (c) 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch.nn as nn

from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from modules.sequence_modeling import BidirectionalLSTM
from modules.prediction import Attention


class LanguageClassifier(nn.Module):
    """Standalone language classification module"""
    def __init__(self, input_size, num_classes=4):
        super(LanguageClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(input_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, hidden_size]
        x = x.permute(0, 2, 1)  # [batch_size, hidden_size, seq_len]
        return self.classifier(x)


class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.stages = {'Trans': opt.Transformation, 'Feat': opt.FeatureExtraction,
                       'Seq': opt.SequenceModeling, 'Pred': opt.Prediction}

        """ Transformation """
        if opt.Transformation == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=opt.num_fiducial, I_size=(opt.imgH, opt.imgW), I_r_size=(opt.imgH, opt.imgW), I_channel_num=opt.input_channel)
        else:
            print('No Transformation module specified')

        """ FeatureExtraction """
        if opt.FeatureExtraction == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'RCNN':
            self.FeatureExtraction = RCNN_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        else:
            raise Exception('No FeatureExtraction module specified')
        self.FeatureExtraction_output = opt.output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        if opt.SequenceModeling == 'BiLSTM':
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
                BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
            self.SequenceModeling_output = opt.hidden_size
        else:
            print('No SequenceModeling module specified')
            self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        if opt.Prediction == 'CTC':
            self.Prediction = nn.Linear(self.SequenceModeling_output, opt.num_class)
        elif opt.Prediction == 'Attn':
            self.Prediction = Attention(self.SequenceModeling_output, opt.hidden_size, opt.num_class)
        else:
            raise Exception('Prediction is neither CTC or Attn')
            
    def forward(self, input, text=None, is_train=True, return_attention=False):
        """ Transformation stage """
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM
            
        """ Prediction stage """
        prediction = None
        attention_weights = None
        if self.stages['Pred'] == 'CTC':
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            prediction, attention_weights = self.Prediction(
                contextual_feature.contiguous(), text, is_train, 
                batch_max_length=self.opt.batch_max_length
            )

        outputs = {
            'prediction': prediction,
            'contextual_feature': contextual_feature,
            'attention_weights': attention_weights
        }
        
        return outputs

################### train_lang_classifier.py #########################
"""
Standalone language classifier training
"""
import os
import sys
import time
import random
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader

from dataset import LmdbDataset
from model import LanguageClassifier
from utils import CTCLabelConverter, AttnLabelConverter, Averager

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_lang_classifier(opt):
    """ Dataset preparation for language classification """
    # Create dataset
    train_dataset = LmdbDataset(root=opt.train_data, opt=opt)
    valid_dataset = LmdbDataset(root=opt.valid_data, opt=opt)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=opt.batch_size,
        shuffle=True,
        num_workers=int(opt.workers),
        pin_memory=True)
    
    valid_loader = DataLoader(
        valid_dataset, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        pin_memory=True)
    
    # Initialize the OCR model to extract features
    ocr_model = Model(opt)
    ocr_model = torch.nn.DataParallel(ocr_model).to(device)
    print(f'Loading OCR model from {opt.ocr_model_path}')
    ocr_model.load_state_dict(torch.load(opt.ocr_model_path, map_location=device))
    ocr_model.eval()  # Freeze OCR model
    
    # Initialize language classifier
    lang_classifier = LanguageClassifier(
        input_size=opt.hidden_size,
        num_classes=4  # Kurdish=0, Arabic=1, English=2, Number=3
    )
    lang_classifier = lang_classifier.to(device)
    
    # Setup loss and optimizer
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(lang_classifier.parameters(), lr=opt.lr)
    
    # Training loop
    best_accuracy = 0
    for epoch in range(opt.num_epoch):
        # Train
        lang_classifier.train()
        train_loss_avg = Averager()
        
        for i, (images, _, lang_labels) in enumerate(train_loader):
            images = images.to(device)
            lang_labels = torch.LongTensor(lang_labels).to(device)
            
            # Extract features from OCR model
            with torch.no_grad():
                features = ocr_model(images)['contextual_feature']
            
            # Forward pass through language classifier
            lang_logits = lang_classifier(features)
            
            # Calculate loss
            loss = criterion(lang_logits, lang_labels)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_avg.add(loss)
            
            if i % 50 == 0:
                print(f'Epoch [{epoch+1}/{opt.num_epoch}], '
                      f'Step [{i}/{len(train_loader)}], '
                      f'Loss: {train_loss_avg.val():.4f}')
        
        # Validation
        lang_classifier.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, _, lang_labels in valid_loader:
                images = images.to(device)
                lang_labels = lang_labels.to(device)
                
                # Extract features from OCR model
                features = ocr_model(images)['contextual_feature']
                
                # Forward pass through language classifier
                lang_logits = lang_classifier(features)
                
                # Calculate accuracy
                _, predicted = torch.max(lang_logits.data, 1)
                total += lang_labels.size(0)
                correct += (predicted == lang_labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{opt.num_epoch}], '
              f'Validation Accuracy: {accuracy:.2f}%')
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(lang_classifier.state_dict(), 
                      f'./saved_models/{opt.exp_name}/best_lang_classifier.pth')
    
    print(f'Training complete. Best accuracy: {best_accuracy:.2f}%')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', required=True, help='Experiment name')
    parser.add_argument('--train_data', required=True, help='Path to training LMDB dataset')
    parser.add_argument('--valid_data', required=True, help='Path to validation LMDB dataset')
    parser.add_argument('--ocr_model_path', required=True, help='Path to pre-trained OCR model')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--num_epoch', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    # OCR model parameters (must match the trained OCR model)
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='Number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='Number of input channels')
    parser.add_argument('--output_channel', type=int, default=512, help='Number of output channels')
    parser.add_argument('--hidden_size', type=int, default=256, help='Size of LSTM hidden state')
    parser.add_argument('--character', type=str, default='!%()-./0123456789:،؟ءآأؤإئابتثجحخدرزسشصضطعغفقكلمنهوي٠١٢٣٤٥٦٧٨٩پچڕژڤگڵۆیێە', help='Character set')
    
    opt = parser.parse_args()
    
    os.makedirs(f'./saved_models/{opt.exp_name}', exist_ok=True)
    train_lang_classifier(opt)

################### demo.py #########################
# ... [previous imports] ...

def demo(opt):
    """ model configuration """
    # ... [previous converter setup] ...
    
    # Load OCR model
    ocr_model = Model(opt)
    ocr_model = torch.nn.DataParallel(ocr_model).to(device)
    print(f'Loading OCR model from {opt.saved_model}')
    ocr_model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    ocr_model.eval()
    
    # Load language classifier if available
    lang_classifier = None
    if opt.lang_classifier_path:
        lang_classifier = LanguageClassifier(
            input_size=opt.hidden_size,
            num_classes=4
        )
        lang_classifier.load_state_dict(torch.load(opt.lang_classifier_path, map_location=device))
        lang_classifier = lang_classifier.to(device)
        lang_classifier.eval()
        lang_classes = ['Kurdish', 'Arabic', 'English', 'Number']
    
    # ... [rest of demo setup] ...
    
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            # ... [previous setup] ...
            
            # Forward pass through OCR model
            ocr_outputs = ocr_model(image, text_for_pred, is_train=False)
            preds = ocr_outputs['prediction']
            attention_weights = ocr_outputs['attention_weights']
            
            # Forward pass through language classifier if available
            lang_pred = None
            if lang_classifier:
                contextual_feature = ocr_outputs['contextual_feature']
                lang_logits = lang_classifier(contextual_feature)
                lang_probs = F.softmax(lang_logits, dim=1)
                _, lang_preds = lang_logits.max(1)
            
            # ... [previous OCR processing] ...
            
            for i, (img_name, pred, pred_max_prob) in enumerate(zip(image_path_list, preds_str, preds_max_prob)):
                # ... [previous OCR processing] ...
                
                # Add language prediction if available
                lang_info = ""
                if lang_classifier:
                    lang_class = lang_preds[i].item()
                    lang_name = lang_classes[lang_class]
                    lang_prob = lang_probs[i][lang_class].item()
                    lang_info = f" | Language: {lang_name} ({lang_prob:.2f})"
                
                print(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}{lang_info}')
                
                # ... [attention visualization] ...

# ... [rest of demo.py] ...

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ... [previous arguments] ...
    parser.add_argument('--lang_classifier_path', default='', help='Path to trained language classifier')
    # ... [rest of arguments] ...