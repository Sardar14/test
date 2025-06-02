# Read Me
Sometimes you want numbered lists:

1. One
2. Two
3. Three

Sometimes you want bullet points:

* Start a line with a star
* Profit!

Alternatively,

- Dashes work just as well
- And if you have sub points, put two spaces before the dash or star:
  - Like this
  - And this

```
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
    """Language classification module"""
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
                       'Seq': opt.SequenceModeling, 'Pred': opt.Prediction,
                       'LangCls': opt.LanguageClassification}

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
            
        """ Language Classification """
        if opt.LanguageClassification:
            self.LanguageClassifier = LanguageClassifier(
                input_size=self.SequenceModeling_output,
                num_classes=4  # Kurdish=0, Arabic=1, English=2, Number=3
            )

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
            
        """ Language Classification """
        lang_logits = None
        if self.stages['LangCls']:
            lang_logits = self.LanguageClassifier(contextual_feature)

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
            'lang_logits': lang_logits,
            'attention_weights': attention_weights
        }
        
        return outputs

################### dataset.py #########################
import os
import sys
import re
import six
import math
import lmdb
import torch

from natsort import natsorted
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, ConcatDataset, Subset
from itertools import accumulate
import torchvision.transforms as transforms

# ... [previous code remains unchanged] ...

class LmdbDataset(Dataset):
    def __init__(self, root, opt):
        self.root = root
        self.opt = opt
        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print('cannot create lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples

            if self.opt.data_filtering_off:
                self.filtered_index_list = [index + 1 for index in range(self.nSamples)]
            else:
                self.filtered_index_list = []
                for index in range(self.nSamples):
                    index += 1
                    label_key = 'label-%09d'.encode() % index
                    label_str = txn.get(label_key).decode('utf-8')
                    
                    # Parse language class from label format: text\tlang_class
                    if '\t' in label_str:
                        label, lang_class = label_str.split('\t')
                    else:
                        label = label_str
                        lang_class = '0'  # Default to Kurdish

                    if len(label) > self.opt.batch_max_length:
                        continue

                    out_of_char = f'[^{self.opt.character}]'
                    if re.search(out_of_char, label.lower()):
                        continue

                    self.filtered_index_list.append(index)

                self.nSamples = len(self.filtered_index_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index = self.filtered_index_list[index]

        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d'.encode() % index
            label_str = txn.get(label_key).decode('utf-8')
            
            # Parse language class from label format: text\tlang_class
            if '\t' in label_str:
                label, lang_class = label_str.split('\t')
                lang_class = int(lang_class)
            else:
                label = label_str
                lang_class = 0  # Default to Kurdish

            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                if self.opt.rgb:
                    img = Image.open(buf).convert('RGB')
                else:
                    img = Image.open(buf).convert('L')

            except IOError:
                print(f'Corrupted image for {index}')
                if self.opt.rgb:
                    img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
                else:
                    img = Image.new('L', (self.opt.imgW, self.opt.imgH))
                label = '[dummy_label]'
                lang_class = 0

            if not self.opt.sensitive:
                label = label.lower()

            out_of_char = f'[^{self.opt.character}]'
            label = re.sub(out_of_char, '', label)

        return (img, label, lang_class)

# ... [rest of dataset.py remains unchanged] ...

################### train.py #########################
# ... [previous imports remain unchanged] ...

def train(opt):
    """ dataset preparation """
    # ... [previous setup code remains unchanged] ...

    """ model configuration """
    # ... [previous converter setup remains unchanged] ...
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    # ... [previous model setup remains unchanged] ...

    """ setup loss """
    # OCR loss
    if 'CTC' in opt.Prediction:
        # ... [previous CTC setup] ...
    else:
        ocr_criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)
    
    # Language classification loss
    lang_criterion = torch.nn.CrossEntropyLoss().to(device)
    
    # ... [rest of setup remains unchanged] ...

    while(True):
        # train part
        image_tensors, labels, lang_labels = train_dataset.get_batch()
        image = image_tensors.to(device)
        text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)
        lang_labels = torch.LongTensor(lang_labels).to(device)
        batch_size = image.size(0)

        # Forward pass
        model_outputs = model(image, text)
        preds = model_outputs['prediction']
        lang_logits = model_outputs['lang_logits']

        # Calculate losses
        if 'CTC' in opt.Prediction:
            # ... [CTC loss calculation] ...
        else:
            target = text[:, 1:]  # without [GO] Symbol
            ocr_loss = ocr_criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
            
        # Language classification loss
        lang_loss = lang_criterion(lang_logits, lang_labels)
        
        # Combined loss
        total_loss = ocr_loss + 0.5 * lang_loss  # Weighted sum

        model.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()

        loss_avg.add(total_loss)

        # validation part
        if (iteration + 1) % opt.valInterval == 0 or iteration == 0:
            elapsed_time = time.time() - start_time
            with open(f'./saved_models/{opt.exp_name}/log_train.txt', 'a', encoding="utf-8") as log:
                model.eval()
                
                with torch.no_grad():
                    # Update validation function to return language accuracy
                    valid_loss, current_accuracy, current_norm_ED, lang_accuracy, preds, confidence_score, labels, infer_time, length_of_data = validation(
                        model, ocr_criterion, lang_criterion, valid_loader, converter, opt)
                    
                model.train()
                
                # ... [logging remains similar, but add language accuracy] ...
                lang_accuracy_log = f'{"Language_accuracy":17s}: {lang_accuracy:0.3f}'
                
                # ... [rest of validation logging] ...

        # ... [model saving remains unchanged] ...

# ... [rest of train.py remains unchanged] ...

################### test.py #########################
# Note: This is a new file for validation function with language classification
import torch
import numpy as np
from tqdm import tqdm
from utils import CTCLabelConverter, AttnLabelConverter

def validation(model, ocr_criterion, lang_criterion, loader, converter, opt):
    """ Evaluation function with language classification """
    n_correct = 0
    norm_ED = 0
    length_of_data = 0
    infer_time = 0
    valid_ocr_loss = 0
    valid_lang_loss = 0
    lang_correct = 0
    total_samples = 0
    
    # For language classification
    lang_conf_matrix = np.zeros((4, 4), dtype=int)  # Kurdish, Arabic, English, Number
    
    for image_tensors, labels, lang_labels in tqdm(loader):
        batch_size = image_tensors.size(0)
        image = image_tensors.to(device)
        text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)
        lang_labels = torch.LongTensor(lang_labels).to(device)
        
        # Forward pass
        model_outputs = model(image, text, is_train=False)
        preds = model_outputs['prediction']
        lang_logits = model_outputs['lang_logits']
        
        # OCR loss calculation
        if 'CTC' in opt.Prediction:
            # ... [CTC loss calculation] ...
        else:
            target = text[:, 1:]
            ocr_loss = ocr_criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
        
        # Language classification loss
        lang_loss = lang_criterion(lang_logits, lang_labels)
        
        valid_ocr_loss += ocr_loss.item()
        valid_lang_loss += lang_loss.item()
        
        # Calculate OCR accuracy
        if 'CTC' in opt.Prediction:
            # ... [CTC decoding] ...
        else:
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length)
        
        # Calculate language accuracy
        _, lang_preds = lang_logits.max(1)
        lang_correct += (lang_preds == lang_labels).sum().item()
        
        # Update confusion matrix
        for t, p in zip(lang_labels.cpu().numpy(), lang_preds.cpu().numpy()):
            lang_conf_matrix[t][p] += 1
        
        # ... [rest of OCR accuracy calculation] ...
        
        total_samples += batch_size
    
    # Calculate metrics
    valid_ocr_loss /= len(loader)
    valid_lang_loss /= len(loader)
    valid_total_loss = valid_ocr_loss + 0.5 * valid_lang_loss
    
    accuracy = n_correct / float(total_samples)
    norm_ED = norm_ED / float(total_samples)  # if you use ED
    lang_accuracy = lang_correct / float(total_samples)
    
    # Print confusion matrix
    print("Language Confusion Matrix:")
    print("     Kurdish  Arabic  English  Number")
    for i, row in enumerate(lang_conf_matrix):
        print(f"{['Kurdish','Arabic','English','Number'][i]}: {row}")
    
    return (
        valid_total_loss, accuracy, norm_ED, lang_accuracy, 
        preds_str, confidence_score, labels, infer_time, length_of_data
    )

################### demo.py #########################
# ... [previous imports remain unchanged] ...

def demo(opt):
    """ model configuration """
    # ... [previous converter setup] ...
    
    # Load model
    model = Model(opt)
    # ... [previous model loading] ...
    
    # Class names
    lang_classes = ['Kurdish', 'Arabic', 'English', 'Number']
    
    # ... [previous data loading] ...
    
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            # ... [previous setup] ...
            
            # Forward pass
            model_outputs = model(image, text_for_pred, is_train=False)
            preds = model_outputs['prediction']
            lang_logits = model_outputs['lang_logits']
            attention_weights = model_outputs['attention_weights']
            
            # Language prediction
            lang_probs = F.softmax(lang_logits, dim=1)
            _, lang_preds = lang_logits.max(1)
            
            # ... [previous OCR processing] ...
            
            for i, (img_name, pred, pred_max_prob, lang_pred) in enumerate(zip(image_path_list, preds_str, preds_max_prob, lang_preds)):
                # ... [previous OCR processing] ...
                
                # Get language prediction
                lang_class = lang_pred.item()
                lang_name = lang_classes[lang_class]
                lang_prob = lang_probs[i][lang_class].item()
                
                # ... [rest of processing] ...
                
                # Add language to output
                print(f'{img_name:25s}\t{pred:25s}\t{lang_name} ({lang_prob:.2f})\t{confidence_score:0.4f}')
                
                # Visualize attention if using attention-based prediction
                if 'Attn' in opt.Prediction and attention_weights is not None:
                    # ... [attention visualization remains unchanged] ...

# ... [rest of demo.py remains unchanged] ...
```
But I have to admit, tasks lists are my favorite:

- [x] This is a complete item
- [ ] This is an incomplete item
