import math
import os
import time

import cv2
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing

import torch
import torch.optim as optim
import torch.nn as nn
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from transformers import AutoModel,ViTModel,ViTFeatureExtractor
from transformers import AutoTokenizer
from transformers.optimization import get_cosine_schedule_with_warmup

from tqdm import tqdm

device = torch.device('mps')
data_path = './data/'
tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large", padding_side = 'left')
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-large-patch32-384')

def reset_weights(m):
    '''
        Try resetting model weights to avoid
        weight leakage.
    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()

def calc_tour_acc(pred, label):
    _, idx = pred.max(1)
    
    acc = torch.eq(idx, label).sum().item() / idx.size()[0] 
    x = label.cpu().numpy()
    y = idx.cpu().numpy()
    f1_acc = f1_score(x, y, average='weighted')
    return acc,f1_acc

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


class CategoryDataset(Dataset):
    def __init__(self, text:np.array, image_path, cats3, tokenizer, feature_extractor, max_len)-> None:
        self.text = text
        self.image_path = image_path
        self.cats3 = cats3
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.max_len = max_len
    def __len__(self):
        return len(self.text)
    def __getitem__(self, item):
        text = str(self.text[item])
        image_path = os.path.join(data_path,str(self.image_path[item])[2:])
        image = cv2.imread(image_path)
        cat3 = self.cats3[item]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding = 'max_length',
            truncation = True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        image_feature = self.feature_extractor(images=image, return_tensors="pt")
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'pixel_values': image_feature['pixel_values'][0],
            'cats3': torch.tensor(cat3, dtype=torch.long)
        }

def create_data_loader(df, tokenizer, feature_extractor, max_len, batch_size, shuffle_= False):
    ds = CategoryDataset(
        text=df.overview.to_numpy(),
        image_path = df.img_path.to_numpy(),
        cats3=df.cat3.to_numpy(),
        tokenizer=tokenizer,
        feature_extractor = feature_extractor,
        max_len=max_len
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4,
        shuffle = shuffle_
    )


class TourClassifier(nn.Module):
    '''
        Simple Convolutional Neural Network
    '''
    def __init__(self, n_classes3, text_model_name, image_model_name):
        super(TourClassifier, self).__init__()
        self.text_model = AutoModel.from_pretrained(text_model_name).to(device)
        self.image_model = ViTModel.from_pretrained(image_model_name).to(device)
        self.text_model.gradient_checkpointing_enable()  
        self.image_model.gradient_checkpointing_enable()

        self.drop = nn.Dropout(p=0.1)

        def get_cls(target_size):
            return nn.Sequential(
                nn.Linear(self.text_model.config.hidden_size, self.text_model.config.hidden_size),
                nn.LayerNorm(self.text_model.config.hidden_size),
                nn.ReLU(),
                nn.Linear(self.text_model.config.hidden_size, 512), # 512 수정 필요 -> 변수로
                nn.Dropout(p = 0.1),
                nn.ReLU(),
                nn.Linear(512, target_size)
            )
        self.cls3 = get_cls(n_classes3)

    def forward(self, input_ids, attention_mask, pixel_values):
        text_output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        image_output = self.image_model(pixel_values = pixel_values)
        concat_outputs = torch.cat([text_output.last_hidden_state, image_output.last_hidden_state],1)
        #config hidden size 일치해야함
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.text_model.config.hidden_size, nhead=8).to(device)
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2).to(device)

        outputs = transformer_encoder(concat_outputs)
        #cls token 
        outputs = outputs[:,0]
        output = self.drop(outputs)
        out3 = self.cls3(output)

        return out3
  
def train_epoch(model,data_loader,loss_fn,optimizer,device,scheduler,n_examples,epoch):
    batch_time = AverageMeter()     
    data_time = AverageMeter()      
    losses = AverageMeter()         
    accuracies = AverageMeter()
    f1_accuracies = AverageMeter()
  
    sent_count = AverageMeter()   
    

    start = end = time.time()

    model = model.train()
    correct_predictions = 0
    for step,d in enumerate(data_loader): 
        data_time.update(time.time() - end)
        batch_size = d["input_ids"].size(0) 
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        pixel_values = d['pixel_values'].to(device)
        
        cats3 = d["cats3"].to(device)
        outputs3 = model(
          input_ids=input_ids,
          attention_mask=attention_mask,
          pixel_values=pixel_values
        )
        _, preds = torch.max(outputs3, dim=1)
        
        # loss_fn = nn.CrossEntropyLoss().to(device)

        loss3 = loss_fn(outputs3, cats3)
        loss = loss3
#         loss = loss1 * 0.05 + loss2 * 0.1 + loss3 * 0.85

        correct_predictions += torch.sum(preds == cats3)
        losses.update(loss.item(), batch_size)

        # Perform backward pass
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Perform optimization
        optimizer.step()
        scheduler.step()
        # Zero the gradients
        optimizer.zero_grad()

        batch_time.update(time.time() - end)
        end = time.time()

        sent_count.update(batch_size)

        # Print statistics
        if step % 200 == 0 or step == (len(data_loader)-1):
            acc,f1_acc = calc_tour_acc(outputs3, cats3)
            accuracies.update(acc, batch_size)
            f1_accuracies.update(f1_acc, batch_size)
            
            print('Epoch: [{0}][{1}/{2}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.3f}({loss.avg:.3f}) '
                  'Acc: {acc.val:.3f}({acc.avg:.3f}) '
                  'f1_Acc: {f1_acc.val:.3f}({f1_acc.avg:.3f}) ' 
                  'sent/s {sent_s:.0f} '
                  .format(
                  epoch, step+1, len(data_loader),
                  data_time=data_time, loss=losses,
                  acc=accuracies,
                  f1_acc=f1_accuracies,
                  remain=timeSince(start, float(step+1)/len(data_loader)),
                  sent_s=sent_count.avg/batch_time.avg
                  ))

    return correct_predictions.double() / n_examples, losses.avg

def validate(model,data_loader,loss_fn,optimizer,device,scheduler,n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    cnt = 0
    for d in tqdm(data_loader):
        with torch.no_grad():
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            pixel_values = d['pixel_values'].to(device)

            cats3 = d["cats3"].to(device)
            
            outputs3 = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values
            )
            _, preds = torch.max(outputs3, dim=1)

            loss3 = loss_fn(outputs3, cats3)
            loss = loss3
#             loss = loss1 * 0.05 + loss2 * 0.1 + loss3 * 0.85

            correct_predictions += torch.sum(preds == cats3)
            losses.append(loss.item())
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if cnt == 0:
                cnt +=1
                outputs3_arr = outputs3
                cats3_arr = cats3
            else:
                outputs3_arr = torch.cat([outputs3_arr, outputs3],0)
                cats3_arr = torch.cat([cats3_arr, cats3],0)
    acc,f1_acc = calc_tour_acc(outputs3_arr, cats3_arr)
    return f1_acc, np.mean(losses)


"""# Inference"""

class InferenceCategoryDataset(Dataset):
    def __init__(self, text, image_path, tokenizer, feature_extractor, max_len):
        self.text = text
        self.image_path = image_path
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.max_len = max_len
    def __len__(self):
        return len(self.text)
    def __getitem__(self, item):
        text = str(self.text[item])
        image_path = os.path.join(data_path,str(self.image_path[item])[2:])
        image = cv2.imread(image_path)
        encoding = self.tokenizer.encode_plus(
          text,
          add_special_tokens=True,
          max_length=self.max_len,
          return_token_type_ids=False,
          padding = 'max_length',
          truncation = True,
          return_attention_mask=True,
          return_tensors='pt',
        )
        image_feature = self.feature_extractor(images=image, return_tensors="pt")
        return {
          'input_ids': encoding['input_ids'].flatten(),
          'attention_mask': encoding['attention_mask'].flatten(),
          'pixel_values': image_feature['pixel_values'][0],
        }

def create_inference_data_loader(df, tokenizer, feature_extractor, max_len, batch_size, shuffle_=False):
    ds = InferenceCategoryDataset(
        text=df.overview.to_numpy(),
        image_path = df.img_path.to_numpy(),
        tokenizer=tokenizer,
        feature_extractor = feature_extractor,
        max_len=max_len
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4,
        shuffle = shuffle_
    )

def inference(model,data_loader,device,n_examples):
    model = model.eval()
    preds_arr3 = []
    for d in tqdm(data_loader):
        with torch.no_grad():
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            pixel_values = d['pixel_values'].to(device)

            outputs3 = model(    
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values
            )
            _, preds3 = torch.max(outputs3, dim=1)
            preds_arr3.append(preds3.cpu().numpy())

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    return preds_arr3

if __name__ == '__main__':

    # Configuration options
    k_folds = 5
    num_epochs = 1
    loss_function = nn.CrossEntropyLoss().to(device)
    
    # For fold results
    results = {}
    
    # Set fixed random number seed
    torch.manual_seed(42)
  
    # Prepare MNIST dataset by concatenating Train/Test part; we split later.
    # dataset_train_part = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor(), train=True)
    # dataset_test_part = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor(), train=False)
    # dataset = ConcatDataset([dataset_train_part, dataset_test_part])
    dataset = pd.read_csv(f'{data_path}train.csv')
    le = preprocessing.LabelEncoder()
    le.fit(dataset.cat3.values)
    dataset.cat3 = le.transform(dataset.cat3.values)

    # Target encoding


    # Define the K-fold Cross Validator
    # kfold = KFold(n_splits=k_folds, shuffle=True)
    kfold = StratifiedKFold(n_splits=k_folds, random_state=42, shuffle=True)
        
    # Start print
    print('--------------------------------')

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, valid_ids) in enumerate(kfold.split(dataset, dataset['cat3'])):

        train_data_loader = create_data_loader(
                                dataset.iloc[train_ids], 
                                tokenizer, feature_extractor, 
                                256, 16, shuffle_=True)
        valid_data_loader = create_data_loader(
                                dataset.iloc[valid_ids], 
                                tokenizer, feature_extractor, 
                                256, 16)

        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')
        
        # Sample elements randomly from a given list of ids, no replacement.
        # train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        # test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
            
        # Define data loaders for training and testing data in this fold
        # trainloader = torch.utils.data.DataLoader(
        #                 dataset, 
        #                 batch_size=10, sampler=train_subsampler)
        # testloader = torch.utils.data.DataLoader(
        #                 dataset,
        #                 batch_size=10, sampler=test_subsampler)
        
        # Init the neural network
        # network = SimpleConvNet() # -> Need to modify
        network = TourClassifier(n_classes3 = 128,
                          text_model_name = 'klue/roberta-large',
                          image_model_name = 'google/vit-large-patch32-384'
                          ).to(device)
        network.apply(reset_weights)
        total_steps = len(train_data_loader) * num_epochs    
        # Initialize optimizer
        # optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
        optimizer = optim.AdamW(network.parameters(), lr= 3e-5)
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps = int(total_steps*0.1),
                                                num_training_steps = total_steps
                                               )
        
        # 여기까지 했음 10월 30일 8시 25분
        # commit here 30, Oct, 8:25 AM 

        # Run the training loop for defined number of epochs
        for epoch in range(0, num_epochs):
            
            # Print epoch
            print(f'Starting epoch {epoch+1}')
            print('-' * 20)
            print(f'Epoch {epoch+1}/{num_epochs} | Fold {fold+1}/{k_folds}')
            print('-' * 20)
            
            # Set current loss value
            current_loss = 0.0
            max_acc = 0
            train_acc, train_loss = train_epoch(network,
                                           train_data_loader,
                                           loss_function,
                                           optimizer,
                                           device,
                                           scheduler,
                                           len(dataset.iloc[train_ids]),
                                           epoch)
            print('start validate |')
            validate_acc, validate_loss = validate(network,
                                               valid_data_loader,
                                               loss_function,
                                               optimizer,
                                               device,
                                               scheduler,
                                               len(dataset.iloc[valid_ids])
                                              )
            if epoch > num_epochs-5 and epoch < num_epochs:
                # 에폭 - 5 전과 에폭 끝나기 직전은 저장 X -> 속도 올려줄거임
                if validate_acc > max_acc:
                    print('update model')
                    save_path = f'./model-fold-{fold}.pth'
                    save_path = f'./epoch-{epoch}_fold-{fold}.pt'
                    max_acc = validate_acc
                    torch.save(network.state_dict(),save_path)
#               if validate_acc > max_acc:
#               max_acc = validate_acc
#               torch.save(model.state_dict(),f'epoch|{epoch}_fold|{fold}.pt')

            print(f'Train loss {train_loss} accuracy {train_acc}')
            print(f'Validate loss {validate_loss} accuracy {validate_acc}')
            print("")
            print("")

    
        # # Process is complete.
        # print('Training process has finished. Saving trained model.')

        # Print about testing
    print('Starting testing')
    
    # tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large", padding_side = 'left')
    # feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-large-patch32-384')
    # model = TourClassifier(n_classes3 = 128,
                            # text_model_name = 'klue/roberta-large',
                            # image_model_name = 'google/vit-large-patch32-384'
                            # ).to(device)
    test = pd.read_csv(f'{data_path}test.csv')

    saved_model_list = [file for file in os.listdir(data_path) if file.endswith('pt')]
    for s in saved_model_list:
        print(f'Load model : f{s}')
        model = TourClassifier()
        model.load_state_dict(torch.load(data_path+'/'+s))
        eval_data_loader = create_inference_data_loader(test, tokenizer, feature_extractor, 256, 1)

        print('inference')
        # inference에서 model.eval
        preds_arr3 = inference(
            model,
            eval_data_loader,
            device,
            len(test)
            )
        print('Create submission csv file')
        sample_submission = pd.read_csv(f'{data_path}/sample_submission.csv')

        for arr in range(len(preds_arr3)):
            submission = sample_submission.copy()
            submission.loc[arr,'cat3'] = le.classes_[preds_arr3[arr][0]]
        submission.to_csv(f'{data_path}/{s[:-3]}_submission.csv')
        print(f"complete saving {s}'s submission file")
        # Saving the model
################################################################
        # Evaluationfor this fold
    
