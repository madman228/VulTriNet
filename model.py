import os
import lap
import torch
import numpy
import pickle
import numpy as np
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from prettytable import PrettyTable
from torch.cuda.amp import autocast
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import multilabel_confusion_matrix
import time
import torch.utils.model_zoo as model_zoo
import math


class SEBlock(nn.Module):
    """ 通道注意力Squeeze-and-Excitation """
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)  

class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
        gamma 
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        
        kernel_size = int(abs((math.log(channel, 2) + 1) / 2))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding= (kernel_size-1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
 
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
 
        # Multi-scale information fusion
        y = self.sigmoid(y)
 
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)  
        y = self.conv(y)
        return x * self.sigmoid(y)  

class EnhancedVulnDetect(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 初始融合 + SE
        self.fuse = nn.Sequential(
            nn.Conv2d(3, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        def attn_block(in_c, out_c):
            return nn.Sequential(
                # nn.Conv2d(in_c, in_c, 3, stride=1, padding=1, groups=in_c),  
                nn.Conv2d(in_c, out_c, 3, stride=1, padding=1),  
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
                #SEBlock(out_c),
                eca_layer(out_c),
                SpatialAttention(),
                nn.MaxPool2d(2) 
            )

        self.backbone = nn.Sequential(
            attn_block(64, 128),   
            attn_block(128, 256),  
            attn_block(256, 512),  
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            #nn.Dropout(0.3),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.fuse(x)
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


    
def sava_data(filename, data):
    print("Begin to save data：", filename)
    with open(filename, 'w') as f:
        f.write(str(data))  
    # f = open(filename, 'wb')
    # f.write(str(data))
    # # pickle.dump(data, f)
    # f.close()

def load_data(filename):
    print("Begin to load data：", filename)
    f = open(filename, 'rb')
    data = pickle.load(f)
    f.close()

    return data

def get_accuracy(labels, prediction):    
    cm = confusion_matrix(labels, prediction)
    #sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    def linear_assignment(cost_matrix):    
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    def _make_cost_m(cm):
        s = np.max(cm)
        return (- cm + s)
    indexes = linear_assignment(_make_cost_m(cm))
    js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
    cm2 = cm[:, js]    
    accuracy = np.trace(cm2) / np.sum(cm2)
    return accuracy 

def get_MCM_score(labels, predictions):
    accuracy = get_accuracy(labels, predictions)
    #precision, recall, f_score, true_sum, MCM = precision_recall_fscore_support(labels, predictions,average='macro')
    precision, recall, f_score, true_sum  = precision_recall_fscore_support(labels, predictions,average='macro')
    MCM = multilabel_confusion_matrix(labels, predictions)
    tn = MCM[:, 0, 0]
    fp = MCM[:, 0, 1]
    fn = MCM[:, 1, 0] 
    tp = MCM[:, 1, 1] 
    fpr_array = fp / (fp + tn)
    fnr_array = fn / (tp + fn)
    f1_array = 2 * tp / (2 * tp + fp + fn)
    sum_array = fn + tp
    M_fpr = fpr_array.mean()
    M_fnr = fnr_array.mean()
    M_f1 = f1_array.mean()
    W_fpr = (fpr_array * sum_array).sum() / sum( sum_array )
    W_fnr = (fnr_array * sum_array).sum() / sum( sum_array )
    W_f1 = (f1_array * sum_array).sum() / sum( sum_array )
    return {
        "M_fpr": format(M_fpr * 100, '.3f'),
        "M_fnr": format(M_fnr * 100, '.3f'),
        "M_f1" : format(M_f1 * 100, '.3f'),
        "W_fpr": format(W_fpr * 100, '.3f'),
        "W_fnr": format(W_fnr * 100, '.3f'),
        "W_f1" : format(W_f1 * 100, '.3f'),
        "ACC"  : format(accuracy * 100, '.3f'),
        "MCM" : MCM
    }

class TraditionalDataset(Dataset):
    def __init__(self, texts, targets, max_len, hidden_size):
        self.texts = texts
        self.targets = targets
        self.max_len = max_len
        self.hidden_size = hidden_size

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        feature = self.texts[idx]
        target = self.targets[idx]
        vectors = numpy.zeros(shape=(3,self.max_len,self.hidden_size))
        for j in range(3):
            current_data = feature[j]
            current_data = np.array(current_data)
            #print(current_data.shape)
            #print(current_data.shape[1])
            if current_data.shape[0] > self.max_len:
                for i in range(self.max_len):
                    for k in range(128):
                        vectors[j][i][k] = feature[j][i][k]
            else:
                for i in range(current_data.shape[0]):
                    for k in range(128):
                        vectors[j][i][k] = feature[j][i][k]
        #print(vectors.shape)
        #print(target)
        return {
            'vector': vectors,
            'targets': torch.tensor(target, dtype=torch.long)
        }

class TextCNN(nn.Module):
    def __init__(self, hidden_size):
        super(TextCNN, self).__init__()
        self.filter_sizes = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)           
        self.num_filters = 32                                        
        classifier_dropout = 0.1
        self.convs = nn.ModuleList(
            [nn.Conv2d(3, self.num_filters, (k, hidden_size)) for k in self.filter_sizes])
        self.dropout = nn.Dropout(classifier_dropout)
        num_classes = 2
        self.fc = nn.Linear(self.num_filters * len(self.filter_sizes), num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = x.float()
        # out = out.unsqueeze(1)
        hidden_state = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(hidden_state)
        out = self.fc(out)
        return out, hidden_state

class CNN_Classifier():
    #max_len = 100
    def __init__(self, max_len=225, n_classes=2, epochs=100, batch_size = 32, learning_rate = 0.001, \
                    result_save_path = "/results", item_num = 0, hidden_size = 128):
        #self.model = TextCNN(hidden_size)
        self.model = EnhancedVulnDetect()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model.to(self.device)
        self.hidden_size = hidden_size
        result_save_path = result_save_path + "/" if result_save_path[-1]!="/" else result_save_path
        if not os.path.exists(result_save_path): os.makedirs(result_save_path)
        self.result_save_path = result_save_path + str(item_num) + "_epo" + str(epochs) + "_bat" + str(batch_size)+ "_maxlen" + str(max_len) + "result" + ".txt"

    def preparation(self, X_train,  y_train, X_valid, y_valid):
        # create datasets
        self.train_set = TraditionalDataset(X_train, y_train, self.max_len, self.hidden_size)
        self.valid_set = TraditionalDataset(X_valid, y_valid, self.max_len, self.hidden_size)

        # create data loaders
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=True)
     
        # helpers initialization
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, correct_bias=False)
        
        #self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[100, 150], gamma=0.1) 
        self.scheduler = get_linear_schedule_with_warmup(
             self.optimizer,
             num_warmup_steps=0,
             #num_warmup_steps = len(self.train_loader) * 20,
             num_training_steps=len(self.train_loader) * self.epochs
         )
        self.loss_fn = torch.nn.CrossEntropyLoss().to(self.device)

    def fit(self):
        self.model = self.model.train()
        losses = []
        labels = []
        predictions = []
        scaler = torch.cuda.amp.GradScaler()
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for i, data in progress_bar:
            self.optimizer.zero_grad()
            vectors = data["vector"].to(self.device)
            vectors = vectors.float() 

            targets = data["targets"].to(self.device)
            with autocast():
                #outputs,_  = self.model( vectors )
                outputs  = self.model( vectors )
                #print('outputs:')
                loss = self.loss_fn(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            preds = torch.argmax(outputs, dim=1).flatten()           
            
            losses.append(loss.item())
            predictions += list(np.array(preds.cpu()))   
            labels += list(np.array(targets.cpu()))      

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scheduler.step()
            progress_bar.set_description(
                f'loss: {loss.item():.3f}, acc : {(torch.sum(preds == targets)/len(targets)):.3f}')
        train_loss = np.mean(losses)
        score_dict = get_MCM_score(labels, predictions)
        return train_loss, score_dict

    def eval(self):
        print("start evaluating...")
        self.model = self.model.eval()
        losses = []
        pre = []
        label = []
        correct_predictions = 0
        progress_bar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader))

        all_time = 0
        with torch.no_grad():
            for _, data in progress_bar:
                vectors = data["vector"].to(self.device)
                vectors = vectors.float() 
                targets = data["targets"].to(self.device)
                #outputs, _ = self.model(vectors)
                #print(len(vectors))
                outputs = self.model(vectors)
                
                loss = self.loss_fn(outputs, targets)
                preds = torch.argmax(outputs, dim=1).flatten()
                #print(preds)
                #print(targets)
                correct_predictions += torch.sum(preds == targets)

                pre += list(np.array(preds.cpu()))
                label += list(np.array(targets.cpu()))
                
                losses.append(loss.item())
                progress_bar.set_description(
                f'loss: {loss.item():.3f}, acc : {(torch.sum(preds == targets)/len(targets)):.3f}')
        val_acc = correct_predictions.double() / len(self.valid_set)
        print("val_acc : ",val_acc)
        score_dict = get_MCM_score(label, pre)
        val_loss = np.mean(losses)
        return val_loss, score_dict

    def train(self):
        learning_record_dict = {}
        train_table = PrettyTable(['typ', 'epo', 'loss', 'M_fpr', 'M_fnr', 'M_f1', 'W_fpr', 'W_fnr', 'W_f1', 'ACC'])
        test_table = PrettyTable(['typ', 'epo', 'loss', 'M_fpr', 'M_fnr', 'M_f1', 'W_fpr', 'W_fnr', 'W_f1', 'ACC'])
        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            train_loss, train_score = self.fit()
            train_table.add_row(["tra", str(epoch+1), format(train_loss, '.4f')] + [train_score[j] for j in train_score if j != "MCM"])
            print(train_table)

            val_loss, val_score = self.eval()
            test_table.add_row(["val", str(epoch+1), format(val_loss, '.4f')] + [val_score[j] for j in val_score if j != "MCM"])
            print(test_table)
            print("\n")
            learning_record_dict[epoch] = {'train_loss': train_loss, 'val_loss': val_loss, \
                    "train_score": train_score, "val_score": val_score}
            sava_data(self.result_save_path, learning_record_dict)
            print("\n")
