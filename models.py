import csv
from pathlib import Path
import pandas as pd
import numpy as np
import torch as torch
from torch import nn
from torch.nn import functional as F
import sklearn
import time
import wandb
import uuid
from utils import *
import math, copy, time
from functools import reduce
from notion.client import NotionClient
from notion.block import *
import json
from tqdm import trange
import gc

with open("secrets.json") as s: # have a json file in same dir with following info for secrets:
    secrets = json.load(s)
NAPI = numerapi.NumerAPI(verbosity="info",public_id=secrets["NUMERAPI_PUBLIC_ID"],secret_key=secrets["NUMERAPI_SECRET_KEY"])
notion_client = NotionClient(token_v2=secrets["NOTION_TOKENV2"])

class NumeraiTrainData(torch.utils.data.Dataset):
    def __init__(self,training_data, feature_names):
        start = time.time()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training_data = training_data # the train image labels
        #era to int
        self.train_era_indexes = self.training_data["era"].unique()
        self.training_data["era"]=self.training_data["era"].map(lambda a: np.where(self.train_era_indexes==a)[0][0]+1)

        self.x_train = self.training_data[feature_names].to_numpy()
        self.target = self.training_data["target"]
        self.y_train = self.target.to_numpy()
        self.x_train = torch.from_numpy(self.x_train).to(device).float().view(-1,1, len(feature_names))
        self.y_train = torch.from_numpy(self.y_train).to(device).float().view(-1,1)
        end = time.time()
        print(f"numerai train init took {end - start}")
    
    def __len__(self):
        return len(self.training_data)
    
    def __getitem__(self, indx):
        return self.x_train[indx], self.y_train[indx]
        
class NumeraiTournamentData(torch.utils.data.Dataset):
    def __init__(self,tournament_data, feature_names):
        start = time.time()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tournament_data = tournament_data # the train image labels
        #era to int
        self.tourn_era_indexes = self.tournament_data["era"].unique()
        self.tournament_data["era"]=self.tournament_data["era"].map(lambda a: np.where(self.tourn_era_indexes==a)[0][0]+1)
        self.x = torch.from_numpy(self.tournament_data[feature_names].to_numpy()).to(device).float()
    
        end = time.time()
        print(f"numerai tourn init took {end - start}")

    def __len__(self):
        return len(self.tournament_data)
    
    def __getitem__(self, indx):
        return self.x[indx]

class NumeraiTournamentModel(nn.Module):
    """Subclass this model for numerai training stuff. Please make sure that your numerai datasets are in the subdirectory right next to your notebook/script. "numerai_datasets/"
    - round: what round is it for numerai
    - use_wandb: do you want to use wandb to provide analytics?
    - override_dataset: have you already gotten the dataset in memory? If so you should set this to true and pass in your training_data,tournament_data, and feature_names to skip the unneccesary, time consuming step of reinitiallizing the dataset
    """
    def __init__(self, train_dataloader,feature_names, use_wandb=True, custom_config={}):
        super(NumeraiTournamentModel,self).__init__()
        round=numerapi.NumerAPI(verbosity="INFO").get_current_round()
        self.notion_numerai_table = notion_client.get_block("https://www.notion.so/998f56ae54164c25afe3f00e340ee9c0?v=4adb401899d249b7a187e5432dfc4628") 
        print("does your model already have a notion data point? (y/n)")
        already_has_notion_dp = input()
        if already_has_notion_dp.lower().__contains__("y"):
            print("Paste in your notion data point link: ")
            self.notion_model_page = notion_client.get_block(input())
            self.mid = self.notion_model_page.model_id
        else: 
            self.mid = str(uuid.uuid4())
            self.notion_model_page = self.notion_numerai_table.collection.add_row()
            #model id is random unique identifier 
            print("Enter a title for this model: ")
            self.notion_model_page.title = input()
            print("what loss function will you be relying on mostly? ")
            self.notion_model_page.model_id = self.mid
            self.notion_model_page.loss = input()

        self.use_wandb = use_wandb
        self.round = round
        custom_config['modelID'] = self.mid
        self.config = CustomConfig(init_dict=custom_config)
        self.feature_names = feature_names
        self.train_dataloader = train_dataloader
        pathlib.Path(f'results/round{self.round}/models/').mkdir(parents=True, exist_ok=True) 
        pathlib.Path(f'results/round{self.round}/submissions/').mkdir(parents=True, exist_ok=True)
        self.log(f"MODEL ID: {self.mid}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def initwandb(self):
        self.run=wandb.init(project='numerai-ryendu',config=self.config.dict_version)
        self.forward(torch.from_numpy(np.random.rand(158,1,self.features_len)).float())
        wandb.watch(self)
        self.notion_model_page.wandb_link = self.run._get_run_url()

    def log(self,text,print_=True):
        if print_:
            print(text)
        txt = self.notion_model_page.children.add_new(TextBlock)
        txt.title = text

    def forward(self,**kwargs):
        raise NotImplementedError("Forward is not implemented in the superclass. Subclass NumeraiTournamentModel to use it.")

    def train_(self,epochs,loss_fn,optimizer_fn=torch.optim.Adam,lb_create_graph=False,init_lr=0.01,gamma=0.99):
        """
        train the stuff.
        - input the model. THe model must have a dictionary called configs with the model configs for wandb. if not, set override_model_config_params=True and pass in all your configs through this func.
        """
        self.train(True)
        if self.use_wandb:
            config = self.run.config
        else: 
            config = self.config
        optimizer = optimizer_fn(self.parameters(),lr=init_lr)
        lrs=[init_lr]
        losses=[]
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        for e in range(epochs):
            start = time.time()
            batches = len((self.train_dataloader))
            batch_trange = tqdm(total=batches,leave=True,position=0)
            for b, (inp, real) in enumerate(iter(self.train_dataloader)):
                optimizer.zero_grad()
                inp = inp
                real = real.view(-1)
                y_pred = self.forward(inp).float().view(-1)
                loss = loss_fn(y_pred,real)
                loss.backward(create_graph=lb_create_graph)
                optimizer.step()
                if self.use_wandb:
                    wandb.log({"loss":float(loss),"lr":lrs[-1] })
                if b % 100 == 0:
                    scheduler.step()
                    lrs.append(optimizer.param_groups[0]["lr"])
                    losses.append(loss)
                    spear_corr = scipy.stats.spearmanr(y_pred.cpu().detach().numpy(),real.cpu().detach().numpy())
                    mse = F.mse_loss(y_pred.float(),real)
                    if self.use_wandb:              
                        wandb.log({"loss": loss.item(),'scipy_spearman':spear_corr.correlation,'mse':mse})                
                    self.notion_model_page.mse = mse.item()
                    self.log(f"Status: Training Batch {b}/{batches} for epoch: {e} lr: {lrs[-1]}, spearman: {spear_corr.correlation}, mse: {mse}", print_=False)
                    batch_trange.set_description(f"spearman: {spear_corr.correlation}, mse: {mse.item()}")
                batch_trange.update(1)
                gc.collect()
                torch.cuda.empty_cache()
            end = time.time()
            self.notion_model_page.epochs = self.notion_model_page.epochs + 1
            if self.use_wandb:
                wandb.log({"epoch":e})
            self.log(f"FINISHED EPOCH {e} with loss: {loss.item()}, time elapsed: {end - start}")


class CitizenModel(nn.Module):
    def __init__(self,layers=8,neurons=256,out=1,batch_normalization=True):
        super(CitizenModel, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(layers):
            if batch_normalization:
                self.layers.append(nn.LazyLinear(neurons))
                self.layers.append(nn.BatchNorm1d(1))
            else:
                self.layers.append(nn.LazyLinear(neurons))
            
        self.outl = nn.LazyLinear(out)
    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
            x = F.leaky_relu(x)
        x = self.outl(x)
        x = torch.sigmoid(x)
        return x

class VanillaNNEnsembleMultiLayer(NumeraiTournamentModel):
    def __init__(self,train_dataloader,feature_names,architecture=[8,6,4], citizen_layers=8,citizen_neurons=256,outp_start=256,use_wandb=False,batch_normalization=True,gamma=0.98,init_lr=0.01,batch_size=158):
        """
            A plain Multilayer Neural Network Ensemble with citizens.

            okay, the architecture parameter is supposed to be an array with the length corresponding to the layers of the forest, and each element corresponding to how many trees in that layer
            citizen layers represents how many layer in each citizen, citizen neurons is how many neurons in each citizen
            outp_start is what is the output shape of each layer of the neurons. The outp_start gets divided by the layer index for each layer to determine the output shape except for last layer which has an output shape of 1
            DO NOT USE THIS MODEL TO TRAIN, USE THE TRAIN FUNCTION OF THIS MODEL!!
        """
        super(VanillaNNEnsembleMultiLayer, self).__init__(train_dataloader,feature_names,use_wandb=use_wandb, custom_config={"gamma":gamma,"initial_lr":init_lr,"batch_size":batch_size,"model_arch":"democratic_multilayer_ensemble","architecture":architecture,"citizen_layers":citizen_layers,"citizen_neurons":citizen_neurons})
        self.configs = {"model_arch":"democratic_multilayer_ensemble","architecture":architecture,"layers":len(architecture),"citizen_neurons":citizen_neurons,"citizen_layers":citizen_layers,"outp_start":outp_start}
        self.stack = nn.ModuleList([nn.ModuleList() for i in range(len(architecture))])
        self.outp_start = outp_start
        self.notion_model_page.architecture = 'vanilla-nn-ensemble-ml'
        for li, trees in enumerate(architecture):
            for ti in range(trees):
                if li == len(architecture) - 1:
                    out_ = 1
                else:
                    out_ = round(outp_start/(li+1))
                self.stack[li].append(CitizenModel(layers=citizen_layers,neurons=citizen_neurons,out=out_,batch_normalization=batch_normalization))

    def forward(self,x):
        
        for layer in self.stack:
            votes=torch.tensor([])
            for citizen in layer:
                res = citizen(x.to(self.device)).view(1,-1)
                votes = torch.cat((votes.to(self.device),res.to(self.device)),dim=0)
            x = torch.mean(votes,dim=0)
        return x

class SingleFeaturePrediction(nn.Module):
    def __init__(self,target_feature_n,features=309,layers=2,neurons=8,out=1):
        super(SingleFeaturePrediction, self).__init__()
        self.layers = nn.ModuleList([nn.LazyLinear(neurons) for _ in range(layers)])
        self.outl = nn.LazyLinear(out)
        self.target_feature_n = target_feature_n
        self.features = features
    def forward_fe(self,x):
        """returns the output of the last hidden layer / second to last layer"""
        x = x.reshape(self.features + 1,1,-1)
        x = torch.cat([x[:self.target_feature_n], x[self.target_feature_n+1:]])
        x = x.view(-1,1,self.features)
        for layer in self.layers:
            x = layer(x)
            x = torch.relu(x)
        return x  
    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
            x = torch.relu(x)
        x = self.outl(x)
        x = torch.sigmoid(x)
        return x 
         
class EnsembleMetaModel(NumeraiTournamentModel):
    def __init__(self,models:nn.ModuleList,train_dataloader,feature_names,use_wandb=False,gamma=0.98,init_lr=0.01,batch_size=158,afe=False):
        """okay, the architecture parameter is supposed to be an array with the length corresponding to the layers of the forest, and each element corresponding to how many trees in that layer
            tree layers represents how many layer in each tree, tree neurons is how many neurons in each tree
            outp_start is what is the output shape of each layer of the neurons. The outp_start gets divided by the layer index for each layer to determine the output shape except for last layer which has an output shape of 1
            democratic_singlelayer_forest_fe: fe stands for feature engineering
        """
        super(EnsembleMetaModel, self).__init__(train_dataloader,feature_names, use_wandb=use_wandb, custom_config={"gamma":gamma,"initial_lr":init_lr,"batch_size":batch_size,"model_arch":"democratic_singlelayer_ensemble","auto_feature_engineering":afe})
        self.models = models
        self.notion_model_page.architecture = 'mixed-meta-ensemble'
        if use_wandb:
            self.initwandb()
        
    def forward(self,x):
        votes=[]
        for model in self.models:
            votes.append(model(x).detach().cpu().numpy())
        votes = np.array(votes)
        pred = np.mean(votes,axis=0)
        x = torch.from_numpy(pred).requires_grad_(True)
        return x

class FeatureEngineeringNN(nn.Module):
    def __init__(self,feature_names,log_fn, sfp_layers=3,sfp_neurons=32):
        super(FeatureEngineeringNN, self).__init__()
        self.stack = nn.ModuleList()
        self.log = log_fn
        self.features = len(feature_names)
        self.feature_names = feature_names
        for i,fe in enumerate(feature_names):
            sfp = SingleFeaturePrediction(target_feature_n=i,features=self.features - 1,layers=sfp_layers,neurons=sfp_neurons,out=1)
            self.stack.append(sfp)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def forward(self,x):
        """ returns the output of the second to last layer / last hidden layer of each sfp. DO NOT USE THIS FUNC TO TRAIN"""
        res = torch.tensor([])
        for sfp in self.stack:
            out = sfp.forward_fe(x).view(-1)
            res = torch.cat((res.to(self.device), out.to(self.device)), dim=0)
        return res

    def train_(self,training_data,epochs=1,lr=0.01,loss_fn=F.mse_loss):
        self.train(True)
        for e in range(epochs):
            losses = [] 
            with tqdm(total=len(self.stack),leave=True,position=0) as fen_progress: 
                for i, fen in enumerate(self.stack):
                    x_train = training_data[self.feature_names]
                    y_train = x_train.pop(self.feature_names[i]).to_numpy()
                    x_train = x_train.to_numpy()
                    batch=get_batch_size(y_train.shape[0])
                    x_train = torch.from_numpy(x_train).view(-1,batch, self.features-1).float().to(self.device)
                    y_train = torch.from_numpy(y_train).view(-1,batch).float().to(self.device)
                    
                    for b in range(batch):
                        optimizer = torch.optim.Adam(fen.parameters(),lr=lr)
                        random_batch = np.random.randint(0,x_train.shape[0]-1)
                        optimizer.zero_grad()
                        inp = x_train[random_batch].view(-1,1,self.features-1)
                        y_pred = fen(inp).float().view(-1,1)
                        y_real = y_train[random_batch].view(-1,1)
                        loss = loss_fn(y_pred,y_real)
                        loss.backward()
                        optimizer.step()
                        
                    losses.append(loss.item())
                    self.log(f"Done with fen {i} with loss {loss.item()}",print_=False)
                    gc.collect()
                    torch.cuda.empty_cache()
                    fen_progress.update(1)
                    fen_progress.set_description(f'loss: {loss.item()} epoch {e}/{epochs}')
            self.log(f"Finished with epoch {e} with average loss {np.mean(losses)}")

class VanillaNNEnsembleSingleLayer(NumeraiTournamentModel):
    def __init__(self,train_dataloader,feature_names,citizens=8,citizen_layers=8,citizen_neurons=256,use_wandb=False,batch_normalization=True,gamma=0.98,init_lr=0.01,batch_size=158):
        """
        **A Plain Neural Network Ensemble that has a single layer of citizens with simple linear citizens. Customize citizen parameters with params that start with citizen_. 
        ##Paramaters
            citizen_layers represents how many layer in each citizen, citizen_neurons is how many neurons in each citizen
        """
        super(VanillaNNEnsembleSingleLayer, self).__init__(train_dataloader,feature_names,use_wandb=use_wandb,custom_config={"gamma":gamma,"initial_lr":init_lr,"batch_size":batch_size,"citizens":citizens,"citizen_layers":citizen_layers,"citizen_neurons":citizen_neurons})
        self.stack = nn.ModuleList()
        self.features = len(feature_names)
        self.notion_model_page.architecture = 'vanilla-nn-ensemble-sl'
        for ci in range(citizens):
            self.stack.append(CitizenModel(layers=citizen_layers,neurons=citizen_neurons,out=1,batch_normalization=batch_normalization))
        if use_wandb:
            self.initwandb()
        
    def forward(self,x):
        votes=torch.tensor([])
        for citizen in self.stack:
            res = citizen(x).view(1,-1)
            votes = torch.cat((votes.to(self.device),res.to(self.device)),dim=0)
        x = torch.mean(votes.to(self.device),dim=0)
        return x

    def train_individually(self,epochs=1,loss_fn=F.mse_loss,lr=0.05):
        self.train(True)
        for e in range(epochs):
            start = time.time()
            batches = len((self.train_dataloader))
            batch_trange = tqdm(total=batches,leave=True,position=0)
            for b, (inp, real) in enumerate(iter(self.train_dataloader)):
                c_losses=[]
                inp = inp.to(self.device).float().view(-1,1,self.features)
                real = real.to(self.device).float().view(-1,1)
                for ci, citizen in enumerate(self.stack):
                    optimizer = torch.optim.Adam(citizen.parameters(),lr=lr)
                    optimizer.zero_grad()
                    y_pred = citizen(inp).float().view(-1,1)
                    loss = loss_fn(y_pred.float(),real)
                    loss.backward()
                    optimizer.step()
                    c_losses.append(loss.item())
                average_citizen_loss=np.mean(c_losses)
                y_pred = self.forward(inp).float().view(-1, 1)
                spear_corr = scipy.stats.spearmanr(y_pred.detach().numpy(),real.detach().numpy())
                mse = F.mse_loss(y_pred.float(),real)
                society_loss = loss_fn(y_pred,real)
                self.log(f"Status: Training Batch {b}/{batches}for epoch: {e} average citizen loss: {average_citizen_loss}, society loss: {society_loss.item()}, soc scipySpearman: {spear_corr.correlation}, soc mse: {mse.item()}",print_=False)
                batch_trange.set_description(f'average citizen loss: {average_citizen_loss}, society scipySpearman: {spear_corr.correlation}, society mse: {mse.item()}')
            end = time.time()
            self.notion_model_page.epochs = self.notion_model_page.epochs + 1
            self.log(f"FINISHED EPOCH {e}, time elapsed: {end - start}")

class ChocolateNNEnsembleSingleLayer(NumeraiTournamentModel):
    def __init__(self, train_dataloader,feature_names,citizens=8,citizen_layers=8,citizen_neurons=256,sfp_layers=3,sfp_neurons=32,use_wandb=False,batch_normalization=True,gamma=0.98,init_lr=0.01,batch_size=158):
        """**A Plain Neural Network Ensemble that has a single layer of citizens with simple linear citizens with the adition of AUTO FEATURE ENGINEERING. Customize citizen parameters with params that start with citizen_. 
            ## When Training
            first train with model.fen.train_(), to train the feature engineering nn.
            then use the built in, basic trans function from the super class, NumeraiTournamentModel to train this model 

            ##Paramaters
            citizen_layers represents how many layer in each citizen, citizen_neurons is how many neurons in each citizen
        """
        super(ChocolateNNEnsembleSingleLayer, self).__init__(train_dataloader,feature_names,use_wandb=use_wandb, custom_config={"gamma":gamma,"initial_lr":init_lr,"batch_size":batch_size,"citizens":citizens,"citizen_layers":citizen_layers,"citizen_neurons":citizen_neurons})
        print("Finished chocolate parent init.")
        self.stack = nn.ModuleList()
        self.batch_size = batch_size
        self.features = len(feature_names)
        self.notion_model_page.architecture = 'chocolate-nn-ensemble-sl'
        self.sfp_neurons = sfp_neurons
        self.fen = FeatureEngineeringNN(feature_names=feature_names,log_fn=self.log,sfp_layers=sfp_layers,sfp_neurons=sfp_neurons).to(self.device)
        for ci in range(citizens):
            self.stack.append(CitizenModel(layers=citizen_layers,neurons=citizen_neurons,out=1,batch_normalization=batch_normalization).to(self.device))
        if use_wandb:
            self.initwandb()
        self.stack = self.stack.to(self.device)
        
    def forward(self,x):
        features_learned = self.fen(x).view(-1,1,self.features * self.sfp_neurons)
        x = torch.cat((x, features_learned), dim=2)
        votes=torch.tensor([],device=self.device)
        for citizen in self.stack:
            res = citizen(x).view(1,-1)
            votes = torch.cat((votes,res),dim=0)
        x = torch.mean(votes,dim=0)
        return x

#OLD ANCIENT STUFF

class YayTransformerModel263_1(nn.Module):
    """transformers are awseome. transformer for stock trading stuff"""
    def __init__(self,nhead=5,num_encoder_layers=3,dim_feedforward=310,n_linear=3,n_linear_neurons=256,dropout_rate=0.15):
        super(YayTransformerModel263_1, self).__init__()
        self.configs = {"model_arch":"transformer",
        "nhead":nhead,"num_encoder_layers":num_encoder_layers,"dim_feedforward":dim_feedforward,"n_linear":n_linear,"n_linear_neurons":n_linear_neurons,"dropout_rate":dropout_rate}
        self.model = nn.ModuleList()
        self.model.append(nn.Transformer(nhead=self.configs.get("nhead"), num_encoder_layers=self.configs.get("num_encoder_layers"), d_model=self.configs.get("dim_feedforward"), dim_feedforward=self.configs.get("dim_feedforward")))
        for i in range(self.configs.get("n_linear")):
            self.model.append(nn.LazyLinear(self.configs.get("n_linear_neurons")) )
            self.model.append(nn.ReLU())
            if i % 2 == 0:
                self.model.append(nn.Dropout(self.configs.get("dropout_rate")))
        self.out_lin = nn.LazyLinear(1)#)
    def forward(self,src, tgt=None):
        for i, layer in enumerate(self.model):
            if i == 0:
                if tgt != None:
                    out = layer(src,tgt)
                else: out = layer(src,src)
            else:
                out = layer (out)
        out = self.out_lin(out)
        out = torch.sigmoid(out)
        return out

class RyenduTransformerModel263_1(nn.Module):
    """transformers are awseome. transformer for stock trading stuff"""
    def __init__(self,nhead=5,num_encoder_layers=5,dim_feedforward=310,n_linear=3,n_linear_neurons=256):
        super(RyenduTransformerModel263_1, self).__init__()
        self.configs = {"model_arch":"transformer",
        "nhead":nhead,"num_encoder_layers":num_encoder_layers,"dim_feedforward":dim_feedforward,"n_linear":n_linear,"n_linear_neurons":n_linear_neurons}
        self.model = nn.ModuleList()
        self.model.append(nn.Transformer(nhead=self.configs.get("nhead"), num_encoder_layers=self.configs.get("num_encoder_layers"), d_model=self.configs.get("dim_feedforward"), dim_feedforward=self.configs.get("dim_feedforward")))
        for i in range(self.configs.get("n_linear")):
            self.model.append(nn.LazyLinear(self.configs.get("n_linear_neurons")))
            if i != 0:
                self.model.append(nn.BatchNorm1d(1))
            self.model.append(nn.ReLU())
        self.out_lin = nn.LazyLinear(1)#)
    def forward(self,src, tgt=None):
        for i, layer in enumerate(self.model):
            if i == 0:
                if tgt != None:
                    out = layer(src,tgt)
                else: out = layer(src,src)
            else:
                out = layer (out)
        out = self.out_lin(out)
        out = torch.sigmoid(out)
        return out

class RyenduCNN1StockModel(nn.Module):
    """cnn for stock trading stuff"""
    def __init__(self,features=310,pool_stride=2,pool_kernel_size=2,n_hidden_linear_layers=3,hidden_linear_layer_neurons=64,n_hidden_conv_layers=3,hidden_conv_layers_neurons=64,pool_every=2,conv_stride=1):
        super(RyenduCNN1StockModel, self).__init__()
        self.features=features
        self.pool_every = pool_every
        self.pool_kernel_size=pool_kernel_size
        self.pool_stride = pool_stride
        self.conv_layers=nn.ModuleList()
        self.linear_layers=nn.ModuleList()
        for i in range(n_hidden_conv_layers):
            if i == 0:
                layer = nn.Conv1d(in_channels=1,out_channels=n_hidden_conv_layers,kernel_size=1,stride=conv_stride)
                self.conv_layers.append(layer)
            if i == n_hidden_conv_layers - 1:
                layer = nn.Conv1d(in_channels=n_hidden_conv_layers,out_channels=64,kernel_size=2+i,stride=conv_stride)
                self.conv_layers.append(layer)
            else:
                layer = nn.Conv1d(in_channels=n_hidden_conv_layers,out_channels=n_hidden_conv_layers,kernel_size=2+i,stride=conv_stride)
                self.conv_layers.append(layer)
            
        for i in range(n_hidden_linear_layers):
            if i == 0:
                in_sample = torch.rand(16,1,self.features)
                for i in self.conv_layers:
                    in_sample = i(in_sample)
                in_ = nn.Flatten()(in_sample).shape[1]
                layer = nn.Linear(in_features=in_,out_features=hidden_linear_layer_neurons)
                self.linear_layers.append(layer)
            if i == n_hidden_conv_layers - 1:
                layer = nn.Linear(in_features=hidden_linear_layer_neurons,out_features=1)
                self.linear_layers.append(layer)
            else:
                layer = nn.Linear(in_features=hidden_linear_layer_neurons,out_features=hidden_linear_layer_neurons)
                self.linear_layers.append(layer)

        self.configs = {"model_type":"ryenduCNN1","pool_stride":pool_stride,"n_hidden_linear_layers":n_hidden_linear_layers,"hidden_linear_layer_neurons":hidden_linear_layer_neurons,"n_hidden_conv_layers":n_hidden_conv_layers,"conv_stride":conv_stride,"hidden_conv_layers_neurons":hidden_conv_layers_neurons,"pool_every":pool_every}

    def forward(self, x):
        for index,layer in enumerate(self.conv_layers):
            x = layer(x)
            x = F.leaky_relu(x)
            if index + 1 % self.pool_every == 0:
                x = F.max_pool1d(x, kernel_size=self.pool_kernel_size, stride=self.pool_stride)
        for index,layer in enumerate(self.linear_layers):
            if index == 0:
                x = nn.Flatten()(x)
            x = layer(x)
            if index != len(self.linear_layers) - 1:
                x = F.leaky_relu(x)
        return x

