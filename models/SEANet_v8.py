import torch as th
import torch 
import torch.nn as nn
import torch.nn.functional as F
import pickle
from transformers import HubertModel, AutoProcessor, Wav2Vec2Model, WavLMModel, AutoModel
from einops import rearrange
from SEANet_v5 import ssl_layer, repeat_features, FCLayer

"""
n_channels: Number of Conv channels

do not input N x L x F

x: Conv Feature Map (N x F x L)
cond: SSL Condition (N x L/320 x 1024)
output:  modulated feature map (N x F x L)
"""
class FiLMLayer(nn.Module):
    def __init__(self, n_channels, visualize=False):
        super().__init__()
        self.n_channels = n_channels
        self.film_gen = nn.Linear(1024, 2*n_channels)
        self.visualize = visualize
        # self.meanpool = nn.AvgPool1d(kernel_size=5)

    def forward(self, x, condition):
        x = rearrange(x,'b f l -> b l f')
        cond = self.film_gen(condition)
        
        ## Avg Pooling 
        if self.visualize: print(cond.shape, "Before Pooling")
        cond = torch.mean(cond, dim=1)

        cond = cond.unsqueeze(1)
        if self.visualize: print(cond.shape, "After Pooling")

        gamma = cond[:,:,:self.n_channels]
        beta = cond[:,:,self.n_channels:]

        # if self.visualize: print(gamma.shape, "GAMMA")
        # Linear Modulation
        x = gamma * x + beta
        x = rearrange(x, 'b l f -> b f l')
        return x
    
# # B x L x Dim
# x = torch.rand(1, 4, 2)
# cond = torch.rand(1, 100, 1024)
# model = FiLMLayer(n_channels=4, visualize=True)

# print(x)
# print(model(x,cond))

""" Total 1.27 M Parameters """
class SEANet_v8(nn.Module):
    
    # def __init__(self, min_dim=8,kmeans_model_path='/home/woongzip/RealTimeBWE/Kmeans/kmeans_modelweight_200.pkl', **kwargs):
    def __init__(self, min_dim=8, kmeans_model_path=None, modelname="wavlm", visualize=False, **kwargs):
        from transformers import AutoProcessor, AutoModel
        super().__init__()
        
        self.visualize = visualize

        # Load Kmeans model
        with open(kmeans_model_path, 'rb') as file:
            self.kmeans = pickle.load(file)
                
        self.min_dim = min_dim
        
        ## Load SSL model
        self.modelname = modelname
        if modelname == 'hubert':
            model_id = "facebook/hubert-large-ls960-ft"
        elif modelname == 'w2v':
            model_id = "facebook/wav2vec2-large-960h-lv60-self"
        elif modelname == 'wavlm':
            model_id = "patrickvonplaten/wavlm-libri-clean-100h-large"
        else:
            raise ValueError("Error: [hubert, w2v, wavlm] required")
        
        self.ssl_model = AutoModel.from_pretrained(model_id)
        self.ssl_processor = AutoProcessor.from_pretrained(model_id)

        # Freeze SSL Parameters            
        for param in self.ssl_model.parameters():
            param.requires_grad = False

        # Feature Extracted SSL Layers
        self.FiLM_e1 = FiLMLayer(n_channels=16, visualize=self.visualize)
        self.FiLM_e2 = FiLMLayer(n_channels=32, visualize=self.visualize)
        self.FiLM_e3 = FiLMLayer(n_channels=64, visualize=self.visualize)
        self.FiLM_e4 = FiLMLayer(n_channels=128, visualize=self.visualize)

        self.FiLM_d1 = FiLMLayer(n_channels=64)
        self.FiLM_d2 = FiLMLayer(n_channels=32)
        self.FiLM_d3 = FiLMLayer(n_channels=16)
        self.FiLM_d4 = FiLMLayer(n_channels=8)

        ## First Conv Layer
        self.conv_in = Conv1d(
            in_channels = 1,
            out_channels = min_dim,
            kernel_size = 7,
            stride = 1
        )

        self.downsampling_factor = 320
        self.encoder = nn.ModuleList([
                                    EncBlock(min_dim*2, 2),
                                    EncBlock(min_dim*4, 4),
                                    EncBlock(min_dim*8, 5),
                                    EncBlock(min_dim*16, 8)                                        
                                    ])
        
        self.conv_bottle1 = Conv1d(
                            in_channels=min_dim*16,
                            out_channels = min_dim*16//4,
                            kernel_size = 7, 
                            stride = 1,
                            )
                        
        self.conv_bottle2 = Conv1d(
                            in_channels=min_dim*16//4,
                            out_channels = min_dim*16,
                            kernel_size = 7,
                            stride = 1,
                            )
        
        self.decoder = nn.ModuleList([
                                    DecBlock(min_dim*8, 8),
                                    DecBlock(min_dim*4, 5),
                                    DecBlock(min_dim*2, 4),
                                    DecBlock(min_dim, 2),
                                    ])
        
        self.conv_out = Conv1d(
            in_channels = min_dim,
            out_channels = 1,
            kernel_size = 7,
            stride = 1,
        )
        
    def forward(self, x, HR):

        input = HR
        #################### Length Adjustment
        ## x and HR has same shape
        ## Match into multiple of downsampling factor
        fragment = torch.randn(0).to(x.device)
        # print(fragment.shape,"shape frag")
        
        if x.dim()== 3: # N x 1 x L
            sig_len = x.shape[2]
            if sig_len % self.downsampling_factor != 0:
                new_len = sig_len // self.downsampling_factor * self.downsampling_factor
                fragment = x[:,:,new_len:].clone().to(x.device)  # 수정된 부분
                # fragment = x[:,:,sig_len:]
                x = x[:,:,:sig_len]
                HR = HR[:,:,:sig_len]
                
        if x.dim()==2:
            sig_len = x.shape[1]
            if sig_len % self.downsampling_factor != 0:
                new_len = sig_len // self.downsampling_factor * self.downsampling_factor
                fragment = x[:,new_len:].clone().to(x.device)  # 수정된 부분
                # fragment = x[:,sig_len:]
                x = x[:,:sig_len]
                HR = HR[:,:sig_len]
                
        while len(x.size()) < 3:
            x = x.unsqueeze(-2)
            HR = HR.unsqueeze(-2)
            # fragment = fragment.unsqueeze(-2).to(x.device)
            
        # print("Input Signal Length: ",sig_len, fragment.shape)
        
        if sig_len % self.downsampling_factor != 0:
            sig_len = sig_len // self.downsampling_factor * self.downsampling_factor
            x = x[:,:,:sig_len]
            HR = HR[:,:,:sig_len]
        # print("Input Signal Length: ",sig_len, fragment.shape)
        ############################################################ Length Adjustment End

        embedding = ssl_layer(self.ssl_model, 
                                 self.ssl_processor, 
                                 input.squeeze(1),
                                 modelname=self.modelname
                                 ).detach()
    
        ################## Kmeans
        embedding_reshape = embedding.reshape(-1,1024)
        cluster_labels = self.kmeans.predict(embedding_reshape.detach().cpu().numpy())  # GPU 사용시, CPU로 이동
        quantized_embedding = self.kmeans.cluster_centers_[cluster_labels]
        
        embedding_new = quantized_embedding.reshape(embedding.shape[0], embedding.shape[1], -1)
        embedding = torch.from_numpy(embedding_new)

        dev = x.device
        embedding = embedding.to(dev)
        # B x L x F

        ################## Forward
        skip = [x]
        
        x = self.conv_in(x)
        skip.append(x)

        if self.visualize: 
            print(embedding.shape, "EMBEDDING: B x L x F")
            print(x.shape, "Conv Feature: B x F x L")

        film_list = [self.FiLM_e1, self.FiLM_e2, self.FiLM_e3, self.FiLM_e4]

        for i, encoder in enumerate(self.encoder):
            x = encoder(x)
            x = film_list[i](x, embedding)
            # print("\t x.shape", x.shape)
            skip.append(x)

        x = self.conv_bottle1(x) 
        x = self.conv_bottle2(x) 

        skip = skip[::-1]

        film_list_d = [self.FiLM_d1, self.FiLM_d2, self.FiLM_d3, self.FiLM_d4]
        for l in range(len(self.decoder)):
            x = x + skip[l]
            x = self.decoder[l](x)
            x = film_list_d[l](x, embedding)
            # print("\t x.shape", x.shape)

        x = x + skip[4]
        x = self.conv_out(x)
        x = x + skip[5]

        # Length Adjustment: Append the fragment back
        if len(fragment.size()) == 2:
            fragment = fragment.unsqueeze(-2)

        x = torch.cat((x, fragment), dim=-1)

        return x

def ssl_layer(model, processor, audio, modelname='hubert'):
    # audio의 끝에 80개의 zero를 pad
    audio = F.pad(audio, (0, 80), "constant", 0)
        
    # 장치를 확인하고 입력 데이터를 올바른 장치로 전송
    dev = audio.device
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True).input_values.to(dev)    
    inputs = inputs.squeeze(0) 
    
    with torch.no_grad():
        outputs = model(inputs, output_hidden_states=True)
        
    if modelname=='hubert':
        out_layer = outputs.hidden_states[22].to(dev)
    elif modelname=='w2v':
        out_layer = outputs.hidden_states[24].to(dev)
    elif modelname=='wavlm':
        out_layer = outputs.hidden_states[22].to(dev)
    
    return out_layer

class EncBlock(nn.Module):
    def __init__(self, out_channels, stride):
        super().__init__()
        

        self.res_units = nn.ModuleList([
                                    ResUnit(out_channels//2, 1),
                                    ResUnit(out_channels//2, 3),
                                    ResUnit(out_channels//2, 9)                                        
                                    ])
        
        self.conv = nn.Sequential(
                    nn.ELU(),
                    Pad((2 * stride - 1, 0)),
                    nn.Conv1d(in_channels = out_channels//2,
                                       out_channels = out_channels,
                                       kernel_size = 2 * stride,
                                       stride = stride, padding = 0),
                    )  
        
        
    def forward(self, x):
        
        for res_unit in self.res_units:
            x = res_unit(x)
        x = self.conv(x)

        return x
        
    
class DecBlock(nn.Module):
    def __init__(self, out_channels, stride):
        super().__init__()

        ## 채널 수 감소 & Upsampling
        self.conv = ConvTransposed1d(
                                 in_channels = out_channels*2, 
                                 out_channels = out_channels, 
                                 kernel_size = 2*stride, stride= stride,
                                 dilation = 1,
                                 )
        
        
        self.res_units = nn.ModuleList([
                                    ResUnit(out_channels, 1),
                                    ResUnit(out_channels, 3),
                                    ResUnit(out_channels, 9)                                       
                                    ])
               
        self.stride = stride
        

    def forward(self, x):
        x = self.conv(x)
        for res_unit in self.res_units:
            x = res_unit(x)
        return x
    
    
class ResUnit(nn.Module):
    def __init__(self, channels, dilation = 1):
        super().__init__()
        

        self.conv_in = Conv1d(
                                 in_channels = channels, 
                                 out_channels = channels, 
                                 kernel_size = 3, stride= 1,
                                 dilation = dilation,
                                 )
        
        self.conv_out = Conv1d(
                                in_channels = channels, 
                                 out_channels = channels, 
                                 kernel_size = 1, stride= 1,
                                 )
        
        self.conv_shortcuts = Conv1d(
                                in_channels = channels, 
                                 out_channels = channels, 
                                 kernel_size = 1, stride= 1,
                                 )
        
    
        
    def forward(self, x):
        y = self.conv_in(x)
        y = self.conv_out(y)
        x = self.conv_shortcuts(x)
        return x + y
        
    
class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation = 1, groups = 1):
        super().__init__()
        
        self.conv = nn.Conv1d(
            in_channels = in_channels, 
            out_channels = out_channels,
            kernel_size= kernel_size, 
            stride= stride, 
            dilation = dilation,
            groups = groups
        )
        self.conv = nn.utils.weight_norm(self.conv)
        
        self.pad = Pad(((kernel_size-1)*dilation, 0)) 
        self.activation = nn.ELU()
            

    def forward(self, x):

        x = self.pad(x)
        x = self.conv(x)
        x = self.activation(x)
        
        return x

class ConvTransposed1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 1, stride = 1, dilation = 1):
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride =stride,
            dilation = dilation
        )
        self.conv = nn.utils.weight_norm(self.conv)
        
        self.activation = nn.ELU()
        self.pad = dilation * (kernel_size - 1) - dilation * (stride - 1)
        
    def forward(self, x):
        x = self.conv(x)
        x = x[..., :-self.pad]
        x = self.activation(x)
        return x
    
class Pad(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad
    
    def forward(self, x):
        return F.pad(x, pad=self.pad)    
    
    
# x = torch.rand(3, 1, 32000)
# model = SEANet_v7(kmeans_model_path="/home/woongzip/SSLBWE_phase2/kmeans/kmeans_modelweight_64_wavlm.pkl")
# print(model(x,x).shape)
