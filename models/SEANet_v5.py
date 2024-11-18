from einops import rearrange
import torch as th
import torch 
import torch.nn as nn
import torch.nn.functional as F
from SEANet import ConvTransposed1d,Pad
import pickle
# from transformers import HubertModel, AutoProcessor, Wav2Vec2Model, WavLMModel, AutoModel

import warnings

"""
ResBlock & Enc & Dec are modified to have input dim
Fixed Embedding Dimensions are concatenated to E1, E2, E3, E4 and B1, B2

Paramters over Embed Dim 
32: 700K
16: 636K
8: 603K 
"""
def ssl_layer(model, processor, audio, modelname='hubert'):
    # audio의 끝에 80개의 zero를 pad
    audio = F.pad(audio, (0, 80), "constant", 0)
        
    # 장치를 확인하고 입력 데이터를 올바른 장치로 전송
    dev = audio.device
    # print("Input device:", dev)
    # print("Model device:", next(model.parameters()).device)  # 모델 파라미터의 장치 확인

    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True).input_values.to(dev)
    
    # print("*************************************")
    # print("Input HuBERT Shape Before Squeeze : ", inputs.shape)
    # print("*************************************")
    ## input shape: 1 x B x L ---> B x L
    
    inputs = inputs.squeeze(0) 
    
    with torch.no_grad():
        outputs = model(inputs, output_hidden_states=True)
        
    if modelname=='hubert':
        out_layer = outputs.hidden_states[22].to(dev)
    elif modelname=='w2v':
        out_layer = outputs.hidden_states[24].to(dev)
    elif modelname=='wavlm':
        out_layer = outputs.hidden_states[22].to(dev)
    
    # print(out_layer.shape,"haha")
    return out_layer


""" Repeat Tensors, Expand the Length of Tensor 
[B, F, L] --> [B, F, NL]
n=3 example: [A B C] --> [AAA BBB CCC]
"""
def repeat_features(tensor, repeat=4, visualize=False):    

    # 세 번째 축(axis=2)을 따라 각 벡터를 4번 반복
    b,f,l = tensor.shape
    repeated_tensor = tensor.unsqueeze(3).repeat(1, 1, 1, repeat).reshape(b, f, l*repeat)

    if visualize:
        print(tensor[:2])
        print("Original shape:", tensor.shape)       # (5, 4, 3)
        print("Repeated shape:", repeated_tensor.shape)  # (5, 4, 12)
        print(repeated_tensor[:2])

    return repeated_tensor

tensor = torch.randint(0, 10, (3, 6, 3), dtype=torch.int)
t = repeat_features(tensor)


class FCLayer(nn.Module):
    def __init__(self, input_dim=1024, output_dim=4):
        super().__init__()

        self.linear = nn.Conv1d(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=1
        )

    # B L F: B 100 1024 input
    def forward(self, x):
        # print("input shape:",x.shape)
        x = rearrange(x,"b l f -> b f l")
        # print("rearrange shape:",x.shape)
        x = self.linear(x)
        # print("output.shape:",x.shape,"\n")

        return x
    
from SEANet import Conv1d,ConvTransposed1d,Pad

class DecBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        ## 채널 수 감소 & Upsampling
        self.conv = ConvTransposed1d(
                                 in_channels = in_channels,
                                 out_channels = out_channels, 
                                 kernel_size = 2*stride, stride= stride,
                                 dilation = 1,
                                )
        
        
        self.res_units = nn.ModuleList([
                                    ResUnit(out_channels, out_channels, 1),
                                    ResUnit(out_channels, out_channels, 3),
                                    ResUnit(out_channels, out_channels, 9)                                       
                                    ])
               
        self.stride = stride
        

    def forward(self, x):
        x = self.conv(x)
        for res_unit in self.res_units:
            x = res_unit(x)
        return x
    
    
class ResUnit(nn.Module):
    def __init__(self, in_channels, channels, dilation = 1):
        super().__init__()
        

        self.conv_in = Conv1d(
                                 in_channels = in_channels, 
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
                                in_channels = in_channels, 
                                 out_channels = channels, 
                                 kernel_size = 1, stride= 1,
                                 )
        

    def forward(self, x):
        y = self.conv_in(x)
        y = self.conv_out(y)
        x = self.conv_shortcuts(x)
        return x + y
    
class EncBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        

        self.res_units = nn.ModuleList([
                                    ResUnit(in_channels,out_channels//2, 1),
                                    ResUnit(out_channels//2,out_channels//2, 3),
                                    ResUnit(out_channels//2,out_channels//2, 9)                                        
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

class SEANet_v5(nn.Module):
    
    # def __init__(self, min_dim=8,kmeans_model_path='/home/woongzip/RealTimeBWE/Kmeans/kmeans_modelweight_200.pkl', **kwargs):
    def __init__(self, min_dim=8, feat_dim=32, kmeans_model_path=None, modelname="hubert",**kwargs):
        from transformers import AutoProcessor, AutoModel
        super().__init__()
        
        ## Load Kmeans model
        # with open(kmeans_model_path, 'rb') as file:
        #     self.kmeans = pickle.load(file)
                
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
            
        for param in self.ssl_model.parameters():
            param.requires_grad = False
        
        ## Linear Projection for HuBERT Embedding
        ## Unuse
        # self.ssl_projection = Conv1d(
        #     in_channels = 1024,
        #     out_channels = min_dim*16//4,
        #     kernel_size = 1
        # )

        self.featdim = feat_dim
        self.FC_embed = FCLayer(output_dim=self.featdim)

        ## First Conv Layer
        self.conv_in = Conv1d(
            in_channels = 1,
            out_channels = min_dim, #4
            kernel_size = 7,
            stride = 1
        )
        self.downsampling_factor = 320  #2*4*5*8
        
        
        self.encoder = nn.ModuleList([
                                    EncBlock(min_dim, min_dim*2, 2), # 8 -> 16
                                    EncBlock(min_dim*2+self.featdim, min_dim*4, 4), # 16+16 -> 32
                                    EncBlock(min_dim*4+self.featdim, min_dim*8, 5), # 32+16 -> 64
                                    EncBlock(min_dim*8+self.featdim, min_dim*16, 8) # 64+16 -> 128                                       
                                    ])
        
                            # 128 + 16 -> 32
        self.conv_bottle1 = Conv1d(
                                in_channels=min_dim*16+self.featdim,
                                out_channels = min_dim*4,
                                kernel_size = 7, 
                                stride = 1,
                                )
        
                            # 32 + 16 -> 64
        self.conv_bottle2 = Conv1d(
                                in_channels=(min_dim*4+self.featdim),
                                out_channels = min_dim*16,
                                kernel_size = 7,
                                stride = 1,
                                )        
        
        self.decoder = nn.ModuleList([
                                    DecBlock(min_dim*16+self.featdim, min_dim*8, 8), # 128+16 -> 64
                                    DecBlock(min_dim*8+self.featdim, min_dim*4, 5), # 64+16 -> 32
                                    DecBlock(min_dim*4+self.featdim, min_dim*2, 4), # 32+16 -> 16
                                    DecBlock(min_dim*2+self.featdim, min_dim, 2),   # 16+16 -> 8
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
                fragment = x[:,:,new_len:].clone().to(x.device)  # 
                # fragment = x[:,:,sig_len:]
                x = x[:,:,:sig_len]
                HR = HR[:,:,:sig_len]
                
        if x.dim()==2:
            sig_len = x.shape[1]
            if sig_len % self.downsampling_factor != 0:
                new_len = sig_len // self.downsampling_factor * self.downsampling_factor
                fragment = x[:,new_len:].clone().to(x.device)  # 
                # fragment = x[:,sig_len:]
                x = x[:,:sig_len]
                HR = HR[:,:sig_len]
                
        while len(x.size()) < 3:
            x = x.unsqueeze(-2)
            HR = HR.unsqueeze(-2)
            
        
        if sig_len % self.downsampling_factor != 0:
            sig_len = sig_len // self.downsampling_factor * self.downsampling_factor
            x = x[:,:,:sig_len]
            HR = HR[:,:,:sig_len]
        # print("Input Signal Length: ",sig_len, fragment.shape)
        #################### Length Adjustment End

        ######################## Extract HuBERT Embeddings: B x L x Dim
    
        embedding = ssl_layer(self.ssl_model, 
                                 self.ssl_processor, 
                                 input.squeeze(1),
                                 modelname=self.modelname
                                 ).detach()
        ## Embedding into 2dim
        embedding_reshape = embedding.reshape(-1, 1024)
        
        # # Kmeans
        # cluster_labels = self.kmeans.predict(embedding_reshape.detach().cpu().numpy())  # GPU 사용시, CPU로 이동
        # quantized_embedding = self.kmeans.cluster_centers_[cluster_labels]
        
        # embedding_new = quantized_embedding.reshape(embedding.shape[0], embedding.shape[1], -1)
        # embedding_new = torch.from_numpy(embedding_new)
        
        ## KMeans forward를 하기 위해 numpy array로 변환했다가 다시 Tensor로 변환
        
        ######################## Embedding Feature Adjustment
        # embedding_new = rearrange(embedding_new, 'b l f -> b f l')
        # dev = next(self.ssl_projection.parameters()).device
        dev = x.device
        # embedding_t = self.FC_embed(embedding_new.to(dev))
        embedding_t = self.FC_embed(embedding.to(dev))

        embedding_16 = repeat_features(embedding_t, self.downsampling_factor//2)
        embedding_32_e = repeat_features(embedding_t, self.downsampling_factor//(2*4))
        embedding_32_b = repeat_features(embedding_t, self.downsampling_factor//(2*4*5*8))
        embedding_64 = repeat_features(embedding_t, self.downsampling_factor//(2*4*5))
        embedding_128 = repeat_features(embedding_t, self.downsampling_factor//(2*4*5*8))        

        # print("embedding16 shape: ", embedding_16.shape)
        # print("embedding32 shape: ", embedding_32_e.shape)
        # print("embedding64 shape: ", embedding_64.shape)
        # print("embedding128 shape: ", embedding_128.shape)
        # print("embedding32B shape: ", embedding_32_b.shape)
        ##############################

        skip = [x]
        # [1 32000] -> [4 32000]
        x = self.conv_in(x)
        # print("C1", x.shape)
        skip.append(x)

        embedding_list = [embedding_16, embedding_32_e, embedding_64, embedding_128]
        ## 8 16 32 64
        for i,encoder in enumerate(self.encoder):
            x = encoder(x)
            # print("ENC", x.shape)
            x = torch.cat((x.to(dev), embedding_list[i]), dim=1)
            # print("After CAT:",i, x.shape)
            skip.append(x)

        x = self.conv_bottle1(x)
        # print("Bottle Neck1:",x.shape) 
        x = torch.cat((x.to(dev), embedding_32_b), dim=1)  
        # print("Concat:", x.shape)

        x = self.conv_bottle2(x)
        x = torch.cat((x.to(dev), embedding_128), dim=1)
        # print("Bottle Neck2:", x.shape)
        
        skip = skip[::-1]

        # return 0
        embedding_list_dec = [embedding_64, embedding_32_e, embedding_16]

        for l in range(len(self.decoder)):
            # print("!!!",l, x.shape)
            x = x + skip[l]
            x = self.decoder[l](x)
            # print("DEC shape:", x.shape)
            if l != len(self.decoder)-1:
                x = torch.cat((x.to(dev), embedding_list_dec[l]), dim=1)
            # print("After CAT:",l, x.shape)

        x = x + skip[4]
        x = self.conv_out(x)
        
        x = x + skip[5]
        
        #################### Length Adjustment
        if len(fragment.size()) == 2:
            fragment = fragment.unsqueeze(-2)
            
        x = torch.cat((x,fragment),dim=-1)
        
        
        return x
