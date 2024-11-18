import torch as th
import torch 
import torch.nn as nn
import torch.nn.functional as F
from SEANet import EncBlock,DecBlock,Conv1d,ConvTransposed1d,Pad
import pickle
# from transformers import HubertModel, AutoProcessor, Wav2Vec2Model, WavLMModel, AutoModel

from einops import rearrange
import warnings
from transformermodel import TransformerDecoder

"""
SEANet with Transformer Cross Attention

Parameters : 1M

d_model: 128
d_ff: 128 * 4
num_heads: 2
num_layers: 2
"""

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

class SEANet_v6(nn.Module):
    
    def __init__(self, min_dim=8, transformer_layers=2, kmeans_model_path=None, modelname="wavlm", visualize=False, **kwargs):
        from transformers import AutoProcessor, AutoModel
        super().__init__()
        
        self.visualize = visualize
        ## Load Kmeans model
        with open(kmeans_model_path, 'rb') as file:
            self.kmeans = pickle.load(file)
                
        self.min_dim = min_dim
        
        self.conv_in = Conv1d(
            in_channels = 1,
            out_channels = min_dim,
            kernel_size = 7,
            stride = 1
        )
        
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
        
        ##################### Linear Projection for HuBERT Embedding
        # self.ssl_projection = nn.Conv1d(
        #     in_channels = 1024,
        #     out_channels = min_dim*16, # 128
        #     kernel_size = 1
        # )
        
        self.downsampling_factor = 320  #2*4*5*8

        self.encoder = nn.ModuleList([
                                    EncBlock(min_dim*2, 2), # out_ch / stride
                                    EncBlock(min_dim*4, 4),
                                    EncBlock(min_dim*8, 5),
                                    EncBlock(min_dim*16, 8)                                        
                                    ])
        
        self.conv_bottle1 = Conv1d(
                            in_channels=min_dim*16, # 128 -> 32
                            out_channels = min_dim*16//4,
                            kernel_size = 7, 
                            stride = 1,
                            )
                        
        self.conv_bottle2 = Conv1d(
                            in_channels=min_dim*16//4, # 32 -> 32
                            out_channels = min_dim*16,
                            kernel_size = 7,
                            stride = 1,
                            )
        

        self.transformer = TransformerDecoder(num_layers=transformer_layers, d_input=1, 
                                                      d_model=128, num_heads=8, d_ff=256)

        # VectorwiseTransformer(num_layers=2, d_input=1, d_model=128, num_heads=8, d_ff=256, dropout=0.1).cuda()

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
                x = x[:,:,:sig_len]
                HR = HR[:,:,:sig_len]
                
        if x.dim()==2:
            sig_len = x.shape[1]
            if sig_len % self.downsampling_factor != 0:
                new_len = sig_len // self.downsampling_factor * self.downsampling_factor
                fragment = x[:,new_len:].clone().to(x.device)  # 수정된 부분
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
        
        # Kmeans
        cluster_labels = self.kmeans.predict(embedding_reshape.detach().cpu().numpy())  # GPU 사용시, CPU로 이동
        quantized_embedding = self.kmeans.cluster_centers_[cluster_labels]
        embedding_new = quantized_embedding.reshape(embedding.shape[0], embedding.shape[1], -1)
        embedding = torch.from_numpy(embedding_new)
        embedding = rearrange(embedding, 'b l f -> b f l').to(x.device)
        if self.visualize: print(embedding.shape, "EMBEDDING SHAPE: B x 1024 x L")

        # ssl_embedding = self.ssl_projection(embedding.to(x.device))
        # if self.visualize: print(ssl_embedding.shape, "SSL EMBEDDING SHAPE: B x F x L")
        ##############################
        
        # Forward
        skip = [x]
        x = self.conv_in(x)
        skip.append(x)

        for encoder in self.encoder:
            x = encoder(x)
            skip.append(x)

        # BottleNeck 1
        x = self.conv_bottle1(x)
        if self.visualize: print(x.shape, "X SHAPE: B x F x L")

        # Transformer
        x = self.transformer(x, embedding, embedding)
        
        # BottleNeck 2
        x = self.conv_bottle2(x)

        skip = skip[::-1]
        for l in range(len(self.decoder)):
            x = x + skip[l]
            x = self.decoder[l](x)

        x = x + skip[4]
        x = self.conv_out(x)
        
        x = x + skip[5]
        
        #################### Length Adjustment
        if len(fragment.size()) == 2:
            fragment = fragment.unsqueeze(-2)
            
        # print(x.shape, fragment.shape, "Two Shapes")
        x = torch.cat((x,fragment),dim=-1)
        # print("Output Signal Length: ",x.size()[-1])
        
        return x
    
    

    