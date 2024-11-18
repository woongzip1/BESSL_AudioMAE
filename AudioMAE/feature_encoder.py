import torch
import os
import torch.nn as nn
from einops import rearrange
from AudioMAE.models_mae import mae_vit_base_patch16

""" 85M Feature Extractor """
class AudioMAEEncoder(nn.Module):
    def __init__(self, img_size=(1024, 128), audio_exp=True, in_chans=1, visualize=False):
        super(AudioMAEEncoder, self).__init__()
        
        self.visualize = visualize
        # Load MAE_VIT base model
        self.model = mae_vit_base_patch16(img_size=img_size, audio_exp=audio_exp, in_chans=in_chans)
        
        # Delete decoder modules
        del self.model.decoder_embed
        del self.model.decoder_blocks
        del self.model.decoder_norm
        del self.model.decoder_pred
        del self.model.log_softmax
        del self.model.decoder_pos_embed
        
        self.model.forward = self.model.forward_encoder_no_mask

        # Load weights
        self.weight_path = "/home/woongjib/Projects/BESSL_AudioMAE/weights/pretrained_AS2M.pth"

    def load_weights(self):
        checkpoint_dict = torch.load(self.weight_path, map_location='cpu')
        model_dict = self.model.state_dict()
        
        # Filter out unnecessary keys and load only matching keys
        filtered_dict = {k: v for k, v in checkpoint_dict['model'].items() if k in model_dict}
        unfiltered = {k: v for k, v in checkpoint_dict['model'].items() if k not in model_dict}

        if self.visualize:
            # Print unused parameters
            print("\n ****** Unused Parameters ******")
            for k in unfiltered:
                print(k, end='\n')

            print("\n ****** Used Parameters ******")
            for k in filtered_dict:
                print(k, end='\n')

        # Update the model's state dict with the filtered checkpoint weights
        model_dict.update(filtered_dict)
        self.model.load_state_dict(model_dict)
        print(f"*** Weights loaded from {os.path.basename(self.weight_path)} ***")

    def forward(self, x, patch_len=64, freq_bin=8):
        x = self.model(x)
        # Delete CLS Token
        x = x[:, 1:, :]
        x = rearrange(x, 'b (t_p f_p) d -> b t_p f_p d', f_p=8)
        # Output Shape: B x T x F x D

        # len = 2 sec -> 200 // 16
        x = x[:,:patch_len, -freq_bin:,:] # B x T x 8 x D
        x = rearrange(x, 'b t f d -> b t (f d)')

        return x 

# Example
if __name__ == "__main__":
    data = torch.rand(1, 1, 1024, 128)
    model = AudioMAEEncoder(visualize=False)
    model.load_weights()

    out = model(data, patch_len=64)
    print(out.shape)
    
    from torchinfo import summary
    # print(summary(model, input_data=data))