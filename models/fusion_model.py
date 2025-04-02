import torch
import torch.nn as nn
from models.modules.imaging_module import ImagingModule
from models.modules.nlp_module import NLPModule
from models.modules.omics_module import OmicsModule

class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism to align features between modalities.
    """
    def __init__(self, embed_dim, num_heads):
        super(CrossModalAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query, key, value):
        """
        Applies attention and residual connection with layer normalization.
        Args:
            query: Query tensor (e.g., text embeddings).
            key: Key tensor (e.g., image or omics embeddings).
            value: Value tensor (e.g., image or omics embeddings).
        """
        attn_output, _ = self.multihead_attn(query, key, value)
        output = self.norm(attn_output + query)  # Residual connection
        return output


class TransformerFusionModel(nn.Module):
    """
    Transformer-based Fusion Model with cross-modal attention for multimodal data.
    """
    def __init__(self, omics_input_dim, embed_dim=512, num_heads=8, num_layers=3):
        super(TransformerFusionModel, self).__init__()

        self.imaging_module = ImagingModule()
        self.nlp_module = NLPModule()  
        self.omics_module = OmicsModule(omics_input_dim)  

        self.img_projection = nn.Linear(64 * 56 * 56, embed_dim)  
        self.text_projection = nn.Linear(768, embed_dim)  
        self.omics_projection = nn.Linear(128, embed_dim)

        self.cross_modal_attn1 = CrossModalAttention(embed_dim, num_heads)  
        self.cross_modal_attn2 = CrossModalAttention(embed_dim, num_heads)  
        self.cross_modal_attn3 = CrossModalAttention(embed_dim, num_heads)  

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_layer = nn.Linear(embed_dim, 1)  

    def forward(self, image, text, omics):
        """
        Forward pass of the fusion model.
        Args:
            image: Image tensor of shape (batch_size, 3, 224, 224)
            text: Tokenized text input as dictionary (from tokenizer)
            omics: Omics data tensor of shape (batch_size, omics_input_dim)
        """

        img_features = self.imaging_module(image) 
        img_features = img_features.view(img_features.size(0), -1)  
        img_features = self.img_projection(img_features) 

        text_features = self.nlp_module(text)  
        text_features = self.text_projection(text_features) 

        omics_features = self.omics_module(omics)  
        omics_features = self.omics_projection(omics_features)  


        img_text_aligned = self.cross_modal_attn1(text_features, img_features, img_features)
        omics_text_aligned = self.cross_modal_attn2(text_features, omics_features, omics_features)
        img_omics_aligned = self.cross_modal_attn3(omics_features, img_features, img_features)

        fused_features = torch.stack([img_text_aligned, omics_text_aligned, img_omics_aligned], dim=1)
        fused_features = fused_features.mean(dim=1)  
        encoded_features = self.transformer_encoder(fused_features.unsqueeze(1)).squeeze(1)

        output = self.output_layer(encoded_features)
        return output


