import torch
import torch.nn as nn
from transformers import (
    CLIPProcessor, CLIPModel,
    GPT2Tokenizer, GPT2LMHeadModel
)

class StableSalientCaptioner(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.encoder = clip_model.vision_model
        self.decoder = GPT2LMHeadModel.from_pretrained("gpt2")
        self.proj = nn.Sequential(
            nn.Linear(768, self.decoder.config.n_embd),
            nn.LayerNorm(self.decoder.config.n_embd),
            nn.Dropout(0.1)
        )
        self.tokenizer = tokenizer
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, pixel_values, input_ids, attention_mask):
        vision_outputs = self.encoder(pixel_values)
        img_embeds = self.proj(vision_outputs.pooler_output).unsqueeze(1)
        text_embeds = self.decoder.transformer.wte(input_ids)
        inputs_embeds = torch.cat([img_embeds, text_embeds], dim=1)
        extended_attention_mask = torch.cat([
            torch.ones((attention_mask.size(0), 1), device=attention_mask.device),
            attention_mask
        ], dim=1)
        labels = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)
        labels = torch.cat([torch.full((labels.size(0), 1), -100, device=labels.device), labels], dim=1)
        return self.decoder(inputs_embeds=inputs_embeds,
                            attention_mask=extended_attention_mask,
                            labels=labels)
