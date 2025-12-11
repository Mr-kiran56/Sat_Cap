from imagepreprocess import Preprocess
import torch
import re
import random
class Caption:
    def __init__(self, image_path, model, processor, tokenizer, device, max_length=30):
        self.image_path = image_path
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

    def generate_quality_caption(self):
        self.model.eval()

        # ✅ Step 1: Preprocess image properly
        img = Preprocess(self.image_path).crop_salient_region()

        # ✅ Step 2: Convert image to tensor using CLIP processor
        image_inputs = self.processor(images=img, return_tensors="pt").to(self.device)

        # ✅ Step 3: Extract image embeddings using CLIP encoder
        with torch.no_grad():
            img_embeds = self.model.proj(
                self.model.encoder(image_inputs["pixel_values"]).pooler_output
            ).unsqueeze(1)

        # ✅ Step 4: Start caption generation
        generated = torch.tensor([[self.tokenizer.bos_token_id]], device=self.device)

        for _ in range(self.max_length):
            with torch.no_grad():
                text_embeds = self.model.decoder.transformer.wte(generated)
                inputs_embeds = torch.cat([img_embeds, text_embeds], dim=1)
                attn = torch.ones(inputs_embeds.shape[:2], device=self.device)
                logits = self.model.decoder(
                    inputs_embeds=inputs_embeds, attention_mask=attn
                ).logits

                next_token_logits = logits[:, -1, :] / 0.8  # temperature
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)

                generated = torch.cat([generated, next_token], dim=1)

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        caption = self.tokenizer.decode(
            generated[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        caption = re.sub(r'</w>', '', caption)
        caption = re.sub(r'[^a-zA-Z0-9.,!?\'" ]+', '', caption)
        caption = re.sub(r'\s+', ' ', caption).strip()
        sentences = [s.strip() for s in caption.split('.') if s.strip()]
        random_sentence = random.choice(sentences)

        return random_sentence
