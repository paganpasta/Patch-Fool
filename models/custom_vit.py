import torch
from transformers import ViTForImageClassification

class CustomViT(torch.nn.Module):
    def __init__(self, model_name='WinKawaks/vit-small-patch16-224', num_labels=1000, device='cpu'):
        super().__init__()
        self.model = ViTForImageClassification.from_pretrained(model_name, num_labels=num_labels).to(device)
        self.backbone = self.model.vit
        self.classifier = self.model.classifier
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device)
        self.use_dataset_preprocess = False
    def normalize(self, x):
        return (x - self.mean.view(1, 3, 1, 1)) / self.std.view(1, 3, 1, 1)

    def forward(self, pixel_values, attn_out=None):
        if attn_out is None:
            attn_out = []

        # normalized_pixel_values = self.normalize(pixel_values)
        outputs = self.model(pixel_values, output_attentions=True)
        logits = outputs.logits

        attn_out.extend(outputs.attentions)

        return logits, attn_out
