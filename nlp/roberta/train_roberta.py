# 1. wget -c https://cdn-datasets.huggingface.co/EsperBERTo/data/oscar.eo.txt
# 2. python split_train_validation.py
# 2. mkdir EsperBERTo


# Roberta X shape when batch_orthonorm() is called: [64, 128, 768].
# Hence: after flattenning of B and T we get: 768 x 8192. The covariance will be 768 x 768.
# If: e ~ sqrt(c^2/2v), then in our case c=768, v=8192 => e=6
# If num_groups = 128 then each group size is 6: c=768, v=8192 => e=0.05
# If num_groups = 32 then each group size is 24: c=24, v=8192 => e=0.2

from pathlib import Path
import torch
from tokenizers import ByteLevelBPETokenizer
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from transformers import RobertaConfig
from transformers import RobertaTokenizerFast
from transformers import RobertaForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

from nlp.nano_gpt.model import BatchWhiteningBlock

# Initialize wandb
import wandb
wandb.init(project="batch-whitening", entity="nv-welcome")

paths = [str(x) for x in Path(".").glob("**/*.txt")]

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

tokenizer.save_model("EsperBERTo")

tokenizer = ByteLevelBPETokenizer(
    "./EsperBERTo/vocab.json",
    "./EsperBERTo/merges.txt",
)

tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)

print(tokenizer.encode("Mi estas Julien."))

print(tokenizer.encode("Mi estas Julien.").tokens)

config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

tokenizer = RobertaTokenizerFast.from_pretrained("./EsperBERTo", max_len=512)

model = RobertaForMaskedLM(config=config)
print(model.num_parameters())

# Replace LayerNorm with BatchWhiteningBlock recursively
def replace_layer_norm_with_batch_whitenin(module):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.LayerNorm):
            # Replace LayerNorm with BatchWhiteningBlock
            normalized_shape = child.normalized_shape[0]
            setattr(module, name, BatchWhiteningBlock(normalized_shape))
        else:
            # Recursively apply to child modules
            replace_layer_norm_with_batch_norm(child)

class CustomBatchNorm1d(torch.nn.BatchNorm1d):
    def forward(self, input):
        # Check if input is 3D (B, T, C)
        if input.dim() == 3:
            batch_size, seq_len, num_features = input.size()

            # Reshape input to (B * T, C)
            input = input.view(batch_size * seq_len, num_features)

            # Apply batch normalization
            output = super(CustomBatchNorm1d, self).forward(input)

            # Reshape back to (B, T, C)
            output = output.view(batch_size, seq_len, num_features)
            return output
        else:
            # If input is not 3D, apply regular BatchNorm1d
            return super(CustomBatchNorm1d, self).forward(input)

# Replace LayerNorm with BatchNore recursively
def replace_layer_norm_with_batch_norm(module):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.LayerNorm):
            # Replace LayerNorm with BatchNorm1d
            normalized_shape = child.normalized_shape[0]
            # Replace LayerNorm with BatchNorm1d, which expects input of shape (B * T, C)
            setattr(module, name, CustomBatchNorm1d(normalized_shape))
        else:
            # Recursively apply to child modules
            replace_layer_norm_with_batch_norm(child)

# Replace LayerNorm layers with BatchWhiteningBlock or CustomBatchNorm1d or leave it as is
type = 'LayerNorm'  # BatchWhiteningBlock / CustomBatchNorm1d / LayerNorm

if type == 'LayerNorm':
    pass
elif type == 'BatchWhiteningBlock':
    replace_layer_norm_with_batch_whitenin(model)
    # Solve runtime error: lazy wrapper should be called at most once
    res = torch.linalg.cholesky_ex(torch.ones((1, 1), device="cuda:0"))
elif type == 'CustomBatchNorm1d':
    replace_layer_norm_with_batch_norm(model)
else:
    print(f'Unrecognized layer type {type}')
    exit(1)

# Load training dataset
train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./oscar.eo_train.txt",
    block_size=128,
)

# Load validation dataset
valid_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./oscar.eo_valid.txt",
    block_size=128,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir="./EsperBERTo",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_gpu_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
    evaluation_strategy="steps",  # Evaluate periodically
    eval_steps=2_000,  # Run validation every 5000 steps
    prediction_loss_only=True,
    # report_to='none'
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)

trainer.train()
