"""Train the reduced FiLM conditional flow on fixed SST-5 activations."""

import json
import math
import pathlib

import torch
from transformers import AutoModel, AutoTokenizer

from nonlinear import ConditionalFlow, conditional_flow_loss
from training import schedule

ENCODER = "distilbert-base-uncased-finetuned-sst-2-english"

@torch.no_grad()
def conditions():
    tokenizer = AutoTokenizer.from_pretrained(ENCODER, local_files_only=True)
    encoder = AutoModel.from_pretrained(ENCODER, local_files_only=True).eval()
    texts = ["The text is positive.", "The text is negative.", "The text has no condition."]
    tokens = tokenizer(texts, padding=True, return_tensors="pt")
    states = encoder(**tokens).last_hidden_state
    mask = tokens["attention_mask"][:, :, None]
    return (states * mask).sum(1) / mask.sum(1)

def main():
    torch.manual_seed(0)
    source = torch.load("datasets/sst5_layer6_e51bdcd.pt", weights_only=False)
    positive, negative = source["positive"].float(), source["negative"].float()
    values = torch.cat((positive[:2048], negative[:2048]))
    labels = torch.cat((torch.zeros(2048, dtype=torch.long), torch.ones(2048, dtype=torch.long)))
    embedding = conditions()
    model = ConditionalFlow()
    model.set_stats(values)
    standardized = model.standardize(values)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: schedule(step, 300, 3))
    output = pathlib.Path("runs/mac_reduced_unisteer")
    output.mkdir(parents=True, exist_ok=True)
    config = {"train_per_class": 1792, "val_per_class": 256, "steps": 300,
              "batch": 64, "lr": 4e-5, "condition_encoder": ENCODER,
              "conditioning": "pooled frozen embedding with FiLM", "dropout": 0.1,
              "model": model.config, "source": source["meta"]}
    (output / "config.json").write_text(json.dumps(config, indent=2))
    train_index = torch.cat((torch.arange(1792), torch.arange(2048, 3840)))
    val_index = torch.cat((torch.arange(1792, 2048), torch.arange(3840, 4096)))
    history, best = [], math.inf
    for step in range(300):
        index = train_index[torch.randint(len(train_index), (64,))]
        condition = embedding[labels[index]].clone()
        condition[torch.rand(64) < 0.1] = embedding[2]
        value = conditional_flow_loss(model, standardized[index], condition)
        value.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        if step % 30 == 0 or step == 299:
            fixed = torch.Generator().manual_seed(10_000)
            with torch.no_grad():
                val = conditional_flow_loss(model, standardized[val_index],
                                            embedding[labels[val_index]], fixed)
            row = {"step": step, "train_loss": float(value), "val_loss": float(val),
                   "grad_norm": float(grad), "lr": optimizer.param_groups[0]["lr"]}
            history.append(row)
            print(json.dumps(row), flush=True)
            if val < best:
                best = float(val)
                save(output / "best.pt", model, embedding, config, step, best)
    (output / "history.jsonl").write_text("".join(json.dumps(row) + "\n" for row in history))
    save(output / "final.pt", model, embedding, config, 299, best)

def save(path, model, embedding, config, step, best):
    torch.save({"model": model.state_dict(), "model_config": model.config,
                "conditions": embedding, "config": config, "step": step,
                "best_val_loss": best}, path)

if __name__ == "__main__":
    main()
