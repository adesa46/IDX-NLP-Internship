import spacy
from spacy.training.example import Example
import json
import random
import os
import warnings

warnings.filterwarnings("ignore")

def train_ner_model(data_path, output_dir, iterations=15):
    with open(data_path, 'r', encoding='utf-8') as f:
        datasets = json.load(f)

    train_data = []
    for d in datasets:
        text = d['text']
        entities = d['entities']
        # Remove entities with zero length
        valid_entities = [e for e in entities if e[1] > e[0]]
        train_data.append((text, {"entities": valid_entities}))

    random.seed(42)
    random.shuffle(train_data)
    split = int(len(train_data) * 0.8)
    train_split = train_data[:split]
    test_split = train_data[split:]

    nlp = spacy.blank("en")
    
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    optimizer = nlp.begin_training()

    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    
    print(f"Training model with {len(train_split)} examples for {iterations} iterations...")

    with nlp.disable_pipes(*unaffected_pipes):
        for itn in range(iterations):
            random.shuffle(train_split)
            losses = {}
            for batch in spacy.util.minibatch(train_split, size=8):
                examples = []
                for text, annots in batch:
                    doc = nlp.make_doc(text)
                    try:
                        example = Example.from_dict(doc, annots)
                        examples.append(example)
                    except ValueError as e:
                        # Skip if there's an alignment issue
                        pass
                if examples:
                    nlp.update(examples, drop=0.35, sgd=optimizer, losses=losses)
            if (itn+1) % 5 == 0:
                print(f"Iteration {itn+1} Losses:", losses)

    os.makedirs(output_dir, exist_ok=True)
    nlp.to_disk(output_dir)
    print(f"Model saved to {output_dir}")

    with open('data/test_dataset.json', 'w', encoding='utf-8') as f:
        json.dump([{"text": t, "entities": e["entities"]} for t, e in test_split], f, indent=2)

if __name__ == "__main__":
    train_ner_model("data/labeled_dataset.json", "data/models/custom_ner")
