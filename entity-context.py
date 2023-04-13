from transformers import TFAutoTokenizer, TFAutoModelForTokenClassification, pipeline

tokenizer = TFAutoTokenizer.from_pretrained("dslim/bert-base-NER")

model = TFAutoModelForTokenClassification.from_pretrained(
    "dslim/bert-base-NER")

ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

sample_text = "Apple Inc. is an American multinational technology company headquartered in Cupertino, California."
entities = ner_pipeline(sample_text)
print(entities)
