from transformers import AutoModelForImageClassification

# Tải mô hình
model = AutoModelForImageClassification.from_pretrained("raffaelsiregar/utkface-race-classifications")

# In ra ánh xạ id2label
print(model.config.id2label)