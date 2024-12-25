
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Tải mô hình CodeGen
model_name = "Salesforce/codegen-6B-mono"  # Mô hình chuyên về mã nguồn
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Tạo pipeline cho AI
code_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Tạo đoạn mã mẫu
prompt = """
Viết một mod Minecraft Forge đơn giản:
- Tên mod: RubyMod
- Tính năng: Thêm một khối Ruby mới.
"""
response = code_generator(prompt, max_length=300, num_return_sequences=1)

print("Đoạn mã gợi ý:")
print(response[0]["generated_text"])
print(response[0][""])