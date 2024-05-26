import os
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

# Пути к модели и датасету
model_checkpoint = "lmsys/vicuna-7b-v1.5"  # Изменено на модель из Hugging Face
dataset_path = "./Dataset/Devices/refrigerators.json"

# Загрузка модели и токенайзера
print("Загрузка модели и токенайзера...")   
model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
print("Модель и токенайзер загружены.")

# Настройки тренировки
GRADIENT_CHECKPOINTING = True
PER_DEVICE_TRAIN_BATCH_SIZE = 1
WARMUP_STEPS = 0
EPOCHС = 3
LEARNING_RATE = 1e-5

# Загрузка и обработка датасета
print("Загрузка датасета...")
dataset = load_dataset('json', data_files=dataset_path)
print("Датасет загружен. Примеры данных:")
print(dataset['train'][0])

# Шаблон для форматирования данных
alpaca_prompt = """system_prompt описывает, кем ты являешься и как нужно себя вести. Далее примеры запросов (instruction) и ответов (output) на них.

### system_prompt:
{}

### instruction:
{}

### output:
{}"""

EOS_TOKEN = tokenizer.eos_token

# Функция для форматирования данных
def formatting_prompts_func(examples):
    instructions = examples["system_prompt"]
    inputs = examples["instruction"]
    outputs = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

# Форматирование датасета
print("Форматирование датасета...")
formatted_dataset = dataset.map(formatting_prompts_func, batched=True)
print("Форматирование завершено. Пример форматированных данных:")
print(formatted_dataset['train'][0])

# Токенизация
print("Токенизация датасета...")
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_dataset = formatted_dataset.map(tokenize_function, batched=True)
print("Токенизация завершена. Пример токенизированных данных:")
print(tokenized_dataset['train'][0])

# Создание директории для сохранения результатов
output_dir = os.path.join('.', datetime.now().strftime("%Y%m%d%H%M%S"))
os.makedirs(output_dir, exist_ok=True)

# Настройки тренировки
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    num_train_epochs=EPOCHС,
    warmup_steps=WARMUP_STEPS,
    learning_rate=LEARNING_RATE,
    logging_steps=10,
    save_total_limit=1,
    save_steps=500,
    gradient_checkpointing=GRADIENT_CHECKPOINTING,
    fp16=True,  # Включение автоматической смешанной точности для ускорения
    no_cuda=False  # Включить использование GPU, если доступно
)

# Коллектор данных
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# Создание объекта Trainer
print("Создание объекта Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    data_collator=data_collator
)

# Запуск тренировки
print("Запуск тренировки...")
trainer.train()
print("Тренировка завершена.")

# Сохранение модели и токенайзера после тренировки
print("Сохранение модели и токенайзера...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Модель и токенайзер сохранены в {output_dir}")
