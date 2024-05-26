import os
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

# Пути к модели и датасету
model_checkpoint = "./Models/Vicuna-7b"
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



# from unsloth import FastLanguageModel
# import torch
# max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
# dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
# load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name = "./Models/Vicuna-7b/",
#     max_seq_length = max_seq_length,
#     dtype = dtype,
#     load_in_4bit = load_in_4bit,
#     # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
# )

# model = FastLanguageModel.get_peft_model(
#     model,
#     r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
#     target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
#                       "gate_proj", "up_proj", "down_proj",],
#     lora_alpha = 16,
#     lora_dropout = 0, # Supports any, but = 0 is optimized
#     bias = "none",    # Supports any, but = "none" is optimized
#     # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
#     use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
#     random_state = 3407,
#     use_rslora = False,  # We support rank stabilized LoRA
#     loftq_config = None, # And LoftQ
# )

# alpaca_prompt = """system_prompt описывает, кем ты являешься и как нужно себя вести. Далее примемы запросов (insturction) и ответов (output) на них.

# ### system_prompt:
# {}

# ### instruction:
# {}

# ### output:
# {}"""

# EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
# def formatting_prompts_func(examples):
#     instructions = examples["system_prompt"]
#     inputs       = examples["instruction"]
#     outputs      = examples["output"]
#     texts = []
#     for instruction, input, output in zip(instructions, inputs, outputs):
#         # Must add EOS_TOKEN, otherwise your generation will go on forever!
#         text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
#         texts.append(text)
#     return { "text" : texts, }
# pass

# from datasets import load_dataset
# dataset = load_dataset("./Dataset/Devices/refrigerators.json", split = "train")
# dataset = dataset.map(formatting_prompts_func, batched = True,)

# from trl import SFTTrainer
# from transformers import TrainingArguments

# trainer = SFTTrainer(
#     model = model,
#     tokenizer = tokenizer,
#     train_dataset = dataset,
#     dataset_text_field = "text",
#     max_seq_length = max_seq_length,
#     dataset_num_proc = 2,
#     packing = False, # Can make training 5x faster for short sequences.
#     args = TrainingArguments(
#         per_device_train_batch_size = 2,
#         gradient_accumulation_steps = 4,
#         warmup_steps = 5,
#         max_steps = 60,
#         learning_rate = 2e-4,
#         fp16 = not torch.cuda.is_bf16_supported(),
#         bf16 = torch.cuda.is_bf16_supported(),
#         logging_steps = 1,
#         optim = "adamw_8bit",
#         weight_decay = 0.01,
#         lr_scheduler_type = "linear",
#         seed = 3407,
#         output_dir = "./Versions/",
#     ),
# )

# trainer.train()

# model.save_pretrained("./Versions/") # Local saving



# from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling

# # Загрузка модели и токенизатора
# model_name = "./Models/Vicuna-7b/"
# model = AutoModelForCausalLM.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# train_file_path = "./Dataset/Devices/train_data.txt"

# # Создание датасета для обучения
# def load_dataset(train_file_path):
#     return TextDataset(
#         tokenizer=tokenizer,
#         file_path=train_file_path,
#         block_size=128
#     )

# train_dataset = load_dataset(train_file_path)
# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# # Настройка параметров обучения
# training_args = TrainingArguments(
#     output_dir="./Versions",
#     overwrite_output_dir=True,
#     num_train_epochs=3,
#     per_device_train_batch_size=4,
#     save_steps=5000,
#     save_total_limit=2,
# )

# # Создание тренера
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     data_collator=data_collator,
#     train_dataset=train_dataset,
# )

# # Обучение модели
# trainer.train()
# trainer.save_model("./Versions")
# tokenizer.save_pretrained("./Versions")
