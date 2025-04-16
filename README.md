### Introduction
The proliferation of large language models (LLMs) has catalyzed the growth of various industries, including smart healthcare and education. It is therefore imperative to ensure the controlled and beneficial application of LLMs across specific domains for downstream tasks through transfer learning, while preserving their general capabilities. 

We propose a novel, on-device efficient fine-tuning optimization algorithm for LLMs, utilizing federated transfer learning in response to these challenges. 

Specifically, we introduce the Fusion of Low Rank (FoRA) optimization algorithm from a micro perspective, which enhances multi-dimensional feature aggregation through the addition of efficient parameters. 

From a meso perspective, we extend the application of the FoRA algorithm across all linear layers within the transformer architecture to facilitate downstream task performance. 

Finally, from a macro perspective and with a focus on the medical domain, we incorporate quantization techniques into the federated learning framework to achieve on-device efficient fine-tuning optimization, thereby offering dual protection for data and model integrity.


At present, we will upload the code related to PEFT first, and then upload part of the FATE code after the review of our paper is completed.


### Environmental preparation
#### 1. download our code
```sh

```

#### 2. download baichuan-13B and 7B LLMs
```sh
# mkdir baichuan-inc in root path to save model 
pip install modelscope
from modelscope.hub.snapshot_download import snapshot_download
model_dir = snapshot_download('baichuan-inc/Baichuan2-13B-Chat', cache_dir='/dt/ft',revision='v1.0.1')
model_dir = snapshot_download('baichuan-inc/Baichuan2-7B-Chat', cache_dir='/dt/ft',revision='v1.0.1')


#### 3. mk conda environment，and install  python requirements environment
```sh
conda create -n FATELLM python=3.10  
conda activate FATELLM
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

```

#### 4. Hardware requirements
- Meso perspective memory：>= 35G
- Macro perspective memory：>= 15G

### Fine Tuning
#### 1. dataset [All datasets in Reference link](https://drive.google.com/drive/folders/1WE2em4vTFYRl64w_OiK_7ZgsZAh3s59k?usp=drive_link)
##### 1.1  NLG-mechanical_manufact
`data` path for save dataset，You can prepare data by yourself according to business needs. The data format is as follows：
- **instruction**：Task instruction, cannot be empty.
- **input**：Task input, can be empty. If it is not empty, when the project internally processes the training data, instruction and input will be spliced together as the input of the task.
- **output**：Task output,cannot be empty.
```json
[
  {
    "instruction": "你现在是一个信息抽取模型，请你帮我抽取出关系内容为\"性能故障\", \"部件故障\", \"组成\"和 \"检测工具\"的相关三元组，三元组内部用\"_\"连接，三元组之间用\\n分割。文本：", 
    "input": "诊断过程有电脑检查是“驾驶员侧安全气囊引爆装置电阻太小”，偶发故障。电脑清完故障车辆行驶100米左右故障再次出现，凭维修经验判断造成安全气囊电阻变化的分两方面，一；电阻太小应该是线路有短路的地方。二；电阻太大应该是线路有开路的地方。现在这个故障因该是线路有短路，拆掉气囊电脑检测故障没有变化。说明安全气囊没有问题，继续拆掉气囊滑环，电脑检查出现“驾驶员侧安全气囊引爆装置电阻太大”问题应该是气囊滑环出现了毛病，因为店内没有新的气囊滑环只能等配件。可是又仔细一想新车不会是气囊滑环有为题是不是接入不好，用万用表测量滑环上的电阻都没有问题，接着用替代法试验。找一个和气囊电阻相似的电阻连接在滑环和转向接线柱上。车辆上和气囊电阻相近的就是大灯上的小插泡，将插泡好后用电脑检查没有故障，仔细观察气囊滑环上两短接片有些短，问题就在这。当两接片接被针脚顶开的时候正常，顶不开时出现故障。因为规定不可对安全气囊上的元件擅自维修和修复，只能更换新的滑环。", 
    "output": "气囊滑环_部件故障_出现了毛病\n气囊滑环_部件故障_接入不好\n线路_部件故障_短路\n驾驶员侧安全气囊引爆装置_部件故障_电阻太小\n驾驶员侧安全气囊引爆装置_部件故障_电阻太大\n气囊滑环上两短接片_部件故障_有些短\n线路_部件故障_开路\n线路_部件故障_短路"
  },
  
]
```
##### 1.2 Medical_datasets
```json
[
  {
    "input": "Hello doctor,I had mumps five months ago and after that, I started to have an infection in my left testes. It was swollen and now it has shrunk to almost half the size of the other one. As I am sexually active, I feel a pain in each of the vas deferens after sex. If I do not have sex for days, they become sensitive. I was treated with Ceftum 500 mg, the first time I had an infection. Now my question is, is there any chance that the infection is still in my body? And, do I need to get examined for it? For the time being, please suggest some precautionary antibiotics for my relief.",
    "answer_icliniq": "Hello, Welcome to Chat Doctor forum. I can understand your concern. You had mumps and this is a viral infection known to cause an inflammation of the testis in some cases. Take care. For more information consult a sexologist online ",
    "answer_chatgpt": "It is possible that you still have an infection in your testes, especially if you have been experiencing pain and sensitivity in your vas deferens. It is recommended that you consult with a doctor or urologist for an examination and further testing to determine if there is an ongoing infection.\n\nAs an AI language model, I cannot prescribe medications or antibiotics. However, taking antibiotics without a doctor's prescription is not recommended. It is important to consult with a healthcare professional for proper diagnosis and treatment.",
    "answer_chatdoctor": "1) There is no need to worry about reinfection as the infection caused by Mumps virus does not persist in the body and is completely cleared up by the immune system.2) The swelling at your scrotum is due to inflammation of epididymis or vas deferens which is a result of mumps infection.3) There is no need to take antibiotics for this problem. You can take anti-inflammatory medicines like Ibuprofen for symptomatic relief.4) Doing cool compresses over the scrotal area will help in reducing the inflammation and pain."
  },
]
```
##### 1.3 GLUE_benchmark_datasets & SquADv1.0： please reference link (https://drive.google.com/drive/folders/1WE2em4vTFYRl64w_OiK_7ZgsZAh3s59k?usp=drive_link)



#### 2. For example, use Lora method to FineTuning mechanical_manufact datasets

- **deepspeed --include=localhost:0,1**: &nbsp;&nbsp;deepspeed more device runing。
- **model_name_or_path**: &nbsp;&nbsp;pre-model weights path。
- **train_file**: &nbsp;&nbsp;train data。
- **output_dir**: &nbsp;&nbsp;save model path。
- **prompt_column**: &nbsp;&nbsp;
- **response_column**: &nbsp;&nbsp;
- **max_source_length**: &nbsp;&nbsp;
- **max_target_length**: &nbsp;&nbsp;
- **per_device_train_batch_size**: &nbsp;&nbsp;
- **gradient_accumulation_steps**: &nbsp;&nbsp;
- **logging_steps**: &nbsp;&nbsp;
- **save_steps**: &nbsp;&nbsp;
- **learning_rate**: &nbsp;&nbsp;AdamW 
- **num_train_epochs**: &nbsp;&nbsp;
- **fp16**: &nbsp;&nbsp;
- **lora_rank**: &nbsp;&nbsp;LoRA 
```sh
deepspeed --include=localhost:0,1  train_qlora.py \
  --train_file /XXX/spo_0.json \
  --model_name_or_path /LLM/baichuan-inc/Baichuan2-7B-Chat \
  --output_dir output15 \
  --prompt_column input \
  --response_column output \
  --cache_dir goat_cache \
  --overwrite_cache \
  --overwrite_output_dir \
  --max_source_length 512 \
  --max_target_length 512 \
  --num_train_epochs 4 \
  --gradient_accumulation_steps 4 \
  --per_device_train_batch_size 1 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type cosine \
  --save_total_limit 1 \
  --learning_rate 2e-5 \
  --logging_steps 50 \
  --save_steps 500 \
  --lora_rank 2 \
  --fp16 \
  --torch_dtype float16 \
  --preprocessing_num_workers 2 \
  --deepspeed ds_config.json

```

#### 3. Test the fine-tuned model
- **CUDA_VISIBLE_DEVICES=0**: &nbsp;&nbsp; Single card operation.
- **do_eval**: &nbsp;&nbsp;
- **model_name_or_path**: &nbsp;&nbsp;
- **checkpoint_dir**: &nbsp;&nbsp;
- **dataset_dir**: &nbsp;&nbsp;
- **dataset**: &nbsp;&nbsp;
- **output_dir**: &nbsp;&nbsp;
- **per_device_eval_batch_size**：&nbsp;&nbsp;
- **predict_with_generate**: &nbsp;&nbsp;
- **padding_side**: &nbsp;&nbsp; pad alignment, left aligned or right aligned.

```sh
CUDA_VISIBLE_DEVICES=0 python predict_fora.py \
    --ori_model_dir baichuan-inc/Baichuan-13B-Chat \
    --model_dir 19baichuan-13B-lora_-mlp_qa-output15 \
    --test_path spo_1.json \
    --output_dir 19baichuan-13B-lora_-mlp_qa-output15 \
    --max_len 512 \
    --max_src_len 512
```

#### 4.FoRA FineTuning
##### 4.1 Micro-Fora
- **Quick way**：Overwrite the lora.py file in PEFT in the huggingface directory with fusion_lora.py [e.g. miniconda3/envs/XXX/lib/python3.10/site-packages/peft/tuners/], 
  and then perform step 2 again -3
##### 4.2 Meso-Fora
invoke find_all_linear_names function to adapter all linear layer
```sh
def find_all_linear_names(model):
    """
      Find all fully connected layers and add FoRA to all fully connected layers
    """
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

.........

#target_modules = find_all_linear_names(model)
print(target_modules)
target_modules = ['W_pack', 'o_proj']
rank=training_args.lora_rank
config = LoraConfig(r=training_args.lora_rank,
                    lora_alpha=rank*2,
                    target_modules=target_modules,
                    lora_dropout=0.1,
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    bias="none"
                    )
```

#### 5 Acknowledgments
We sincerely appreciate the contributions of the following methods
>[FATE LLM ](https://github.com/FederatedAI/FATE-LLM)
> 
>[Baichuan-13B](https://github.com/baichuan-inc/Baichuan-13B)
> 
>[PEFT](https://github.com/huggingface/peft)
> 
>[LoRA](https://github.com/microsoft/LoRA)


## Reference
If you find this repository useful or our work is related to your research, please kindly cite it:
```
@ARTICLE{10856857,
  author={Li, Chuantao and Gu, Bruce and Zhao, Zhigang and Qu, Youyang and Xin, Guomao and Huo, Jidong and Gao, Longxiang},
  journal={Big Data Mining and Analytics}, 
  title={Federated Transfer Learning for On-Device LLMs Efficient Fine Tuning Optimization}, 
  year={2025},
  volume={8},
  number={2},
  pages={430-446},
  keywords={Industries;Quantization (signal);Federated learning;Large language models;Transfer learning;Transformers;Data models;Protection;Optimization;Tuning;Federated learning;fine-tuning;deep learning;Large Language Models (LLMs)},
  doi={10.26599/BDMA.2024.9020068}}

```
