### Introduction
The proliferation of large language models (LLMs) has catalyzed the growth of various industries, including smart healthcare and education. It is therefore imperative to ensure the controlled and beneficial application of LLMs across specific domains for downstream tasks through transfer learning, while preserving their general capabilities. 

We propose a novel, on-device efficient fine-tuning optimization algorithm for LLMs, utilizing federated transfer learning in response to these challenges. 

Specifically, we introduce the Fusion of Low Rank (FoRA) optimization algorithm from a micro perspective, which enhances multi-dimensional feature aggregation through the addition of efficient parameters. 

From a meso perspective, we extend the application of the FoRA algorithm across all linear layers within the transformer architecture to facilitate downstream task performance. 

Finally, from a macro perspective and with a focus on the medical domain, we incorporate quantization techniques into the federated learning framework to achieve on-device efficient fine-tuning optimization, thereby offering dual protection for data and model integrity.


At present, we will upload the code related to PEFT first, and then upload part of the FATE code after the review of our paper is completed.
[FATE LLM ](https://github.com/FederatedAI/FATE-LLM)
[Baichuan-13B](https://github.com/baichuan-inc/Baichuan-13B)

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
#### 1. dataset process
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
  {
    "instruction": "你现在是一个信息抽取模型，请你帮我抽取出关系内容为\"性能故障\", \"部件故障\", \"组成\"和 \"检测工具\"的相关三元组，三元组内部用\"_\"连接，三元组之间用\\n分割。文本：", 
    "input": "故障原因简要分析客户说凉车时打方向有时传来吱吱的异响，有时还没有。到店时没有，进行检查，把方向盘拆下检查组合开关，也没有发现异常，经客户同意把车留下我们进行观察。当天晚上快下班时我去试车，故障来了，赶紧逐一拆下方向盘的各个原件，但声音还在，这时用手触摸转向柱时有种异样感觉，贴身过去一听，声音是从转向中柱传出来的，看来毛病在这，但转向柱不可拆解，只能更换。件到后更换上故障排除。维修方案更换转向柱总成。", 
    "output": "车_部件故障_异响"
  }
]
```

#### 2. Lora FineTuning

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
  --train_file /LLM/Abirate/spo_0.json \
  --model_name_or_path /LLM/baichuan-inc/Baichuan2-7B-Chat \
  --output_dir 19baichuan-13B-lora_-mlp_qa-output15 \
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
#####Quick way: Overwrite the lora.py file in PEFT in the huggingface directory with fusion_lora.py [e.g. miniconda3/envs/XXX/lib/python3.10/site-packages/peft/tuners/], 
  and then perform step 2 again -3


