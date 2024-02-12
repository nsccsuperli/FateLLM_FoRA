### 介绍
随着大模型迅速发展，通过迁移学习保留大语言模型通用能力且可控良性的应用到具体行业处理下游任务非常重要。而当前，在处理行业具体任务时存在数据隐私问题、模型隐私问题、客户端资源有限问题以及性能不足问题。在本文中，我们提出了一种基于联邦迁移学习的本地大模型高效微调优化算法，具体的，我们在微观视角下提出FoRA优化算法，通过增加高效训练参数提取多维度的特征。
随后，在介观视角下，我们将FoRA算法适配到整个Transformer-Layer中所有的Linear层。
最后，在宏观层面，面向医疗领域，我们将量化技术融入到联邦学习架构中，在数据保护和模型保护基础上实现本地大模型高效微调优化。实验中，我们在NLU、NLG和QA任务中进行了全面的实验，结果表明，我们算法有效提升了大模型性能同时对数据和模型双重保护。

目前我们将先上传和PEFT相关的代码，随后，当我们论文审稿完成后会上传FATE部分代码
[FATE LLM 官方链接](https://github.com/FederatedAI/FATE-LLM)
[Baichuan-13B官方链接](https://github.com/baichuan-inc/Baichuan-13B)

### 环境准备
#### 1. 下载微调项目
```sh

```

#### 2. 下载 baichuan-13B 和 7B 大模型
```sh
# 在根目录下新建 baichuan-inc 保存大模型
pip install modelscope
from modelscope.hub.snapshot_download import snapshot_download
model_dir = snapshot_download('baichuan-inc/Baichuan2-13B-Chat', cache_dir='/dt/ft',revision='v1.0.1')
model_dir = snapshot_download('baichuan-inc/Baichuan2-7B-Chat', cache_dir='/dt/ft',revision='v1.0.1')


#### 3. 创建conda环境，安装 python 库
```sh
conda create -n FATELLM python=3.10  
conda activate FATELLM
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

```

#### 4. 硬件要求
- LoRA 显存：>= 35G
- QLoRA 显存：>= 15G

### 微调
#### 1. 准备训练数据
`data` 目录下存储了训练数据，可根据业务需要自行准备数据，数据格式如下：
- **instruction**：任务指令，不能为空。
- **input**：任务输入，可为空。如果不为空，项目内部处理训练数据时，会将 instruction、input 拼接在一起作为任务的输入。
- **output**：任务输出，不能为空。
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

#### 2. Lora 微调

- **deepspeed --include=localhost:0,1**: &nbsp;&nbsp;deepspeed多卡运行。
- **model_name_or_path**: &nbsp;&nbsp;预训练模型路径。
- **train_file**: &nbsp;&nbsp;训练数据。
- **output_dir**: &nbsp;&nbsp;微调后的模型保存路径。
- **prompt_column**: &nbsp;&nbsp;数据集中的输入key名称。
- **response_column**: &nbsp;&nbsp;数据集中输出结果key名称。
- **max_source_length**: &nbsp;&nbsp;输入序列的最大长度，即 source_prefix + instruction + input 的长度。
- **max_target_length**: &nbsp;&nbsp;输出序列的最大长度，即 output 的长度。
- **per_device_train_batch_size**: &nbsp;&nbsp;用于训练的批处理大小。可根据 GPU 显存大小自行设置。
- **gradient_accumulation_steps**: &nbsp;&nbsp;梯度累加次数。
- **logging_steps**: &nbsp;&nbsp;多少步输出一次 log。
- **save_steps**: &nbsp;&nbsp;多少步保存一次参数。
- **learning_rate**: &nbsp;&nbsp;AdamW 优化器的初始学习率。
- **num_train_epochs**: &nbsp;&nbsp;训练轮数（若非整数，则最后一轮只训练部分数据）
- **fp16**: &nbsp;&nbsp;使用半精度（混合精度）训练。
- **lora_rank**: &nbsp;&nbsp;LoRA 微调中的秩大小。
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

#### 3. 测试微调后的模型
- **CUDA_VISIBLE_DEVICES=0**: &nbsp;&nbsp;单卡运行。
- **do_eval**: &nbsp;&nbsp;是否执行测试。
- **model_name_or_path**: &nbsp;&nbsp;预训练模型路径。
- **checkpoint_dir**: &nbsp;&nbsp;微调模型路径。
- **dataset_dir**: &nbsp;&nbsp;测试数据存储目录。
- **dataset**: &nbsp;&nbsp;测试数据集名称，可在 data/dataset_info.json 中增加自定义数据集。
- **output_dir**: &nbsp;&nbsp;测试结果保存路径。
- **per_device_eval_batch_size**：&nbsp;&nbsp;测试数据的批处理大小。可根据 GPU 显存大小，自行设置。
- **predict_with_generate**: &nbsp;&nbsp;是否生成序列用于计算 ROUGE 或 BLEU 分数。
- **padding_side**: &nbsp;&nbsp; pad对齐方式，左对齐或者右对齐。

```sh
CUDA_VISIBLE_DEVICES=0 python predict_fora.py \
    --ori_model_dir baichuan-inc/Baichuan-13B-Chat \
    --model_dir 19baichuan-13B-lora_-mlp_qa-output15 \
    --test_path spo_1.json \
    --output_dir 19baichuan-13B-lora_-mlp_qa-output15 \
    --max_len 512 \
    --max_src_len 512
```

#### 4.FoRA微调
#####快速方式：将fusion_lora.py对huggingface目录下的PEFT中的lora.py文件进行覆盖[e.g. miniconda3/envs/XXX/lib/python3.10/site-packages/peft/tuners/]，然后重新执行步骤2-3

####  更多参数设置
请参考 `config.py` 文件。


