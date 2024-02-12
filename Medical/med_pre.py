# -*- coding: UTF-8 -*-
import torch
import json
#coding=utf-8
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from tqdm import tqdm
import time
import os
import argparse
from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig
def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', default='/LLM/Abirate/healthy_val_dataset.json', type=str, help='')
    parser.add_argument('--device', default='1', type=str, help='')
    parser.add_argument('--ori_model_dir',
                        default="/LLM/Baichuan2-13B-Chat/", type=str,
                        help='')
    parser.add_argument('--model_dir',
                        default="/LLM/baichuan/med-lora-output4/", type=str,help='')
    parser.add_argument('--max_len', type=int, default=512, help='')
    parser.add_argument('--max_src_len', type=int, default=450, help='')
    parser.add_argument('--prompt_text', type=str,
                        default="If you are a doctor, please answer the medical questions based on the patient's description.",
                        help='')
    return parser.parse_args()


def main():
    prefix="If you are a doctor, please answer the medical questions based on the patient's description."
    print(prefix)
    args = set_args()

    tokenizer = AutoTokenizer.from_pretrained(args.ori_model_dir, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(args.ori_model_dir, trust_remote_code=True)
    model.eval()
    model_lora = PeftModel.from_pretrained(model, args.model_dir,  trust_remote_code=True).half().to("cuda:1")
    # model.half().
    # model.eval()
    save_data = []
    f1 = 0.0
    # max_tgt_len = args.max_len - args.max_src_len - 3
    # max_tgt_len =512
    s_time = time.time()
    # model.generation_config = GenerationConfig.from_pretrained(args.ori_model_dir)
    import json
    
    with open(args.test_path, "r", encoding="utf-8") as fh:
        dataset = json.load(fh)
        for i, line in enumerate(tqdm(dataset, desc="iter")):
            with torch.no_grad():

                sample = line
                print(sample)
                src_tokens = tokenizer.tokenize(sample["input"])
                #ori_ll = sample["answer"]

                real_res = sample["answer_icliniq"]
                prompt_tokens = tokenizer.tokenize(args.prompt_text)
                print("start:-------------------------------------")
                print("sample_real:" + str(sample["answer_icliniq"]))
                # prompt = tokenizer.build_prompt(sample["input"])
                #prefix = "你现在是一个信息抽取模型，请你帮我抽取出关系内容为\"性能故障\", \"部件故障\", \"组成\"和 \"检测工具\"的相关三元组，三元组内部用\"_\"连接，三元组之间用\\n分割。文本："
                ppp=args.prompt_text
                prompt = prefix + sample["input"]
                inputs = tokenizer(prompt + "\n", return_tensors='pt')
                inputs = inputs.to('cuda:1')
                time_token = time.time()
                # print()
                lora_pred = model_lora.generate(**inputs, max_new_tokens=756, do_sample=True)
                lora_answer = tokenizer.decode(lora_pred.cpu()[0], skip_special_tokens=True)
                # res.append(r)
                pre_res = lora_answer.split("\n")

                    # [rr for rr in res[0].split("\n") if len(rr.split("_")) == 3]
                print("start:-----------------------------------")
                print("lora_answer:" + str(pre_res))
                print("end -------------------------------------")
                same_res = set(pre_res) & set(real_res)
                print("same_res："+str(same_res))
                if len(set(pre_res)) == 0:
                    p = 0.0
                else:
                    pre_lien1=set(pre_res[1:])
                    pre_lien = len(set(pre_res[1:]))
                    p = len(same_res) / len(set(pre_res[1:]))
                    p1 = len(same_res) / len(set(pre_res))
                r = len(same_res) / len(set(real_res))
                if (p + r) != 0.0:
                    f = 2 * p * r / (p + r)
                else:
                    f = 0.0
                f1 += f
                save_data.append(
                    {"text": sample["input"], "ori_answer": sample["answer_icliniq"], "gen_answer": pre_res[1:], "f1": f})

    e_time = time.time()
    print("总耗时：{}s".format(e_time - s_time))
    print(f1 / 50)
    save_path = os.path.join(args.model_dir, "ft_pt_answer.json")
    fin = open(save_path, "w", encoding="utf-8")
    json.dump(save_data, fin, ensure_ascii=False, indent=4)

    fin.close()


if __name__ == '__main__':
    main()
