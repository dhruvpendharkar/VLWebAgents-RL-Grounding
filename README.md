##TODO:

Resolve job scheduling issues to ensure that this can be run on IBM cluster


## 🛠️ Setup

- Install the environment by following the instructions [here](https://github.com/om-ai-lab/VLM-R1?tab=readme-ov-file#%EF%B8%8F-setup).
- Download the training dataset from the [link](https://huggingface.co/datasets/HelloKKMe/grounding_dataset/tree/main).
- If using a custom dataset, please store the data as a list of JSON objects, where each entry follows the structure below:

<pre>
{
    "image": "images/4.png",
    "bbox": [38, 166, 961, 218],
    "conversations": [
        {
            "from": "human",
            "value": "<image>Click on the search bar"
        },
        {
            "from": "gpt",
            "value": "any thing here"
        }
    ]
}
</pre>

**Note:** The bounding box (`bbox`) should use the format `[x0, y0, x1, y1]`, with all coordinates normalized to the range `[0, 1000]`.

## 🧹 Data Cleaning

We provide a cleaned version of the dataset on [Hugging Face](https://huggingface.co/datasets/HelloKKMe/grounding_dataset/tree/main).

If you're using a custom dataset, please refer to the [preprocessing folder](preprocessing/README.md) for instructions on how to clean and format your data.

## [Custom Use] Train Your Model
An example script on slurm
```shell
module load *** ... loading enviroment ***
export *** ... setting your own enviroment vairable ***

export RDZV_HOST=$(hostname)
export RDZV_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export RDZV_PORT=29505

RUN_NAME=test
srun torchrun \
    --nnodes $SLURM_JOB_NUM_NODES \
    --nproc_per_node 8 \
    --max-restarts 3 \
    --rdzv_id $SLURM_JOB_ID \
    --rdzv_backend c10d \
    --rdzv_endpoint "$RDZV_HOST:$RDZV_PORT"  src/grpo_grounding.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir grounding/$RUN_NAME \
    --model_name_or_path "Qwen/Qwen3-VL-4B-Instruct"  \
    --dataset_name preprocessing/inp.json \
    --image_root "./preprocessing" \
    --max_prompt_length 1024 \
    --max_completion_length 128 \
    --num_generations 8 \
    --per_device_train_batch_size 1 \
    --freeze_vision_modules true \
    --reward_funcs accuracy \
    --beta 0 \
    --dataloader_num_workers 2 \
    --max_pixels $((4096 * 2160)) \
    --gradient_accumulation_steps 32 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to tensorboard \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 2 \
    --run_name output/$RUN_NAME \
    --save_steps 10 \
    --save_total_limit 4 \
    --save_only_model false
```
✅ Make sure to modify paths, model names, and any relevant hyperparameters based on your specific setup.

## License

This dataset follows CC-BY-NC-SA 4.0 license. Please use this dataset for non-commercial use ONLY.

## Citation
If you use this repository or find it helpful in your research, please cite it as follows:
```bibtex
@misc{yang2025gta1guitesttimescaling,
      title={GTA1: GUI Test-time Scaling Agent}, 
      author={Yan Yang and Dongxu Li and Yutong Dai and Yuhao Yang and Ziyang Luo and Zirui Zhao and Zhiyuan Hu and Junzhe Huang and Amrita Saha and Zeyuan Chen and Ran Xu and Liyuan Pan and Silvio Savarese and Caiming Xiong and Junnan Li},
      year={2025},
      eprint={2507.05791},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2507.05791}, 
}
```

