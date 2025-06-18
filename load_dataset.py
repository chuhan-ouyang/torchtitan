import os
from datasets import load_dataset, config

os.environ["HUGGINGFACE_HUB_TOKEN"] = "..."

# ds = load_dataset(
#     "allenai/c4",
#     "en",
#     split="train",
#     streaming=False,                         # fully download
#     cache_dir="/pscratch/sd/c/co232/hf_cache",  # catch all entry
#     download_mode="reuse_cache_if_exists",
# )

# 1) Ensure no HTTP requests are allowed
config.HF_DATASETS_OFFLINE = True

ds = load_dataset(
    "allenai/c4",                                # dataset ID
    name="en",                                   # config name
    split="train",                               # only the train split
    cache_dir="/pscratch/sd/c/co232/hf_cache/allenai___c4",
    streaming=False,                             # map-style access
    download_mode="reuse_cache_if_exists"        # reuse what's already there
)

# 3) Save it in the proper Dataset.on-disk format
output_path = "/pscratch/sd/c/co232/hf_cache/c4_train_saved"
ds.save_to_disk(output_path, num_proc=32)

print(f"Saved train split to {output_path}")
