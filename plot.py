import json
import argparse
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="plot args.")
    parser.add_argument(
        "-i", "--sample-input-file", type=str, default="aime.json"
    )
    args = parser.parse_args()
    
    with open(args.sample_input_file, "r") as f:
        output_info = json.load(f)
    
    num_req = len(output_info["instances"])
    
    avg_score_list = []
    avg_decode_len_list = []
    prompt_len_list = []
    
    for idx in range(len(output_info["instances"])):
        avg_score_list.append(
            np.mean(output_info["instances"][idx]["score"])
        )
        
        avg_decode_len_list.append(
            np.mean(output_info["instances"][idx]["gen_length"])
        )
        
        prompt_len_list.append(
            output_info["instances"][idx]["raw_text_len"]
        )
    
    my_cmap = plt.get_cmap("Set2")
    
    fig, axes = plt.subplots(3, 1, figsize=(5, 15))

    # First dataset
    axes[0].bar(np.arange(num_req), avg_score_list, color=my_cmap(0))
    axes[0].set_title("acc")
    axes[0].set_ylabel("Values")

    # Second dataset
    axes[1].bar(np.arange(num_req), avg_decode_len_list, color=my_cmap(1))
    axes[1].set_title("decode len")
    axes[1].set_ylabel("Values")
    
    # Third dataset
    axes[2].bar(np.arange(num_req), prompt_len_list, color=my_cmap(2))
    axes[2].set_title("prompt len")
    axes[2].set_ylabel("Values")

    plt.tight_layout()
    plt.savefig("aime.pdf", bbox_inches="tight", dpi=500)