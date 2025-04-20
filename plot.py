import json
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="plot args.")
    parser.add_argument(
        "-i", "--sample-input-file", type=str, default="aime.json"
    )
    args = parser.parse_args()
    
    with open(args.sample_input_file, "r") as f:
        output_info = json.load(f)
    
    print(output_info.keys())
    