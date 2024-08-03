import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='llama3')
    args = parser.parse_args()
    if args.model_name == 'llama3':
        os.system('sh scripts/llama3.sh')
    elif args.model_name == 'qwen2':
        os.system('sh scripts/qwen2.sh')
    elif args.model_name == 'unichat':
        os.system('sh scripts/unichat-llama3.sh')
    else:
        raise ValueError('Unsupported model name')
    