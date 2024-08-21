import argparse
from tool import *
from generator import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str, default='gpt-3.5-turbo',
                        help='Name of the model to be evaluated')
    # If you want to evaluate your own models or other models you can change the key here, or choose not to use the
    # key. If LLMAPI needs more than one key you can enter multiple keys here and separate them with commas,
    # for example, Baidu AI Cloud needs API_KEY and SECRET_KEY.
    parser.add_argument('--key', type=str, default='')
    parser.add_argument('--questionnaires', required=True, type=str,
                        help='Multiple scales can be selected separated by commas')
    parser.add_argument('--language', required=True, type=str,
                        help='Scale language en or zh')
    parser.add_argument('--run_count', required=True, type=int, default=1,
                        help='How many rounds you want to run? Default is 1')
    
    args = parser.parse_args()
    
    run(args)
