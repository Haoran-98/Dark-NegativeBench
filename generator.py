import argparse
import re
from tool import *


def run(args):
    """
    Evaluation of the main program for LLMs
    @param args:Input parameters
    @return:
    """
    print("*********************************************************")
    print("↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ Input Parameters ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓")
    print(f"             model: {args.model}")
    masking_key = re.sub(r'\d+', '*', args.key)
    print(f"             key: {masking_key}")
    print(f"             questionnaires: {args.questionnaires}")
    print(f"             language: {args.language}")
    print(f"             run_count: {args.run_count}")
    print("*********************************************************")
    questionnaires_list = args.questionnaires.split(',')
    for questionnaire_name in questionnaires_list:
        log(questionnaire_name, args.model, args.language),
        response_content_list = []
        for i in range(10):
            response_content = llms_api(questionnaire_name, args.model, args.key, args.language)
            response_content_list.append(response_content)
            save_result_json_file(response_content_list, questionnaire_name, args.model, args.language, args.run_count)
        calculate_result_json_file_score(questionnaire_name, args.model, args.language, args.run_count)
        result_display(questionnaire_name, args.model, args.language, args.run_count)


# For testing purposes
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.model = 'gpt-3.5-turbo'
    args.language = 'en'
    args.run_count = 1
    args.questionnaires = 'DTDD'
    args.key = 'sk-'
    run(args)
