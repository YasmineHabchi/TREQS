'''
@author Ping Wang and Tian Shi
Please contact ping@vt.edu or tshi@vt.edu
'''
import uvicorn
import argparse
import torch
from model import modelABS
from LeafNATS.utils.utils import str2bool
from LeafNATS.engines.end2end_large import End2EndBase
from fastapi import FastAPI, HTTPException, Request
import re
app = FastAPI()
parser = argparse.ArgumentParser()
'''
Use in the framework and cannot remove.
'''
parser.add_argument('--task', default='train', help='train | validate | test | evaluate')

parser.add_argument('--data_dir', default='mimicsql_data/mimicsql_natural', help='directory that store the data.')
parser.add_argument('--file_train', default='train.json', help='Training')
parser.add_argument('--file_val', default='dev.json', help='validation')
parser.add_argument('--file_test', default='test.json', help='test data')

parser.add_argument('--n_epoch', type=int, default=20, help='number of epochs.')
parser.add_argument('--batch_size', type=int, default=16, help='batch size.')
parser.add_argument('--checkpoint', type=int, default=100, help='How often you want to save model?')
parser.add_argument('--val_num_batch', type=int, default=100, help='how many batches')
parser.add_argument('--nbestmodel', type=int, default=5, help='How many models you want to keep?')

parser.add_argument('--continue_training', type=str2bool, default=True, help='Do you want to continue?')
parser.add_argument('--train_base_model', type=str2bool, default=False, help='True: Use Pretrained Param | False: Transfer Learning')
parser.add_argument('--use_move_avg', type=str2bool, default=False, help='move average')
parser.add_argument('--use_optimal_model', type=str2bool, default=True, help='Do you want to use the best model?')
parser.add_argument('--model_optimal_key', default='0,0', help='epoch,batch')
parser.add_argument('--is_lower', type=str2bool, default=True, help='convert all tokens to lower case?')
'''
User specified parameters.
'''
parser.add_argument('--device', default=torch.device("cuda:0"), help='device')
parser.add_argument('--file_vocab', default='vocab', help='file store training vocabulary.')

parser.add_argument('--max_vocab_size', type=int, default=50000, help='max number of words in the vocabulary.')
parser.add_argument('--word_minfreq', type=int, default=5, help='min word frequency')

parser.add_argument('--emb_dim', type=int, default=128, help='source embedding dimension')
parser.add_argument('--src_hidden_dim', type=int, default=256, help='encoder hidden dimension')
parser.add_argument('--trg_hidden_dim', type=int, default=256, help='decoder hidden dimension')
parser.add_argument('--src_seq_len', type=int, default=400, help='length of source documents.')
parser.add_argument('--trg_seq_len', type=int, default=100, help='length of target documents.')
parser.add_argument('--nLayers', type=int, default=1, help='Encoder RNN layers')

parser.add_argument('--learning_rate', type=float, default=0.0005, help='learning rate.')
parser.add_argument('--grad_clip', type=float, default=2.0, help='clip the gradient norm.')

parser.add_argument('--file_output', default='output.json', help='test output file')
parser.add_argument('--beam_size', type=int, default=5, help='beam size.')
parser.add_argument('--copy_words', type=str2bool, default=True, help='Do you want to copy words?')
# scheduler
parser.add_argument('--step_size', type=int, default=2, help='---')
parser.add_argument('--step_decay', type=float, default=0.8, help='---')

args = parser.parse_args()
def convert(query):
    nums = ["0","1", "2", "3", "4", "5", "6", "7", "8", "9"]
    where_pos = query.find("where")
    query_list = list(query)
    
    i = where_pos
    while i<len(query):
        if query_list[i] in ["=", ">", "<"]:
            if query_list[i + 2] != '"':
                query_list[i + 3] = "'"
                query_list.append("'")
            if query_list[i + 2] == '"':
                if query_list[i+3] in nums:
                    query_list[i + 2] = ""
                else:
                    query_list[i + 2] = "'"
            k = i + 3
            while query_list[k] != '"':
                k += 1
            if query_list[k-1] in nums:
                query_list[k] = ""
            else:
                query_list[k] = "'"
        i += 1
    
    return ''.join(query_list)
def convert_1(chaine):
    
    query_list = list(chaine)
    query_list[-1]="'"
    cnt = 0
    for i in range(len(query_list)):
        if query_list[i] == "=":
            cnt+=1
            if cnt == 2:
                query_list[i+1] = "'"
    query = ''.join(query_list)
    return(query)
        
if args.task == 'train' or args.task == 'validate' or args.task == 'test':
    from model import modelABS
    model = modelABS(args)
if args.task == "train":
    model.train()
if args.task == "validate":
    model.validate()
if args.task == "test":
    model.test()

if args.task == "evaluate":
    from LeafNATS.eval_scripts.eval_pyrouge_v2 import run_pyrouge
    run_pyrouge(args)
if args.task == "input":
   
    model=modelABS(args)
    @app.post('/predict')
    async def predict(request: Request):
        data = await request.json()
        question = data['question']
        try:
            sql_query = model.app2Go(question)
            try:
                sql_query = convert(sql_query)
            except:
                sql_query = convert_1(sql_query)
            print(sql_query)
            return {"sql_query": sql_query}
        except Exception as e:
            print(e)
            raise HTTPException(status_code=500, detail=str(e))

    if __name__ == '__main__':  
        uvicorn.run(app, host='127.0.0.1', port=8000)
    
if args.task == "evaluate":
    from LeafNATS.eval_scripts.eval_pyrouge_v2 import run_pyrouge
    run_pyrouge(args)
