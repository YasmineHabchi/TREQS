'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import glob
import json
import os
import pickle
import re
import shutil
import time
from pprint import pprint

import numpy as np
import torch
from torch.autograd import Variable

from LeafNATS.data.utils import create_batch_file
from LeafNATS.utils.utils import show_progress


class End2EndBase(object):
    '''
    This engine is for the end2end training for seq2seq models.
    It is a new version of previous one. 
    Our goal is to extend its application to all kinds of language generation tasks.
    '''

    def __init__(self, args=None):
        '''
        Initialize
        '''
        self.args = args
        self.base_models = {}
        self.train_models = {}
        self.batch_data = {}
        self.test_data = {}
        self.global_steps = 0

    def build_vocabulary(self):
        '''
        vocabulary
        '''
        raise NotImplementedError

    def build_models(self):
        '''
        Models:
            self.base_models: models that will be trained
                Format: {'name1': model1, 'name2': model2}
            self.train_models: models that will be trained.
                Format: {'name1': model1, 'name2': model2}
        '''
        raise NotImplementedError

    def init_base_model_params(self):
        '''
        Initialize Base Model Parameters.
        self.base_models.
        '''
        raise NotImplementedError

    def build_pipelines(self):
        '''
        Pipelines and loss here.
        '''
        raise NotImplementedError

    def build_optimizer(self, params):
        '''
        define optimizer
        '''
        raise NotImplementedError

    def print_info_train(self):
        '''
        Print additional information on screen.
        '''
        print('NATS Message: ')

    def build_batch(self, batch_id):
        '''
        process batch data.
        '''
        raise NotImplementedError

    def test_worker(self):
        '''
        Used in decoding.
        Users can define their own decoding process.
        You do not have to worry about path and prepare input.
        '''
        raise NotImplementedError

    def app_worker(self, question):
        '''
        For application.
        '''
        
        max_lens=400
        """file_path = Your dataset path""""
        vocab2id, id2vocab = construct_vocab(file_path,
                    max_size=200000,
                    mincount=5)
        vocab_size = len(vocab2id)
        emb_dim = 128
        self.args.batch_size = 1
        
        self.batch_data = {}
        cnt = len(vocab2id)
        
        quest_tok = word_tokenize(question)
        dart = quest_tok
        #print(dart)
        ext_vocab = {}
        ext_id2oov = {}
        for wd in dart:
            if wd not in vocab2id:
                ext_vocab[wd] = {}
        for wd in ext_vocab:
            ext_vocab[wd] = cnt
            ext_id2oov[cnt] = wd
            cnt += 1
        
        src_txt = [dart]
        # UNK
        dart2id = [
            vocab2id[wd] if wd in vocab2id
            else vocab2id['<unk>']
            for wd in dart]
        
        src_idx = [dart2id]
        
        dart2id_ex = [
            vocab2id[wd] if wd in vocab2id
            else ext_vocab[wd]
            for wd in dart]
       
        src_idx_ex = [dart2id_ex]
        # UNK mask
        dart2wt = [0.0 if wd in vocab2id else 1.0 for wd in dart]
        
        src_mask_unk = [dart2wt]

        cnt = len(vocab2id)
        
        src_max_lens = 400
        trg_max_lens = 400
        
        src_idx = [
            itm + [vocab2id['<pad>']]*(src_max_lens-len(itm))
            for itm in src_idx]
        
        src_var = Variable(torch.LongTensor(src_idx)).to(self.args.device)
        
        src_idx_ex = [itm[:src_max_lens] for itm in src_idx_ex]
        src_idx_ex = [
            itm + [vocab2id['<pad>']]*(src_max_lens-len(itm))
            for itm in src_idx_ex]
        #src_idx_ex[0][-1] = vocab2id['<stop>']
        #print("src_idx_ex",src_idx_ex)
        src_var_ex = torch.LongTensor(src_idx_ex).to(self.args.device)
        
        src_mask_unk = [
            itm[:src_max_lens] for itm in src_mask_unk]
        src_mask_unk = [
            itm + [0.0]*(src_max_lens-len(itm))
            for itm in src_mask_unk]
        src_mask_unk = Variable(
            torch.FloatTensor(src_mask_unk)).to(self.args.device)

        src_mask_pad = Variable(torch.FloatTensor(src_idx)).to(self.args.device)
        src_mask_pad[src_mask_pad
                               != float(vocab2id['<pad>'])] = -1.0
        src_mask_pad[src_mask_pad
                               == float(vocab2id['<pad>'])] = 0.0
        src_mask_pad = -1.0*src_mask_pad
        self.batch_data['src_mask_pad'] = src_mask_pad
        self.batch_data['vocab2id'] = vocab2id
        self.batch_data['batch_size'] = 1
        self.batch_data['trg_seq_len'] = 1
        self.batch_data['vocab_size'] = 2353
        self.batch_data['src_seq_len'] = 400
        self.batch_data['src_var'] = src_var
        self.batch_data['id2vocab'] = id2vocab
        self.batch_data['src_mask_unk'] = src_mask_unk
        self.batch_data['ext_id2oov'] = ext_id2oov
        self.batch_data['src_txt'] = src_txt
        
        
        ques_id = []
        for e in quest_tok:
            if e in vocab2id:
                ques_id.append(int(vocab2id[e]))
                
            else:
                ques_id.append(int(vocab2id['<unk>']))
        
        src_text = src_txt
       
        batch_size = src_var.size(0)
        src_seq_len = src_var.size(1)
        
        
        src_emb = self.base_models['embedding'](src_var).to(self.args.device)
        pipe_data = {}
        self.pipe_data['encoder'] = {}
        self.pipe_data['encoder']['src_emb'] = src_emb
        
        self.pipe_data['decoderA'] = {}
        hy_encoder, (ht_encoder, ct_encoder) = self.base_models['encoder'](src_emb)
        src_enc, hidden_encoder = hy_encoder, (ht_encoder, ct_encoder)
        self.pipe_data['encoder']['src_enc'] = src_enc
        enhy = hy_encoder
        (decoder_h0, decoder_c0) = self.base_models['encoder2decoder'](hidden_encoder)
        trg_hidden0 = (decoder_h0, decoder_c0)
        past_attn=Variable(torch.ones(1, src_seq_len
        )/float(src_seq_len)).to(self.args.device)
        
        past_dech = Variable(torch.zeros(1, 1)).to(self.args.device)
        self.pipe_data['decoderB'] = {}
        h_attn= Variable(torch.zeros(self.args.batch_size,self.args.trg_hidden_dim)).to(self.args.device)
        output_ = []
        accu_attn = []
        genprb = []
        
        self.beam_search()
        try:
            myseq = self.word_copy()
        except: 
            print('Running without manually word copying.')
            myseq = torch.cat(self.beam_data[0][0], 0)
            myseq = myseq.data.cpu().numpy().tolist()
            myseq = [self.batch_data['id2vocab'][idx] for idx in myseq]

        
        pred = ' '.join(myseq)
        print(pred)
        return(pred[:-6])

    def train(self):
        '''
        training here.
        Don't overwrite.
        '''
        self.build_vocabulary()
        self.build_models()
        print(self.base_models)
        print(self.train_models)
        if len(self.base_models) > 0:
            self.init_base_model_params()
        # here it is necessary to put list. Instead of directly append.
        for model_name in self.train_models:
            try:
                params += list(self.train_models[model_name].parameters())
            except:
                params = list(self.train_models[model_name].parameters())
        if self.args.train_base_model:
            for model_name in self.base_models:
                try:
                    params += list(self.base_models[model_name].parameters())
                except:
                    params = list(self.base_models[model_name].parameters())
        # define optimizer
        optimizer = self.build_optimizer(params)
        # load checkpoint
        uf_model = [0, -1]
        out_dir = os.path.join('..', 'nats_results')
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        if self.args.continue_training:
            model_para_files = glob.glob(os.path.join(out_dir, '*.model'))
            if len(model_para_files) > 0:
                uf_model = []
                for fl_ in model_para_files:
                    arr = re.split(r'\/', fl_)[-1]
                    arr = re.split(r'\_|\.', arr)
                    arr = [int(arr[-3]), int(arr[-2])]
                    if arr not in uf_model:
                        uf_model.append(arr)
                cc_model = sorted(uf_model)[-1]
                try:
                    print("Try *_{}_{}.model".format(cc_model[0], cc_model[1]))
                    for model_name in self.train_models:
                        fl_ = os.path.join(
                            out_dir, model_name+'_'+str(cc_model[0])+'_'+str(cc_model[1])+'.model')
                        self.train_models[model_name].load_state_dict(
                            torch.load(fl_, map_location=lambda storage, loc: storage))
                except:
                    cc_model = sorted(uf_model)[-2]
                    print("Try *_{}_{}.model".format(cc_model[0], cc_model[1]))
                    for model_name in self.train_models:
                        fl_ = os.path.join(
                            out_dir, model_name+'_'+str(cc_model[0])+'_'+str(cc_model[1])+'.model')
                        self.train_models[model_name].load_state_dict(
                            torch.load(fl_, map_location=lambda storage, loc: storage))
                print(
                    'Continue training with *_{}_{}.model'.format(cc_model[0], cc_model[1]))
                uf_model = cc_model

        else:
            shutil.rmtree(out_dir)
            os.mkdir(out_dir)
        # train models
        fout = open('../nats_results/args.pickled', 'wb')
        pickle.dump(self.args, fout)
        fout.close()
        start_time = time.time()
        cclb = 0
        for epoch in range(uf_model[0], self.args.n_epoch):
            n_batch = create_batch_file(
                path_data=self.args.data_dir,
                path_work=os.path.join('..', 'nats_results'),
                is_shuffle=True,
                fkey_=self.args.task,
                file_=self.args.file_train,
                batch_size=self.args.batch_size,
                is_lower=self.args.is_lower
            )
            print('The number of batches: {}'.format(n_batch))
            self.global_steps = n_batch * max(0, epoch)
            for batch_id in range(n_batch):
                self.global_steps += 1
                if cclb == 0 and batch_id <= uf_model[1]:
                    continue
                else:
                    cclb += 1

                self.build_batch(batch_id)
                loss = self.build_pipelines()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, self.args.grad_clip)
                optimizer.step()

                end_time = time.time()
                if batch_id % self.args.checkpoint == 0:
                    for model_name in self.train_models:
                        fmodel = open(os.path.join(
                            out_dir, model_name+'_'+str(epoch)+'_'+str(batch_id)+'.model'), 'wb')
                        torch.save(
                            self.train_models[model_name].state_dict(), fmodel)
                        fmodel.close()
                if batch_id % 1 == 0:
                    end_time = time.time()
                    print('epoch={}, batch={}, loss={}, time_escape={}s={}h'.format(
                        epoch, batch_id,
                        np.round(float(loss.data.cpu().numpy()), 4),
                        np.round(end_time-start_time, 2),
                        np.round((end_time-start_time)/3600.0, 4)
                    ))
                    self.print_info_train()
                del loss

            for model_name in self.train_models:
                fmodel = open(os.path.join(
                    out_dir, model_name+'_'+str(epoch)+'_'+str(batch_id)+'.model'), 'wb')
                torch.save(self.train_models[model_name].state_dict(), fmodel)
                fmodel.close()

    def validate(self):
        '''
        Validation here.
        Don't overwrite.
        '''
        self.build_vocabulary()
        self.build_models()
        pprint(self.base_models)
        pprint(self.train_models)
        if len(self.base_models) > 0:
            self.init_base_model_params()

        best_arr = []
        val_file = os.path.join('..', 'nats_results', 'model_validate.txt')
        if os.path.exists(val_file):
            fp = open(val_file, 'r')
            for line in fp:
                arr = re.split(r'\s', line[:-1])
                best_arr.append(
                    [arr[0], arr[1], arr[2], float(arr[3]), float(arr[4])])
            fp.close()

        for model_name in self.base_models:
            self.base_models[model_name].eval()
        for model_name in self.train_models:
            self.train_models[model_name].eval()
        with torch.no_grad():
            while 1:
                model_para_files = []
                model_para_files = glob.glob(os.path.join(
                    '..', 'nats_results', sorted(list(self.train_models))[0]+'*.model'))
                for j in range(len(model_para_files)):
                    arr = re.split(r'\_|\.', model_para_files[j])
                    arr = [int(arr[-3]), int(arr[-2]), model_para_files[j]]
                    model_para_files[j] = arr
                model_para_files = sorted(model_para_files)

                for fl_ in model_para_files:
                    best_model = {itm[0]: itm[3] for itm in best_arr}
                    if fl_[-1] in best_model:
                        continue
                    print('Validate *_{}_{}.model'.format(fl_[0], fl_[1]))

                    losses = []
                    start_time = time.time()
                    if os.path.exists(fl_[-1]):
                        time.sleep(3)
                        try:
                            for model_name in self.train_models:
                                fl_tmp = os.path.join(
                                    '..', 'nats_results',
                                    model_name+'_'+str(fl_[0])+'_'+str(fl_[1])+'.model')
                                self.train_models[model_name].load_state_dict(
                                    torch.load(fl_tmp, map_location=lambda storage, loc: storage))
                        except:
                            continue
                    else:
                        continue
                    val_batch = create_batch_file(
                        path_data=self.args.data_dir,
                        path_work=os.path.join('..', 'nats_results'),
                        is_shuffle=True,
                        fkey_=self.args.task,
                        file_=self.args.file_val,
                        batch_size=self.args.batch_size
                    )
                    print('The number of batches (test): {}'.format(val_batch))
                    if self.args.val_num_batch > val_batch:
                        self.args.val_num_batch = val_batch
                    for batch_id in range(self.args.val_num_batch):

                        self.build_batch(batch_id)
                        loss = self.build_pipelines()

                        losses.append(loss.data.cpu().numpy())
                        show_progress(batch_id+1, self.args.val_num_batch)
                    print()
                    losses = np.array(losses)
                    end_time = time.time()
                    if self.args.use_move_avg:
                        try:
                            losses_out = 0.9*losses_out + \
                                0.1*np.average(losses)
                        except:
                            losses_out = np.average(losses)
                    else:
                        losses_out = np.average(losses)
                    best_arr.append(
                        [fl_[2], fl_[0], fl_[1], losses_out, end_time-start_time])
                    best_arr = sorted(best_arr, key=lambda bb: bb[3])
                    if best_arr[0][0] == fl_[2]:
                        out_dir = os.path.join('..', 'nats_results', 'model')
                        try:
                            shutil.rmtree(out_dir)
                        except:
                            pass
                        os.mkdir(out_dir)
                        for model_name in self.base_models:
                            fmodel = open(os.path.join(
                                out_dir, model_name+'.model'), 'wb')
                            torch.save(
                                self.base_models[model_name].state_dict(), fmodel)
                            fmodel.close()
                        for model_name in self.train_models:
                            fmodel = open(os.path.join(
                                out_dir, model_name+'.model'), 'wb')
                            torch.save(
                                self.train_models[model_name].state_dict(), fmodel)
                            fmodel.close()
                        try:
                            shutil.copy2(os.path.join(
                                self.args.data_dir, self.args.file_vocab), out_dir)
                        except:
                            pass
                    for itm in best_arr[:self.args.nbestmodel]:
                        print('model={}_{}, loss={}, time={}'.format(
                            itm[1], itm[2], np.round(float(itm[3]), 4),
                            np.round(float(itm[4]), 4)))

                    for itm in best_arr[self.args.nbestmodel:]:
                        tarr = re.split(r'_|\.', itm[0])
                        if tarr[-2] == '0':
                            continue
                        if os.path.exists(itm[0]):
                            for model_name in self.train_models:
                                fl_tmp = os.path.join(
                                    '..', 'nats_results',
                                    model_name+'_'+str(itm[1])+'_'+str(itm[2])+'.model')
                                os.unlink(fl_tmp)
                    fout = open(val_file, 'w')
                    for itm in best_arr:
                        if len(itm) == 0:
                            continue
                        fout.write(' '.join([itm[0], str(itm[1]), str(
                            itm[2]), str(itm[3]), str(itm[4])])+'\n')
                    fout.close()

    def test(self):
        '''
        testing
        Don't overwrite.
        '''
        self.args.batch_size = 1

        self.build_vocabulary()
        self.build_models()
        pprint(self.base_models)
        pprint(self.train_models)
        if len(self.base_models) > 0:
            self.init_base_model_params()

        _nbatch = create_batch_file(
            path_data=self.args.data_dir,
            path_work=os.path.join('..', 'nats_results'),
            is_shuffle=False,
            fkey_=self.args.task,
            file_=self.args.file_test,
            batch_size=self.args.batch_size
        )
        print('The number of samples (test): {}'.format(_nbatch))

        for model_name in self.base_models:
            self.base_models[model_name].eval()
        for model_name in self.train_models:
            self.train_models[model_name].eval()
        with torch.no_grad():
            if self.args.use_optimal_model:
                model_valid_file = os.path.join(
                    '..', 'nats_results', 'model_validate.txt')
                fp = open(model_valid_file, 'r')
                for line in fp:
                    arr = re.split(r'\s', line[:-1])
                    model_optimal_key = ''.join(
                        ['_', arr[1], '_', arr[2], '.model'])
                    break
                fp.close()
            else:
                arr = re.split(r'\D', self.args.model_optimal_key)
                model_optimal_key = ''.join(
                    ['_', arr[0], '_', arr[1], '.model'])
            print("You choose to use *{} for decoding.".format(model_optimal_key))

            for model_name in self.train_models:
                model_optimal_file = os.path.join(
                    '..', 'nats_results', model_name+model_optimal_key)
                self.train_models[model_name].load_state_dict(torch.load(
                    model_optimal_file, map_location=lambda storage, loc: storage))

            start_time = time.time()
            output_file = os.path.join('evaluation/generated_sql',
                                       self.args.file_output)
            data_check = []
            try:
                self.args.continue_decoding
            except:
                self.args.continue_decoding = True
            if os.path.exists(output_file) and self.args.continue_decoding:
                fchk = open(output_file, 'r')
                for line in fchk:
                    data_check.append(line)
                fchk.close()
                data_check = data_check[:-1]
                fchk = open(output_file, 'w')
                for line in data_check:
                    fchk.write(line)
                fchk.close()
            else:
                fout = open(output_file, 'w')
                fout.close()
            try:
                fout = open(output_file, 'a')
            except:
                fout = open(output_file, 'w')
            for batch_id in range(_nbatch):
                if batch_id < len(data_check):
                    continue
                self.build_batch(batch_id)
                self.test_worker()
                json.dump(self.test_data, fout)
                fout.write('\n')

                end_time = time.time()
                show_progress(batch_id, _nbatch, str(
                    (end_time-start_time)/3600)[:8]+"h")
            fout.close()
            print()

    def app2Go(self):
        '''
        For the application.
        Don't overwrite.
        '''
        self.build_vocabulary()
        self.build_models()
        for model_name in self.train_models:
            self.base_models[model_name] = self.train_models[model_name]
        pprint(self.base_models)
        if len(self.base_models) > 0:
            self.init_base_model_params()

        for model_name in self.base_models:
            self.base_models[model_name].eval()
        with torch.no_grad():
            while 1:
                sql_query=self.app_worker(question)
                return (sql_query)
