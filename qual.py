import re
import sys
import nltk
nltk.download('punkt')
import numpy as np
from nltk import sent_tokenize
from metric.scorer import UniEvaluator
sys.path.append("..")
from prettytable import PrettyTable

class SumEvaluator:
    def __init__(self, max_length=1024, device='cuda:0', cache_dir=None):
        """ Set up evaluator for text summarization """
        self.scorer = UniEvaluator(model_name_or_path='MingZhong/unieval-sum', 
                                   max_length=max_length, 
                                   device=device, cache_dir=cache_dir)
        self.task = 'summarization'
        self.dimensions = ['coherence', 'consistency', 'fluency', 'relevance']
    
    def evaluate(self, data, dims=None, overall=True, print_result=False):
        """
            Get the scores of all the given dimensions

            dims: A list of dimensions to be evaluated. If dims is None, SumEvaluator will evaluate
                  four dimensions: coherence, consistency, fluency, relevance.

            overall: indicates whether the overall score is to be calculated.
                     Overall score can be customized to a combination of scores based on different
                     dimensions. The default here is the average score of all the given dimensions.
                     
            print_result: whether to print the average score of each dimension on the screen
        """
        n_data = len(data)
        eval_scores = [{} for _ in range(n_data)]

        if dims == None:
            eval_dims = self.dimensions
        else:
            assert isinstance(dims, list)
            eval_dims = dims

        for dim in eval_dims:
            print('Evaluating {} of {} samples !!!'.format(dim, n_data))

            # Calculate average sentence-level scores for 'consistency' and 'fluency'
            if dim == 'consistency' or dim == 'fluency':
                src_list, output_list = [], []
                n_sents = [] # the number of sentences in each generated summary
                for i in range(n_data):
                    if dim == 'consistency':
                        source = data[i]['source']
                    else:
                        source = ''
                    system_outputs = sent_tokenize(data[i]['system_output'])
                    n_sents.append(len(system_outputs))
                    for j in range(len(system_outputs)):
                        src_list.append(source)
                        output_list.append(system_outputs[j])
                input_list = add_question(dimension=dim, output=output_list, 
                                          src=src_list, task=self.task)
                sent_score = self.scorer.score(input_list)
                
                # Get average score for each sample
                start_idx = 0
                score = []
                for cur_n_sent in n_sents:
                    score.append(sum(sent_score[start_idx: start_idx + cur_n_sent]) / cur_n_sent)
                    start_idx += cur_n_sent
            
            # Calculate summary-level score for 'coherence' and 'relevance'
            elif dim == 'coherence' or dim == 'relevance':
                src_list, output_list, ref_list = [], [], []
                for i in range(n_data):
                    src_list.append(data[i]['source'])
                    output_list.append(data[i]['system_output'])
                    if dim == 'relevance':
                        ref_list.append(data[i]['reference'])
                input_list = add_question(dimension=dim, output=output_list, 
                                          src=src_list, ref=ref_list, task=self.task)
                score = self.scorer.score(input_list)
            
            # Please customize other dimensions here for summarization
            else:
                raise NotImplementedError('The input format for this dimension is still undefined. \
                                           Please customize it first.')
            
            for i in range(n_data):
                eval_scores[i][dim] = score[i]

        # Customize your overall score here.
        if overall == True:
            for i in range(n_data):
                eval_scores[i]['overall'] = np.mean(list(eval_scores[i].values()))

        if print_result == True:
            print_scores(eval_scores)

        return eval_scores


class DialogEvaluator:
    def __init__(self, max_length=1024, device='cuda:0', cache_dir=None):
        """ Set up evaluator for dialogues """
        self.scorer = UniEvaluator(model_name_or_path='MingZhong/unieval-dialog', 
                                   max_length=max_length, 
                                   device=device, cache_dir=cache_dir)
        self.task = 'dialogue'
        self.dimensions = ['naturalness', 'coherence', 'engagingness', 
                           'groundedness', 'understandability']

    def evaluate(self, data, dims=None, overall=True, print_result=False):
        """
            Get the scores of all the given dimensions

            dims: A list of dimensions to be evaluated. If dims is None, DialogEvaluator will evaluate
                  five dimensions: naturalness, coherence, engagingness, groundedness and understandability.

            overall: indicates whether the overall score is to be calculated.
                     Overall score can be customized to a combination of scores based on different
                     dimensions. The default here is the average score of all the given dimensions.

            print_result: whether to print the average score of each dimension on the screen
        """
        n_data = len(data)
        eval_scores = [{} for _ in range(n_data)]

        if dims == None:
            eval_dims = self.dimensions
        else:
            assert isinstance(dims, list)
            eval_dims = dims

        for dim in eval_dims:
            print('Evaluating {} of {} samples !!!'.format(dim, n_data))

            # Calculate summation score for 'engagingness'
            if dim == 'engagingness':
                src_list, output_list, context_list = [], [], []
                n_sents = [] # the number of sentences in each generated response
                for i in range(n_data):
                    source = data[i]['source']
                    context = data[i]['context']
                    system_outputs = sent_tokenize(data[i]['system_output'])
                    n_sents.append(len(system_outputs))
                    for j in range(len(system_outputs)):
                        src_list.append(source)
                        context_list.append(context)
                        output_list.append(system_outputs[j])
                input_list = add_question(dimension=dim, output=output_list, 
                                          src=src_list, context=context_list, task=self.task)
                sent_score = self.scorer.score(input_list)
                
                # Get the summation score for each sample
                start_idx = 0
                score = []
                for cur_n_sent in n_sents:
                    score.append(sum(sent_score[start_idx: start_idx + cur_n_sent]))
                    start_idx += cur_n_sent
            
            # Calculate turn-level score for other dimensions
            elif dim in ['naturalness', 'coherence', 'groundedness', 'understandability']:
                src_list, output_list, context_list = [], [], []
                for i in range(n_data):
                    if dim == 'coherence':
                        src_list.append(data[i]['source'])
                    else:
                        src_list.append('')
                    output_list.append(data[i]['system_output'])
                    if dim == 'groundedness':
                        context_list.append(data[i]['context'])
                    else:
                        context_list.append('')
                input_list = add_question(dimension=dim, output=output_list, 
                                          src=src_list, context=context_list, task=self.task)
                score = self.scorer.score(input_list)

            # Please customize other dimensions here for summarization
            else:
                raise NotImplementedError('The input format for this dimension is still undefined. \
                                           Please customize it first.')
            
            for i in range(n_data):
                eval_scores[i][dim] = score[i]

        # Customize your overall score here.
        if overall == True:
            for i in range(n_data):
                eval_scores[i]['overall'] = np.mean(list(eval_scores[i].values()))

        if print_result == True:
            print_scores(eval_scores)

        return eval_scores


class D2tEvaluator:
    def __init__(self, max_length=1024, device='cuda:0', cache_dir=None):
        """ Set up evaluator for data-to-text """
        self.scorer = UniEvaluator(model_name_or_path='MingZhong/unieval-sum', 
                                   max_length=max_length, 
                                   device=device, cache_dir=cache_dir)
        self.task = 'data2text'
        self.dimensions = ['naturalness', 'informativeness']

    def evaluate(self, data, dims=None, overall=True, print_result=False):
        """
            Get the scores of all the given dimensions

            dims: A list of dimensions to be evaluated. If dims is None, D2tEvaluator will evaluate
                  two dimensions: naturalness and informativeness.

            overall: indicates whether the overall score is to be calculated.
                     Overall score can be customized to a combination of scores based on different
                     dimensions. The default here is the average score of all the given dimensions.
                     
            print_result: whether to print the average score of each dimension on the screen
        """
        n_data = len(data)
        eval_scores = [{} for _ in range(n_data)]

        if dims == None:
            eval_dims = self.dimensions
        else:
            assert isinstance(dims, list)
            eval_dims = dims

        for dim in eval_dims:
            print('Evaluating {} of {} samples !!!'.format(dim, n_data))

            output_list, ref_list = [], []
            for i in range(n_data):
                output_list.append(data[i]['system_output'])
                ref_list.append(data[i]['reference'])

            input_list = add_question(dimension=dim, output=output_list, 
                                      ref=ref_list, task=self.task)
            score = self.scorer.score(input_list)

            for i in range(n_data):
                eval_scores[i][dim] = score[i]

        # Customize your overall score here.
        if overall == True:
            for i in range(n_data):
                eval_scores[i]['overall'] = np.mean(list(eval_scores[i].values()))

        if print_result == True:
            print_scores(eval_scores)

        return eval_scores


class FactEvaluator:
    def __init__(self, max_length=1024, device='cuda:0', cache_dir=None):
        """ Set up evaluator for factual consistency detection """
        self.scorer = UniEvaluator(model_name_or_path='MingZhong/unieval-fact', 
                                   max_length=max_length, 
                                   device=device, cache_dir=cache_dir)
        self.task = 'fact'
        self.dim = 'consistency'
    
    def evaluate(self, data, print_result=False):
        """
            Get the factual consistency score (only 1 dimension for this task)
   
            print_result: whether to print the average factual score on the screen
        """
        n_data = len(data)
        eval_scores = [{} for _ in range(n_data)]

        print('Evaluating {} of {} samples !!!'.format(self.dim, n_data))

        # Calculate average sentence-level scores for facutal consistency
        src_list, output_list = [], []
        n_sents = [] # the number of sentences in the claim
        for i in range(n_data):
            source = data[i]['source']
            system_outputs = sent_tokenize(data[i]['system_output'])
            n_sents.append(len(system_outputs))
            for j in range(len(system_outputs)):
                src_list.append(source)
                output_list.append(system_outputs[j])
        input_list = add_question(dimension=self.dim, output=output_list, 
                                  src=src_list, task=self.task)
        sent_score = self.scorer.score(input_list)
        
        # Get average score for each sample
        start_idx = 0
        score = []
        for cur_n_sent in n_sents:
            score.append(sum(sent_score[start_idx: start_idx + cur_n_sent]) / cur_n_sent)
            start_idx += cur_n_sent
           
        for i in range(n_data):
            eval_scores[i][self.dim] = score[i]

        if print_result == True:
            print_scores(eval_scores)

        return eval_scores

def get_evaluator(task, max_length=1024, device='cuda:0', cache_dir=None):
    assert task in ['summarization', 'dialogue', 'data2text', 'fact']
    if task == 'summarization':
        return SumEvaluator(max_length=max_length,
                            device=device,
                            cache_dir=cache_dir)
    elif task == 'dialogue':
        return DialogEvaluator(max_length=max_length,
                               device=device,
                               cache_dir=cache_dir)
    elif task == 'data2text':
        return D2tEvaluator(max_length=max_length,
                            device=device,
                            cache_dir=cache_dir)
    elif task == 'fact':
        return FactEvaluator(max_length=max_length,
                             device=device,
                             cache_dir=cache_dir)
    else:
        raise NotImplementedError('Other tasks are not implemented, \
                                   please customize specific tasks here.')
    
    
def convert_to_json(output_list, src_list=None, ref_list=None, context_list=None, \
            scores=None, doc_id=None, system_id=None):
    """
        Convert the data into the json format.

        output_list: a list of model output
        src_list: source input for different NLG tasks. For example, source document for summarization 
                  and dialogue history for dialogue response generation
        ref_list: human-annotated groundtruth
        context_list: the context needed to evaluate several specific dimension. For example,
                      additional factual information when evaluating engagingness and groundedness in dialogues
        scores: human scores for evaluating the model output. They can be used to calculate the correlation
                between evaluators and human judgements. The scores should be stored in a dictionary. For example,
                {'fluency': 2.0, 'coherence': 3.0} could be the human score for a sample.
        doc_id: the index of the input source. It can be used to calculate summary-level correlation for summarzation
        system_id: the index of the generation system. It can be used to calculate system-level correlation.
    """
    json_data = []
    for i in range(len(output_list)):
        cur = {}
        cur['system_output'] = output_list[i]
        if src_list is not None:
            cur['source'] = src_list[i]
        if ref_list is not None:
            cur['reference'] = ref_list[i]
        if context_list is not None:
            cur['context'] = context_list[i]
        if scores is not None:
            cur['scores'] = scores[i]
        if doc_id is not None:
            cur['doc_id'] = doc_id[i]
        if system_id is not None:
            cur['system_id'] = system_id[i]
        json_data.append(cur)
    return json_data
def add_question(dimension, output, src=None, ref=None, context=None, task=None):
    """
        Add questions to generate input in Bool-QA format for UniEval.
        
        dimension: specific dimension to be evaluated
        src: source input for different NLG tasks. For example, source document for summarization 
             and dialogue history for dialogue response generation.
        output: output text generated by the models
        ref: human-annotataed groundtruth
        context: the context needed to evaluate several specific dimension. For example,
                 additional factual information when evaluating engagingness and groundedness in dialogues.
    """
    
    input_with_question = []
    for i in range(len(output)):
        # For summarization
        if task == 'summarization':
            if dimension == 'fluency':
                cur_input = 'question: Is this a fluent paragraph? </s> paragraph: ' + output[i]
            elif dimension == 'coherence':
                cur_input = 'question: Is this a coherent summary to the document? </s> summary: ' + output[i] + ' </s> document: ' + src[i]
            elif dimension == 'consistency':
                cur_input = 'question: Is this claim consistent with the document? </s> claim: ' + output[i] + ' </s> document: ' + src[i]
            elif dimension == 'relevance':
                cur_input = 'question: Is this summary relevant to the reference? </s> summary: ' + output[i] + ' </s> reference: ' + ref[i]
            else:
                raise NotImplementedError('The input format for this dimension is still undefined. Please customize it first.')
        # For dialogues
        elif task == 'dialogue':
            if dimension == 'naturalness':
                cur_input = 'question: Is this a natural response in the dialogue? </s> response: ' + output[i]
            elif dimension == 'coherence':
                cur_input = 'question: Is this a coherent response given the dialogue history? </s> response: '\
                            + output[i] + ' </s> dialogue history: ' + src[i]
            elif dimension == 'engagingness':
                cur_input = 'question: Is this an engaging and informative response according to the dialogue history and fact? </s> response: '\
                            + output[i] + ' </s> dialogue history: ' + src[i] + ' </s> fact: ' + context[i]
            elif dimension == 'groundedness':
                cur_input = 'question: Is this response consistent with knowledge in the fact? </s> response: '\
                            + output[i] + ' </s> fact: ' + context[i]
            elif dimension == 'understandability':
                cur_input = 'question: Is this an understandable response in the dialogue? </s> response: ' + output[i]
            else:
                raise NotImplementedError('The input format for this dimension is still undefined. Please customize it first.')
        # For data-to-text
        elif task == 'data2text':
            if dimension == 'naturalness':
                cur_input = 'question: Is this a fluent utterance? </s> utterance: ' + output[i]
            elif dimension == 'informativeness':
                cur_input = 'question: Is this sentence informative according to the reference? </s> sentence: '\
                            + output[i] + ' </s> reference: ' + ref[i]
            else:
                raise NotImplementedError('The input format for this dimension is still undefined. Please customize it first.')
        # For factual consistency detection
        elif task == 'fact':
            if dimension == 'consistency':
                cur_input = 'question: Is this claim consistent with the document? </s> claim: ' + output[i] + ' </s> document: ' + src[i]
            else:
                raise NotImplementedError('No other dimensions for the factual consistency detection task.')
        # For new customized tasks
        else:
            raise NotImplementedError('Other tasks are not implemented, please customize specific tasks here.')
        input_with_question.append(cur_input)
    return input_with_question


def print_scores(scores):
    table = PrettyTable(['Dimensions','Score'])
    print('\nEvaluation scores are shown below:')
    dims = list(scores[0].keys())
    for dim in dims:
        cur_score = 0
        for i in range(len(scores)):
            cur_score += scores[i][dim]
        table.add_row([dim, round(cur_score / len(scores), 6)])
    print(table)
    

def parse_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    records = {}
    for line in lines:
        line = line.strip()
        try:
            if line.startswith("T-"):
                record_id = line.split("\t")[0].replace("T-","")
                src_text = line.split("\t")[1]
                if record_id not in records:
                    records[record_id] = {}
                records[record_id]['src'] = src_text
            elif line.startswith("H-"):
                record_id = line.split("\t")[0].replace("H-","")
                hyp_text = line.split("\t")[2]
                if record_id not in records:
                    records[record_id] = {}
                records[record_id]['hyp'] = hyp_text
            elif line.startswith("D-"):
                record_id = line.split("\t")[0].replace("D-","")
                det_text = line.split("\t")[2]
                if record_id not in records:
                    records[record_id] = {}
                records[record_id]['det'] = det_text
            elif line.startswith("P-"):
            # We can ignore P- lines for now as they contain scores not needed for evaluation
                continue
        except:
            print(line)

    return records

langs = #ISO Language codes ['as','bn','gu','hi','kn','mr','ml','mni','ta','te','or','pa','sa','sd','ur']
for i in langs:
    file_path = #file path
    data_records = parse_file(file_path)

    data_records = {k: v for k, v in data_records.items() if 'src' in v and 'hyp' in v and 'det' in v}

# Ensure that at least one record remains after filtering
    if not data_records:
        raise ValueError("No valid records found in the file.")
    
    task = 'summarization'  # or 'dialogue' or 'fact' based on your specific use-case   
    evaluator = get_evaluator(task)

    output_list = [record['det'] for record in data_records.values()]
    src_list = [record['src'] for record in data_records.values()]
# If relevance is to be evaluated, you need ref_list
    ref_list = [record['hyp'] for record in data_records.values()]

# Prepare data for pre-trained evaluators
    data = convert_to_json(output_list=output_list, src_list=src_list, ref_list=ref_list)
    print(i)
# Get multi-dimensional evaluation scores
    eval_scores = evaluator.evaluate(data, print_result=True)