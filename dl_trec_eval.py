import os
import re
import sys
import subprocess
from collections import OrderedDict


parent_path = 'path_to_trec_eval'
trec_eval_script_path = os.path.join(parent_path, 'trec_eval.9.0/trec_eval')


def run(command, get_ouput=False):
  try:
    if get_ouput:
      process = subprocess.Popen(command, stdout=subprocess.PIPE)
      output, err = process.communicate()
      exit_code = process.wait()
      return output
    else:
      subprocess.call(command)
  except subprocess.CalledProcessError as e:
    print(e)


def evaluate_trec(qrels, res, metrics):

  ''' all_trecs, '''
  command = [trec_eval_script_path, '-m', 'all_trec', '-M', '1000', qrels, res]
  output = run(command, get_ouput=True)
  output = str(output, encoding='utf-8')

  metrics_val = []
  for metric in metrics:
    metrics_val.append(re.findall(r'{0}\s+all.+\d+'.format(metric), output)[0].split('\t')[2].strip())

  return OrderedDict(zip(metrics, metrics_val))


def evaluate_metrics(qrels, res, metrics=None):
  normal_metrics = [met for met in metrics if not met.startswith('i')]
  metrics_val_dict = OrderedDict()
  if len(normal_metrics) > 0:
    metrics_val_dict.update(evaluate_trec(qrels, res, metrics=normal_metrics))

  return metrics_val_dict


if __name__ == '__main__':
  """Command line:
      python dl_trec_eval.py <path to reference> <path_to_candidate_file>
  """
  argv = sys.argv
  res1, res2 = argv[1], argv[2]
  print(evaluate_metrics(argv[1], argv[2],
                         ['recip_rank', 'ndcg_cut_10', 'map', 'recall_100', 'recall_500', 'recall_1000']))
