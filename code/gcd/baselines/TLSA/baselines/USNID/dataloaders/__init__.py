from .bert_loader import BERT_Loader
max_seq_lengths = {'clinc':30, 'stackoverflow':45, 'trec':45, 'banking':55,'hwu':55,'mcid': 65, 'ecdt': 65, 'thucnews': 256}

backbone_loader_map = {
                            'bert_USNID': BERT_Loader,
                      }
