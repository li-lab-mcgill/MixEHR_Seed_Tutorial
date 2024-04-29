import logging
import argparse
import torch
from MixEHR_Seed import MixEHR_Seed
from corpus import Corpus

logger = logging.getLogger("MixEHR-Seed training processing")
parser = argparse.ArgumentParser()
# default arguments
parser.add_argument('corpus', help='Path to read corpus file', default='./store/')
parser.add_argument('output', help='Directory to store model', default='./parameters/')
parser.add_argument("-epoch", "--max_epoch", help="Maximum number of max_epochs", type=int, default=10)
parser.add_argument("-batch_size", "--batch_size", help="Batch size of a minibatch", type=int, default=500)
parser.add_argument("-every", "--save_every", help="Store model every X number of iterations", type=int, default=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # we use GPU, printed result is "cuda"
print(device)

def run(args):
    # print(args)
    # cmd = args.cmd
    seeds_topic_matrix = torch.load("./phecode_mapping/seed_topic_matrix.pt", map_location=device) # get seed word-topic mapping, V x K matrix
    print("V and K are", seeds_topic_matrix.shape) # torch.Size([V, K])
    corpus = Corpus.read_corpus_from_directory(args.corpus)
    print("trained modalities include", corpus.modalities)
    model = MixEHR_Seed(corpus, seeds_topic_matrix, corpus.modalities, guided_modality=0, stochastic_VI=False, batch_size=args.batch_size, out=args.output)
    model = model.to(device)
    logger.info('''
    #     ======= Parameters =======
    #     mode: \t\ttraining
    #     file:\t\t%s
    #     output:\t\t%s
    #     max iterations:\t%s
    #     batch size:\t%s
    #     save every:\t\t%s
    #     ==========================
    # ''' % (args.corpus, args.output, args.max_epoch, args.batch_size, args.save_every))
    elbo = model.inference(max_epoch=args.max_epoch)
    # elbo = model.inference_SCVB_SGD(max_epoch=args.max_epoch, save_every=args.save_every)
    print("elbo : %s" % (elbo))
    print('elbo: ', model.elbo)
    print('pz: ', model.term1)
    print('qz: ', model.term2)
    print('pw: ', model.term3)

    # Open files for writing
    # with open('elbo1.txt', 'a') as file_elbo1, \
    #      open('elbo2.txt', 'a') as file_elbo2, \
    #      open('pz.txt', 'a') as file_pz, \
    #      open('qz.txt', 'a') as file_qz, \
    #      open('pw.txt', 'a') as file_pw:
    #
    #     # Convert lists to strings and write to files
    #     file_elbo1.write(str(elbo) + '\n')
    #     file_elbo2.write(str(model.elbo) + '\n')
    #     file_pz.write(str(model.term1) + '\n')
    #     file_qz.write(str(model.term2) + '\n')
    #     file_pw.write(str(model.term3) + '\n')

if __name__ == '__main__':
    run(parser.parse_args(['./store/', './parameters/']))
