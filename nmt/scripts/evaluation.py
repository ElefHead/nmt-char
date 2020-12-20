import torch
import numpy as np

from nmt.datasets import batch_iter, read_corpus
from nmt.models import NMT
from typing import Tuple, List
from .decoding import Hypothesis, beam_search

import sys
from argparse import Namespace

from nltk.translate.bleu_score import corpus_bleu


def evaluate_ppl(model: NMT,
                 dev_data: Tuple[List[List[str]], List[List[str]]],
                 batch_size: int = 32) -> float:
    """ Evaluate perplexity on dev sentences
    @param model (NMT): NMT Model
    @param dev_data (list of (src_sent, tgt_sent)): list of tuples
        containing source and target sentence
    @param batch_size (batch size)
    @returns ppl (perplixty on dev sentences)
    """
    was_training = model.training
    model.eval()

    cum_loss = 0.
    cum_tgt_words = 0.

    # no_grad() signals backend to throw away all gradients
    with torch.no_grad():
        for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
            loss = -model(src_sents, tgt_sents).sum()

            cum_loss += loss.item()
            tgt_word_num_to_predict = sum(
                len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)

    if was_training:
        model.train()

    return ppl


def compute_corpus_level_bleu_score(
        references: List[List[str]],
        hypotheses: List[Hypothesis]) -> float:
    """ Given decoding results and reference sentences,
        compute corpus-level BLEU score.
    @param references (List[List[str]]): a list of gold-standard
        reference target sentences
    @param hypotheses (List[Hypothesis]): a list of hypotheses,
        one for each reference
    @returns bleu_score: corpus-level BLEU score
    """
    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]
    bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp.value for hyp in hypotheses])
    return bleu_score


def evaluate(args: Namespace):
    """
    Performs decoding on a test set, and save
        the best-scoring decoding results.
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    @param args (Dict): args from cmd line
    """

    print("load test source sentences from [{}]".format(
        args.test_src), file=sys.stderr)
    test_data_src = read_corpus(args.test_src, is_target=False)
    test_data_tgt = []
    if args.test_tgt:
        print("load test target sentences from [{}]".format(
            args.test_tgt), file=sys.stderr)
        test_data_tgt = read_corpus(args.test_tgt, is_target=True)

    print("load model from {}".format(args.model_path), file=sys.stderr)
    model = NMT.load(args.model_path)

    if args.cuda:
        model = model.cuda()

    hypotheses = beam_search(
        model, test_data_src,
        beam_size=int(args.beam_size),
        max_decoding_time_step=int(args.max_decoding_time_step)
    )

    if args.test_tgt:
        top_hypotheses = [hyps[0] for hyps in hypotheses]
        bleu_score = compute_corpus_level_bleu_score(
            test_data_tgt,
            top_hypotheses
        )
        print('Corpus BLEU: {}'.format(bleu_score * 100), file=sys.stderr)

    with open(args.output_path, 'w') as f:
        for src_sent, hyps in zip(test_data_src, hypotheses):
            top_hyp = hyps[0]
            hyp_sent = ' '.join(top_hyp.value)
            f.write(hyp_sent + '\n')
