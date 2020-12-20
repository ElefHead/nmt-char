import torch
from tqdm import tqdm
from nmt.models import NMT
from typing import List
from collections import namedtuple
import sys

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


def beam_search(
     model: NMT,
     test_data_src: List[List[str]],
     beam_size: int,
     max_decoding_time_step: int) -> List[List[Hypothesis]]:
    """ Run beam search to construct hypotheses for a list
        of src-language sentences.
    @param model (NMT): NMT Model
    @param test_data_src (List[List[str]]): List of sentences (words)
        in source language, from test set.
    @param beam_size (int): beam_size (# of hypotheses to hold for
        a translation at every step)
    @param max_decoding_time_step (int): maximum sentence
        length that Beam search can produce
    @returns hypotheses (List[List[Hypothesis]]): List of Hypothesis
        translations for every source sentence.
    """
    was_training = model.training
    model.eval()

    hypotheses = []
    with torch.no_grad():
        for src_sent in tqdm(
            test_data_src,
            desc='Decoding',
            file=sys.stdout
        ):
            example_hyps = beam_search_decoder(
                model,
                src_sent,
                beam_size=beam_size,
                max_decoding_time_step=max_decoding_time_step
            )

            hypotheses.append(example_hyps)

    if was_training:
        model.train(was_training)

    return hypotheses


def beam_search_decoder(
    model: NMT,
    src_sent: List[str],
    beam_size: int = 5,
    max_decoding_time_step: int = 70
) -> List[str]:
    src_char_tensor = model.vocab.src.to_tensor(
        [src_sent],
        tokens=False,
        device=model.device
    )
    src_encode, dec_state = model.encoder(
        src_char_tensor,
        [len(src_sent)]
    )
    hypotheses = [['<s>']]
    completed_hypotheses = []
    hyp_scores = torch.zeros(
        len(hypotheses), dtype=torch.float,
        device=model.device
    )
    o_prev = torch.zeros(1, model.hidden_size, device=model.device)
    t = 0
    while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
        t += 1

        num_hyp = len(hypotheses)

        exp_src_encodings = src_encode.expand(
            num_hyp,
            src_encode.size(1),
            src_encode.size(2)
        )

        y_t = model.vocab.tgt.to_tensor(
            list([hyp[-1]] for hyp in hypotheses),
            tokens=False, device=model.device
        )
        out, dec_state, _ = model.decoder(
            y_t,
            exp_src_encodings,
            dec_state,
            o_prev,
            None
        )
        log_p_t = model.generator(out)
        live_hyp_num = beam_size - len(completed_hypotheses)

        continuing_hyp_scores = (hyp_scores.unsqueeze(
            1).expand_as(log_p_t) + log_p_t).view(-1)

        top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(
            continuing_hyp_scores, k=live_hyp_num)

        prev_hyp_ids = top_cand_hyp_pos // len(model.vocab.tgt)
        hyp_word_ids = top_cand_hyp_pos % len(model.vocab.tgt)

        new_hypotheses = []
        live_hyp_ids = []
        new_hyp_scores = []

        decoderStatesForUNKsHere = []

        for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(
                prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
            prev_hyp_id = prev_hyp_id.item()
            hyp_word_id = hyp_word_id.item()
            cand_new_hyp_score = cand_new_hyp_score.item()

            hyp_word = model.vocab.tgt.to_tokens(hyp_word_id)

            # Record output layer in case UNK was generated
            if hyp_word == "<unk>":
                hyp_word = "<unk>"+str(len(decoderStatesForUNKsHere))
                decoderStatesForUNKsHere.append(o_prev[prev_hyp_id])

            new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
            if hyp_word == '</s>':
                completed_hypotheses.append(
                    Hypothesis(
                        value=new_hyp_sent[1:-1],
                        score=cand_new_hyp_score
                    )
                )
            else:
                new_hypotheses.append(new_hyp_sent)
                live_hyp_ids.append(prev_hyp_id)
                new_hyp_scores.append(cand_new_hyp_score)

        if len(decoderStatesForUNKsHere) > 0 and \
                model.char_decoder is not None:  # decode UNKs
            decoderStatesForUNKsHere = torch.stack(
                decoderStatesForUNKsHere, dim=0)
            decodedWords = model.greedy_char_decode(
                (
                    decoderStatesForUNKsHere.unsqueeze(0),
                    decoderStatesForUNKsHere.unsqueeze(0)
                ), max_length=21)
            assert len(decodedWords) == decoderStatesForUNKsHere.size()[
                0], "Incorrect number of decoded words"
            for hyp in new_hypotheses:
                if hyp[-1].startswith("<unk>"):
                    hyp[-1] = decodedWords[int(hyp[-1][5:])]

        if len(completed_hypotheses) == beam_size:
            break

        live_hyp_ids = torch.tensor(
            live_hyp_ids,
            dtype=torch.long,
            device=model.device
        )

        o_prev = out[live_hyp_ids]
        hidden, cell = dec_state
        dec_state = (
            hidden[live_hyp_ids],
            cell[live_hyp_ids]
        )

        hypotheses = new_hypotheses
        hyp_scores = torch.tensor(
            new_hyp_scores, dtype=torch.float, device=model.device)

    if len(completed_hypotheses) == 0:
        completed_hypotheses.append(
            Hypothesis(
                value=hypotheses[0][1:],
                score=hyp_scores[0].item()
            )
        )

    completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

    return completed_hypotheses
