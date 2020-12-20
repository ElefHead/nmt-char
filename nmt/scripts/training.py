from argparse import Namespace
from nmt.datasets import read_corpus, batch_iter, Vocab
from nmt.models import NMT
from .evaluation import evaluate_ppl
import sys
import numpy as np
import torch
import time


def train(args: Namespace):
    train_data_src = read_corpus(args.train_src, is_target=False)
    train_data_tgt = read_corpus(args.train_tgt, is_target=True)

    dev_data_src = read_corpus(args.dev_src, is_target=False)
    dev_data_tgt = read_corpus(args.dev_tgt, is_target=False)

    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))

    train_batch_size = args.batch_size

    clip_grad = float(args.clip_grad)
    valid_niter = int(args.valid_niter)
    log_every = int(args.log_every)
    model_save_path = args.save_path

    vocab = Vocab.load(args.vocab)

    model = NMT(
        vocab=vocab,
        embedding_dim=args.embedding_size,
        hidden_size=args.hidden_size,
        use_char_decoder=args.use_chardecoder
    )

    model.train()

    uniform_init = float(args.uniform_init)
    if np.abs(uniform_init) > 0.:
        print('uniformly initialize parameters [-%f, +%f]' %
              (uniform_init, uniform_init), file=sys.stderr)
        for p in model.parameters():
            p.data.uniform_(-uniform_init, uniform_init)

    vocab_mask = torch.ones(len(vocab.tgt))
    vocab_mask[vocab.tgt['<pad>']] = 0

    if args.cuda:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cum_tgt_words\
        = report_tgt_words = 0
    cum_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training')

    for epoch in range(args.max_epoch):
        for src_sents, tgt_sents in batch_iter(
                train_data, batch_size=train_batch_size, shuffle=True):

            train_iter += 1

            optimizer.zero_grad()

            batch_size = len(src_sents)

            example_losses = -model(src_sents, tgt_sents)  # (batch_size,)
            batch_loss = example_losses.sum()
            loss = batch_loss / batch_size

            loss.backward()

            # clip gradient
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), clip_grad)

            optimizer.step()

            batch_losses_val = batch_loss.item()
            report_loss += batch_losses_val
            cum_loss += batch_losses_val

            tgt_words_num_to_predict = sum(
                len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            report_tgt_words += tgt_words_num_to_predict
            cum_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cum_examples += batch_size

            if train_iter % log_every == 0:
                print(
                    f"epoch {epoch}, "
                    f"iter {train_iter}, "
                    f"avg. loss {report_loss / report_examples}, "
                    f"avg. ppl {np.exp(report_loss / report_tgt_words)}, "
                    f"cum. examples {cum_examples}, "
                    f"speed {report_tgt_words/(time.time() - train_time)} "
                    "words/sec, "
                    f"time_elapsed {time.time() - begin_time} sec ",
                    file=sys.stderr
                )

                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            # perform validation
            if train_iter % valid_niter == 0:
                print(
                    f'epoch {epoch}, '
                    f"iter {train_iter}, "
                    f"cum. loss {cum_loss / cum_examples}, "
                    f"cum. ppl {np.exp(cum_loss / cum_tgt_words)} "
                    f"cum. examples {cum_examples}",
                    file=sys.stderr
                )

                cum_loss = cum_examples = cum_tgt_words = 0.
                valid_num += 1

                print('begin validation ...', file=sys.stderr)

                # compute dev. ppl and bleu
                # dev batch size can be a bit larger
                dev_ppl = evaluate_ppl(model, dev_data, batch_size=128)
                valid_metric = -dev_ppl

                print('validation: iter %d, dev. ppl %f' %
                      (train_iter, dev_ppl), file=sys.stderr)

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(
                    hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    patience = 0
                    print(
                        f'save currently the '
                        f'best model to [{model_save_path}]',
                        file=sys.stderr
                    )
                    model.save(model_save_path)

                    # also save the optimizers' state
                    torch.save(optimizer.state_dict(),
                               model_save_path + '.optim')

                elif patience < int(args.patience):
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)

                    if patience == int(args.patience):
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=sys.stderr)
                        if num_trial == int(args.max_num_trial):
                            print('early stop!', file=sys.stderr)
                            exit(0)

                        # decay lr, and restore from previously best checkpoint
                        lr = optimizer.param_groups[0]['lr'] * \
                            float(args.lr_decay)
                        print(
                            f'load previously best model and \
                            decay learning rate to{lr}',
                            file=sys.stderr
                        )

                        # load model
                        params = torch.load(
                            model_save_path,
                            map_location=lambda storage, loc: storage
                        )
                        model.load_state_dict(params['state_dict'])
                        if args.cuda:
                            model = model.cuda()

                        print('restore parameters of the optimizers',
                              file=sys.stderr)
                        optimizer.load_state_dict(
                            torch.load(model_save_path + '.optim'))

                        # set new lr
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                        # reset patience
                        patience = 0
