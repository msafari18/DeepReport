from DeepReport_model.utils import join_words
import torch
import torch.nn.functional as F

def eval_transformer(decoder, device, vocab, feats, max_step=50, beam_size=3):
    decoder.eval()
    word_map = vocab.word2idx
    idx_map = vocab.idx2word
    vocab_size = len(word_map)
    # f = open(log, 'w+')

    with torch.no_grad():
        k = beam_size
        feats = torch.tensor(feats)

        encoder_out = torch.permute(feats, (0, 2, 3, 1)).to(device)
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(-1)
        encoder_out = encoder_out.expand(
            k, enc_image_size, enc_image_size, encoder_dim)
        k_prev_words = torch.LongTensor(
            [[word_map['<start>']] * max_step] * k).to(device)

        seqs = torch.LongTensor([[word_map['<start>']]] * k).to(device)
        top_k_scores = torch.zeros(k, 1).to(device)
        complete_seqs = []
        complete_seqs_scores = []
        hypotheses = []
        references = []

        step = 1
        while True:
            cap_len = torch.LongTensor([max_step]).repeat(k, 1)
            # print(encoder_out.shape, k_prev_words.shape, cap_len)
            scores, _, _, _, _ = decoder(
                encoder_out, k_prev_words, cap_len)
            scores = scores[:, step-1, :].squeeze(1)
            scores = F.log_softmax(scores, dim=1)
            scores = top_k_scores.expand_as(scores) + scores

            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(
                    k, 0, True, True)
            else:
                top_k_scores, top_k_words = scores.view(
                    -1).topk(k, 0, True, True)

            prev_word_inds = top_k_words // vocab_size
            next_word_inds = top_k_words % vocab_size
            seqs = torch.cat(
                [seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)

            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['<end>']]
            complete_inds = list(
                set(range(len(next_word_inds))) - set(incomplete_inds))

            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])

            k -= len(complete_inds)
            if k == 0:
                break

            seqs = seqs[incomplete_inds]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = k_prev_words[incomplete_inds]
            k_prev_words[:, :step+1] = seqs

            if step > max_step - 2:
                break
            step += 1
            if step % 10 == 0:
                print(step)
            # if len(complete_seqs_scores) == 0:
            #     continue

        indices = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[indices]

        hypotheses.append(join_words([idx_map[w] for w in seq if w not in {
            word_map['<start>'], word_map['<end>'], word_map['<pad>']}]))
        print(hypotheses[0])
        # f.write('Prediction\n')
        # f.write(hypotheses[0] + '\n')

    # f.close()
    return hypotheses[0] + '\n'
