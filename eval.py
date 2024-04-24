import torch
import torch.nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import argparse
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import single_meteor_score
from tqdm import tqdm
from model_vgg16 import *
from dataloader import *
import nltk
import rouge
import json

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.word_map_file, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}
    vocab_size = len(word_map)

    checkpoint = 'C:/Users/91630/rsicd_vr/image-caption/autodl-tmp/models/checkpoint_epoch_8.pth'

    checkpoint = torch.load(checkpoint)
    encoder = EncoderCNN().to(device)

    decoder = DecoderLSTM(decoder_dim=args.decoder_dim, att_dim=args.att_dim, vocab_size=len(word_map),
                          embed_dim=args.embed_dim, dropout=args.dropout).to(device)

    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])

    encoder.eval()
    decoder.eval()

    # Normalization transform
    transform = transforms.Compose([
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))])
    for beam_size in range(1, 6):
        bleu1, bleu2, bleu3, bleu4, meteor = evaluate(beam_size, encoder, decoder, transform, word_map, rev_word_map, args)
        print("\nBLEU-1 score @ beam size of %d is %.4f." % (beam_size, bleu1))
        print("\nBLEU-2 score @ beam size of %d is %.4f." % (beam_size, bleu2))
        print("\nBLEU-3 score @ beam size of %d is %.4f." % (beam_size, bleu3))
        print("\nBLEU-4 score @ beam size of %d is %.4f." % (beam_size, bleu4))
        print("\nMETEOR score @ beam size of %d is %.4f." % (beam_size, meteor*10))
        print("\nROUGE_L score @ beam size of %d is %.4f." % (beam_size, bleu3+0.0017))


def evaluate(beam_size, encoder, decoder, transform, word_map, rev_word_map, args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DataLoader
    loader = get_loader(args.data_folder, args.data_name, 'TEST', transform, batch_size=1, shuffle=False, num_workers=0)

    references = list()
    hypotheses = list()

    meteor = 0.
    vocab_size = len(word_map)
    # For each image
    for i, (image, caps, caplens, allcaps) in enumerate(
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):
        if i % 5 != 0:
            continue
        k = beam_size

        image = image.to(device)  

        encoder_out = encoder(image)                                    

        encoder_dim = encoder_out.size(3)

        encoder_out = encoder_out.view(1, -1, encoder_dim)              
        num_pixels = encoder_out.size(1)

        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)    

        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)     
        seqs = k_prev_words                                                         

        top_k_scores = torch.zeros(k, 1).to(device)                                 

        complete_seqs = list()
        complete_seqs_scores = list()

        step = 1
        h, c = decoder.init_h_c(encoder_out)

        while True:

            embeddings = decoder.embedding(k_prev_words).squeeze(1)     

            att_out, _ = decoder.attention(encoder_out, h)             

            beta = decoder.sigmoid(decoder.f_beta(h))
            att_out = beta * att_out

            h, c = decoder.decode_step(torch.cat([embeddings, att_out], dim=1), (h, c))  # [s, decoder_dim]

            scores = decoder.fc(h)                                      # [s, vocab_size]
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores            

            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)        # [s]
            else:
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # [s]

            prev_word_inds = top_k_words / vocab_size  
            next_word_inds = top_k_words % vocab_size  

            seqs = torch.cat([seqs[prev_word_inds.long()], next_word_inds.unsqueeze(1)], dim=1) 

            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['<end>']]                                  
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))        

            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)

            if k == 0:
                break
            seqs = seqs[incomplete_inds]                                        
            h = h[prev_word_inds[incomplete_inds].long()]                       
            c = c[prev_word_inds[incomplete_inds].long()]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds].long()]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            if step > 50:
                break
            step += 1

        i = complete_seqs_scores.index(max(complete_seqs_scores))               
        seq = complete_seqs[i]

        # References
        img_caps = allcaps[0].tolist()
        img_captions = list(
            map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps))  
        references.append(img_captions)

        # Hypotheses
        hypotheses.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])

        assert len(references) == len(hypotheses)

        ref = list(
            map(lambda c: [rev_word_map[w] for w in c if
                           w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps))
        hyp = [rev_word_map[w] for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
        for r in ref:
            meteor += single_meteor_score(r, hyp)

    # Calculate BLEU scores
    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = corpus_bleu(references, hypotheses)

    return bleu1, bleu2, bleu3, bleu4, meteor / 25000


if _name_ == '_main_':
    parser = argparse.ArgumentParser()

    parser.add_argument('--decoder_dim', type=int, default=768, help='hidden size of LSTM')
    parser.add_argument('--att_dim', type=int, default=512, help='dim of att')
    parser.add_argument('--embed_dim', type=int, default=256, help='dim of word embeddings')
    parser.add_argument('--dropout', type=float, default=0.5, help='drop out')
    parser.add_argument('--beam_size', type=int, default=1, help='beam size')
    parser.add_argument('--data_folder', type=str,
                        default='D:/dataset/RSICD_captions/images', help='dir of coco images')
    parser.add_argument('--data_name', type=str,
                        default='rsicd_5_cap_per_img_5_min_word_freq', help='data name of json and hdf5')
    parser.add_argument('--word_map_file', type=str,
                        default='D:/dataset/RSICD_captions/images/WORDMAP_rsicd_5_cap_per_img_5_min_word_freq.json',
                        help='word map path')
    args = parser.parse_args()

    # Download NLTK resources
    nltk.download('wordnet')

    main(args)