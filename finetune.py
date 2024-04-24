import os
import pickle
import torch.nn as nn
import torch
import time
import argparse
import nltk
import torch.optim as opt
import json

from torch.nn.utils.rnn import pack_padded_sequence
from model_vgg16 import EncoderCNN, DecoderLSTM
from torchvision import transforms
from dataloader import get_loader
from nltk.translate.bleu_score import corpus_bleu

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    word_map_file = os.path.join(args.data_folder, 'WORDMAP_' + args.data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    checkpoint = torch.load('C:/Users/91630/rsicd_vr/image-caption/autodl-tmp/models/checkpoint_epoch_7.pth')

    encoder = EncoderCNN().to(device)
    decoder = DecoderLSTM(decoder_dim=args.decoder_dim, att_dim=args.att_dim, vocab_size=len(word_map),
                          embed_dim=args.embed_dim, dropout=args.dropout).to(device)

    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])

    for param in encoder.parameters():
        param.requires_grad = True

    encoder_optimizer = opt.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()), lr=args.encoder_lr)
    decoder_optimizer = opt.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()), lr=args.decoder_lr)

    lr = args.encoder_lr
    criterion = nn.CrossEntropyLoss().to(device)

    transform = transforms.Compose([
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    train_loader = get_loader(data_folder=args.data_folder, data_name=args.data_name, split='TRAIN',
                              transform=transform,
                              batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = get_loader(data_folder=args.data_folder, data_name=args.data_name, split='VAL', transform=transform,
                            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    best_bleu4 = 0
    best_bleu1 = 0
    best_bleu2 = 0
    best_bleu3 = 0

    print('start training...')
    for epoch in range(args.epochs):
        if epoch > 0:
            lr = lr * 0.8
            encoder_optimizer = opt.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()), lr=lr)
            decoder_optimizer = opt.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()), lr=lr)
        train(train_loader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, epoch, args.print_step)
        bleu_1, bleu_2, bleu_3, bleu_4 = val(val_loader, encoder, decoder, criterion, args.print_step, word_map)
        is_best = bleu_1 > best_bleu1
        best_bleu4 = max(best_bleu4, bleu_4)
        best_bleu1 = max(best_bleu1, bleu_1)
        best_bleu2 = max(best_bleu2, bleu_2)
        best_bleu3 = max(best_bleu3, bleu_3)

        save_checkpoint(args.save_path, epoch, encoder, decoder, encoder_optimizer, decoder_optimizer, bleu_4, is_best,
                        bleu_1, bleu_2, bleu_3)


def train(train_loader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, epoch, print_step):
    encoder.train()
    decoder.train()
    start_time = time.time()
    total_loss = 0
    for i, (images, captions, lens) in enumerate(train_loader):
        images = images.to(device)
        captions = captions.to(device)
        lens = lens.to(device)

        images = encoder(images)
        predictions, alphas, caps_sorted, decode_lens, sort_ind = decoder(images, captions, lens)

        targets = caps_sorted[:, 1:]

        predictions = pack_padded_sequence(predictions, decode_lens, batch_first=True)[0]
        targets = pack_padded_sequence(targets, decode_lens, batch_first=True)[0]

        loss = criterion(predictions, targets)
        loss += args.lbd * ((1. - alphas.sum(dim=1)) ** 2).mean()  

        total_loss += loss.item()

        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()

        loss.backward()

        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        if i % print_step == 0:
            end_time = time.time()
            print('Epoch: {}, Step: {}/{}, Loss: {:.4f}, Time: {:.2f}s'.format(epoch, i, len(train_loader), loss.item(),
                                                                      end_time - start_time))
            start_time = time.time()

    print('Epoch: {}, Average Loss: {:.4f}'.format(epoch, total_loss / len(train_loader)))


def val(val_loader, encoder, decoder, criterion, print_step, word_map):
    decoder.eval()
    if encoder is not None:
        encoder.eval()

    references = {k: [] for k in range(1, 5)}
    hypotheses = {k: [] for k in range(1, 5)}

    start_time = time.time()
    with torch.no_grad():
        for i, (images, captions, lens, all_caps) in enumerate(val_loader):
            images = images.to(device)
            captions = captions.to(device)
            lens = lens.to(device)
            all_caps = all_caps.to(device)

            images = encoder(images)
            predictions, alphas, caps_sorted, decode_lens, sort_ind = decoder(images, captions, lens)

            targets = caps_sorted[:, 1:]

            predictions_copy = predictions.clone()
            predictions = pack_padded_sequence(predictions, decode_lens, batch_first=True)[0]
            targets = pack_padded_sequence(targets, decode_lens, batch_first=True)[0]

            loss = criterion(predictions, targets)
            loss += args.lbd * ((1. - alphas.sum(dim=1)) ** 2).mean()  

            if i % print_step == 0:
                end_time = time.time()
                print('Step: {}/{}, Loss: {:.4f}, Time: {:.2f}s'.format(i, len(val_loader), loss.item(),
                                                                         end_time - start_time))
                start_time = time.time()

            # References
            all_caps = all_caps[sort_ind]  # because images were sorted in the decoder
            for j in range(all_caps.shape[0]):
                img_caps = all_caps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                for k in range(1, 5):
                    references[k].append(img_captions)

            # Hypotheses
            _, preds = torch.max(predictions_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lens[j]])  # remove pads
            preds = temp_preds
            for k in range(1, 5):
                hypotheses[k].extend(preds)

    bleus = {}
    for n in range(1, 5):
        if len(references[n]) > 0 and len(hypotheses[n]) > 0:
            bleus[n] = corpus_bleu(references[n], hypotheses[n], weights=(1.0 / n,))
        else:
            bleus[n] = 0.0

    print('BLEU-1: {:.4f}, BLEU-2: {:.4f}, BLEU-3: {:.4f}, BLEU-4: {:.4f}'.format(bleus[4], bleus[3], bleus[2], bleus[1]))

    return bleus[4], bleus[3], bleus[2], bleus[1]


def save_checkpoint(save_path, epoch, encoder, decoder, encoder_optimizer, decoder_optimizer, bleu4, is_best,
                    bleu1, bleu2, bleu3):
    state = {'epoch': epoch,
             'bleu-1': bleu1,
             'bleu-2': bleu2,
             'bleu-3': bleu3,
             'bleu-4': bleu4,
             'encoder': encoder.state_dict(),
             'decoder': decoder.state_dict(),
             'encoder_optimizer': None,
             'decoder_optimizer': decoder_optimizer.state_dict()}

    if encoder_optimizer is not None:
        state['encoder_optimizer'] = encoder_optimizer.state_dict()

    filename = 'checkpoint_' + 'finetune_epoch_{}'.format(epoch) + '.pth'
    save_path = os.path.join(save_path, filename)
    torch.save(state, save_path)

    if is_best:
        torch.save(state, save_path + '_BEST_BLEU')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--decoder_dim', type=int, default=768, help='hidden size of LSTM')
    parser.add_argument('--att_dim', type=int, default=512, help='dim of att')
    parser.add_argument('--embed_dim', type=int, default=256, help='dim of word embeddings')

    parser.add_argument('--data_folder', type=str,
                        default='D:/dataset/RSICD_captions/images', help='dir of rsicd images')
    parser.add_argument('--data_name', type=str,
                        default='rsicd_5_cap_per_img_5_min_word_freq', help='data name of json and hdf5')

    parser.add_argument('--num_workers', type=int, default=4, help='num workers')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--epochs', type=int, default=3, help='epochs')
    parser.add_argument('--encoder_lr', type=float, default=1e-4, help='lr of encoder')
    parser.add_argument('--decoder_lr', type=float, default=1e-4, help='lr of decoder')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
    parser.add_argument('--lbd', type=float, default=0.2, help='lambda')
    parser.add_argument('--save_path', type=str, default='C:/Users/91630/rsicd_vr/image-caption/autodl-tmp/models/', help='path for saving models')
    parser.add_argument('--print_step', type=int, default=20, help='time step for printing states')

    args = parser.parse_args()
    main(args)
