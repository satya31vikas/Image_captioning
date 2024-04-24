import os
import numpy as np
import h5py
import json
import torch
from PIL import Image
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample


def create_input_files(dataset, annotations_json_path, image_folder, captions_per_image, min_word_freq, output_folder,
                       max_len=100):
    

    assert dataset == 'rsicd'

    with open(annotations_json_path, 'r') as j:
        data = json.load(j)

    image_paths = []
    image_captions = []
    word_freq = Counter()

    for img in data['images']:
        captions = []
        for c in img['sentences']:  
            caption_text = c['raw']  
            word_freq.update(caption_text.split())
            if len(caption_text.split()) <= max_len:
                captions.append(caption_text.split())
        if len(captions) == 0:
            continue

        path = os.path.join(image_folder, img['filename'])

        image_paths.append(path)
        image_captions.append(captions)

    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'

    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)

    seed(123)
    with h5py.File(os.path.join(output_folder, 'IMAGES_' + base_filename + '.hdf5'), 'a') as h:
        h.attrs['captions_per_image'] = captions_per_image
        images = h.create_dataset('images', (len(image_paths), 3, 224, 224), dtype='uint8')

        print("\nReading images and captions, storing to file...\n")

        enc_captions = []
        caplens = []

        for i, path in enumerate(tqdm(image_paths)):
            if len(image_captions[i]) < captions_per_image:
                captions = image_captions[i] + [choice(image_captions[i]) for _ in range(captions_per_image - len(image_captions[i]))]
            else:
                captions = sample(image_captions[i], k=captions_per_image)


            assert len(captions) == captions_per_image

            img = Image.open(path)
            width, height = img.size
            if width > height:
                left = (width - height) / 2
                right = width - left
                top = 0
                bottom = height
            else:
                top = (height - width) / 2
                bottom = height - top
                left = 0
                right = width

            img = img.crop((left, top, right, bottom))
            img = img.resize([224, 224], Image.ANTIALIAS)

            img = np.array(img)
            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]
                img = np.concatenate([img, img, img], axis=2)
            img = np.transpose(img, (2, 0, 1))

            assert img.shape == (3, 224, 224)
            assert np.max(img) <= 255

            images[i] = img

            for j, c in enumerate(captions):
                enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                    word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))
                
                c_len = len(c) + 2

                enc_captions.append(enc_c)
                caplens.append(c_len)

        assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)

        with open(os.path.join(output_folder, 'CAPTIONS_' + base_filename + '.json'), 'w') as j:
            json.dump(enc_captions, j)

        with open(os.path.join(output_folder, 'CAPLENS_' + base_filename + '.json'), 'w') as j:
            json.dump(caplens, j)


if __name__ == '__main__':
    create_input_files(dataset='rsicd',
                       annotations_json_path='D:/dataset/RSICD_captions/dataset_rsicd.json',
                       image_folder='D:/dataset/RSICD_captions/images',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='D:/dataset/RSICD_captions/images',
                       max_len=50)
