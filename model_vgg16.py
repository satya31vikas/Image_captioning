import torch
import torch.nn as nn
import torchvision.models as models
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        vgg = models.vgg16_bn(pretrained=True)
        vgg = nn.Sequential(*list(vgg.children())[:-2])     
        vgg = nn.Sequential(*list(*vgg)[:-1])               
        self.vgg = vgg

    def forward(self, images):
        output = self.vgg(images)                           
        output = output.permute(0, 2, 3, 1)                 
        return output
    # (Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    # (Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    # (MaxPool2D(pool_size=(2,2),strides=(2,2)))
    # (Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    # (Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    # (MaxPool2D(pool_size=(2,2),strides=(2,2)))
    # (Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    # (Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    # (Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    # (MaxPool2D(pool_size=(2,2),strides=(2,2)))
    # (Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    # (Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    # (Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    # (MaxPool2D(pool_size=(2,2),strides=(2,2)))
    # (Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    # (Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    # (Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    # (MaxPool2D(pool_size=(2,2),strides=(2,2)))

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, att_dim):
        super(Attention, self).__init__()
        self.att_dim = att_dim
        self.k = nn.Linear(encoder_dim, att_dim)
        self.q = nn.Linear(decoder_dim, att_dim)
        self.v = nn.Linear(encoder_dim, encoder_dim)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, h):
        att_k = self.k(encoder_out)
        att_q = self.q(h)
        att_v = self.v(encoder_out)
        att_final = torch.bmm(att_k, att_q.unsqueeze(2))            
        att_final = torch.div(att_final, math.sqrt(self.att_dim))   

        alpha = self.softmax(self.relu(att_final))
        att_out = alpha * att_v

        att_out = att_out.sum(dim=1)    
        alpha = alpha.squeeze(2)
        return att_out, alpha           


class DecoderLSTM(nn.Module):
    def __init__(self, decoder_dim, att_dim, embed_dim, vocab_size, encoder_dim=512, dropout=0.5):
        super(DecoderLSTM, self).__init__()
        self.encoder_dim = encoder_dim                                  
        self.decoder_dim = decoder_dim
        self.att_dim = att_dim
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, att_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        self.f_init_h = nn.Linear(encoder_dim, decoder_dim)             
        self.f_init_c = nn.Linear(encoder_dim, decoder_dim)

        self.f_beta = nn.Linear(decoder_dim, encoder_dim)               
        self.sigmoid = nn.Sigmoid()

        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.dropout = nn.Dropout(p=self.dropout)


    def forward(self, encoder_out, encoded_captions, caption_lens):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)  
        vocab_size = self.vocab_size

        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)     
        num_pixels = encoder_out.size(1)                                

        caption_lens, sort_ind = caption_lens.squeeze(1).sort(dim=0, descending=True)       

        encoder_out = encoder_out[sort_ind]             
        encoded_captions = encoded_captions[sort_ind]   

        h, c = self.init_h_c(encoder_out)               

        embeddings = self.embedding(encoded_captions)   

        decode_len = [c - 1 for c in caption_lens]      

        predictions = torch.zeros(batch_size, max(decode_len), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_len), num_pixels).to(device)        

        for i in range(max(decode_len)):  
            batch_size_i = sum([l > i for l in decode_len])
            attended_encode_out, alpha = self.attention(encoder_out[: batch_size_i], h[: batch_size_i])
            beta = self.sigmoid(self.f_beta(h[: batch_size_i]))     

            attended_encode_out = beta * attended_encode_out        

            decoder_input = torch.cat([embeddings[: batch_size_i, i, :], attended_encode_out], dim=1)
            h, c = self.decode_step(decoder_input, (h[: batch_size_i], c[: batch_size_i]))
            predict_tokens = self.fc(self.dropout(h))               
            predictions[: batch_size_i, i, :] = predict_tokens
            alphas[: batch_size_i, i, :] = alpha                    
        
        
        return predictions, alphas, encoded_captions, decode_len, sort_ind

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)

    def init_h_c(self, encoder_out):
        mean = encoder_out.mean(dim=1)
        h = self.f_init_h(mean)
        c = self.f_init_c(mean)
        return h, c

