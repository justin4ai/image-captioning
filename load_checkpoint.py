import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import numpy as np
import copy
import argparse

import torch
import torch.nn as nn

#from ..transformer_layers import *
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

"""
This file defines layer types that are commonly used for transformers.
"""

class PositionalEncoding(nn.Module):
    """
    Encodes information about the positions of the tokens in the sequence. In
    this case, the layer has no learnable parameters, since it is a simple
    function of sines and cosines.
    """
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        """
        Construct the PositionalEncoding layer.

        Inputs:
         - embed_dim: the size of the embed dimension
         - dropout: the dropout value
         - max_len: the maximum possible length of the incoming sequence
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        assert embed_dim % 2 == 0
        # Create an array with a "batch dimension" of 1 (which will broadcast
        # across all examples in the batch).
        pe = torch.zeros(1, max_len, embed_dim)
        ############################################################################
        # TODO: Construct the positional encoding array as described in            #
        # Transformer_Captioning.ipynb.  The goal is for each row to alternate     #
        # sine and cosine, and have exponents of 0, 0, 2, 2, 4, 4, etc. up to      #
        # embed_dim. Of course this exact specification is somewhat arbitrary, but #
        # this is what the autograder is expecting. For reference, our solution is #
        # less than 5 lines of code.                                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pos = torch.arange(0, max_len, dtype = torch.float).view(max_len, 1)#.unsqueeze(1)
        t = torch.pow(torch.tensor([1e-4]), torch.arange(0, embed_dim, 2)/embed_dim)
        
        pe[0, :, 0::2] = torch.sin(pos * t)
        pe[0, :, 1::2] = torch.cos(pos * t)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # Make sure the positional encodings will be saved with the model
        # parameters (mostly for completeness).
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Element-wise add positional embeddings to the input sequence.

        Inputs:
         - x: the sequence fed to the positional encoder model, of shape
              (N, S, D), where N is the batch size, S is the sequence length and
              D is embed dim
        Returns:
         - output: the input sequence + positional encodings, of shape (N, S, D)
        """
        N, S, D = x.shape
        # Create a placeholder, to be overwritten by your code below.
        output = torch.empty((N, S, D))
        ############################################################################
        # TODO: Index into your array of positional encodings, and add the         #
        # appropriate ones to the input sequence. Don't forget to apply dropout    #
        # afterward. This should only take a few lines of code.                    #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        output = x + self.pe[:, :S, :]
        output = self.dropout(output)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return output


class MultiHeadAttention(nn.Module):
    """
    A model layer which implements a simplified version of masked attention, as
    introduced by "Attention Is All You Need" (https://arxiv.org/abs/1706.03762).

    Usage:
      attn = MultiHeadAttention(embed_dim, num_heads=2)

      # self-attention
      data = torch.randn(batch_size, sequence_length, embed_dim)
      self_attn_output = attn(query=data, key=data, value=data)

      # attention using two inputs
      other_data = torch.randn(batch_size, sequence_length, embed_dim)
      attn_output = attn(query=data, key=other_data, value=other_data)
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Construct a new MultiHeadAttention layer.

        Inputs:
         - embed_dim: Dimension of the token embedding
         - num_heads: Number of attention heads
         - dropout: Dropout probability
        """
        super().__init__()
        assert embed_dim % num_heads == 0

        # We will initialize these layers for you, since swapping the ordering
        # would affect the random number generation (and therefore your exact
        # outputs relative to the autograder). Note that the layers use a bias
        # term, but this isn't strictly necessary (and varies by
        # implementation).
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        self.attn_drop = nn.Dropout(dropout)

        self.n_head = num_heads
        self.emd_dim = embed_dim
        self.head_dim = self.emd_dim // self.n_head

    def forward(self, query, key, value, attn_mask=None):
        """
        Calculate the masked attention output for the provided data, computing
        all attention heads in parallel.

        In the shape definitions below, N is the batch size, S is the source
        sequence length, T is the target sequence length, and E is the embedding
        dimension.

        Inputs:
        - query: Input data to be used as the query, of shape (N, S, E)
        - key: Input data to be used as the key, of shape (N, T, E)
        - value: Input data to be used as the value, of shape (N, T, E)
        - attn_mask: Array of shape (S, T) where mask[i,j] == 0 indicates token
          i in the source should not influence token j in the target.

        Returns:
        - output: Tensor of shape (N, S, E) giving the weighted combination of
          data in value according to the attention weights calculated using key
          and query.
        """
        N, S, E = query.shape
        N, T, E = value.shape
        # Create a placeholder, to be overwritten by your code below.
        output = torch.empty((N, S, E))
        ############################################################################
        # TODO: Implement multiheaded attention using the equations given in       #
        # Transformer_Captioning.ipynb.                                            #
        # A few hints:                                                             #
        #  1) You'll want to split your shape from (N, T, E) into (N, T, H, E/H),  #
        #     where H is the number of heads.                                      #
        #  2) The function torch.matmul allows you to do a batched matrix multiply.#
        #     For example, you can do (N, H, T, E/H) by (N, H, E/H, T) to yield a  #
        #     shape (N, H, T, T). For more examples, see                           #
        #     https://pytorch.org/docs/stable/generated/torch.matmul.html          #
        #  3) For applying attn_mask, think how the scores should be modified to   #
        #     prevent a value from influencing output. Specifically, the PyTorch   #
        #     function masked_fill may come in handy.                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Referencing my implementation : https://github.com/justin4ai/NAT-Diffuser/blob/nat-diffuser/models/transformer.py

        q = self.query(query).view(N, S, self.n_head, E // self.n_head).transpose(1,2) 
        k = self.key(key).view(N, T, self.n_head, E // self.n_head).transpose(1,2)
        v = self.value(value).view(N, T, self.n_head, E // self.n_head).transpose(1,2) 

        att = torch.matmul(q, k.transpose(2, 3)) * (1.0 / math.sqrt(k.size(-1)) )
        
        if attn_mask is not None:
            att = att.masked_fill(attn_mask == 0, float('-inf'))

        att = F.softmax(att, dim = -1) 
        att = self.attn_drop(att) 
        y = torch.matmul(att, v)

        y = y.transpose(1, 2).contiguous().view(N, S, -1)
        output = self.proj(y)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return output, att



from rotary_embedding_torch import RotaryEmbedding
from tqdm import tqdm

# Setup cell.
import time, os, json
import numpy as np
import matplotlib.pyplot as plt

from CV.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from CV.transformer_layers import *
from CV.captioning_solver_transformer import CaptioningSolverTransformer
from CV.classifiers.transformer import CaptioningTransformer
from CV.coco_utils import load_coco_data, sample_coco_minibatch, decode_captions
from CV.image_utils import image_from_url

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


class CaptioningTransformer(nn.Module):
    """
    A CaptioningTransformer produces captions from image features using a
    Transformer decoder.

    The Transformer receives input vectors of size D, has a vocab size of V,
    works on sequences of length T, uses word vectors of dimension W, and
    operates on minibatches of size N.
    """
    def __init__(self, word_to_idx, input_dim, wordvec_dim, num_heads=4,
                 num_layers=2, max_length=50):
        """
        Construct a new CaptioningTransformer instance.

        Inputs:
        - word_to_idx: A dictionary giving the vocabulary. It contains V entries.
          and maps each string to a unique integer in the range [0, V).
        - input_dim: Dimension D of input image feature vectors.
        - wordvec_dim: Dimension W of word vectors.
        - num_heads: Number of attention heads.
        - num_layers: Number of transformer layers.
        - max_length: Max possible sequence length.
        """
        super().__init__()

        vocab_size = len(word_to_idx)
        self.vocab_size = vocab_size
        self._null = word_to_idx["<NULL>"]
        self._start = word_to_idx.get("<START>", None)
        self._end = word_to_idx.get("<END>", None)

        self.visual_projection = nn.Linear(input_dim, wordvec_dim)
        self.embedding = nn.Embedding(vocab_size, wordvec_dim, padding_idx=self._null)
        self.positional_encoding = PositionalEncoding(wordvec_dim, max_len=max_length)


        decoder_layer = TransformerDecoderLayer(input_dim=wordvec_dim, num_heads=num_heads)
        self.transformer = TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.apply(self._init_weights)

        self.output = nn.Linear(wordvec_dim, vocab_size)


    def _init_weights(self, module):
        """
        Initialize the weights of the network.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, features, captions):
        """
        Given image features and caption tokens, return a distribution over the
        possible tokens for each timestep. Note that since the entire sequence
        of captions is provided all at once, we mask out future timesteps.

        Inputs:
         - features: image features, of shape (N, D)
         - captions: ground truth captions, of shape (N, T)

        Returns:
         - scores: score for each token at each timestep, of shape (N, T, V)
        """
        N, T = captions.shape
        # Create a placeholder, to be overwritten by your code below.
        scores = torch.empty((N, T, self.vocab_size))
        ############################################################################
        # TODO: Implement the forward function for CaptionTransformer.             #
        # A few hints:                                                             #
        #  1) You first have to embed your caption and add positional              #
        #     encoding. You then have to project the image features into the same  #
        #     dimensions.                                                          #
        #  2) You have to prepare a mask (tgt_mask) for masking out the future     #
        #     timesteps in captions. torch.tril() function might help in preparing #
        #     this mask.                                                           #
        #  3) Finally, apply the decoder features on the text & image embeddings   #
        #     along with the tgt_mask. Project the output to scores per token      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


        caption_embeddings = self.embedding(captions)

        caption_embeddings = self.positional_encoding(caption_embeddings)


        projected_features = self.visual_projection(features).unsqueeze(1)


        tgt_mask = torch.tril(torch.ones(T, T,
                                         device = caption_embeddings.device,
                                         dtype = caption_embeddings.dtype))


        features = self.transformer(caption_embeddings, projected_features, tgt_mask)
        scores = self.output(features)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return scores

    #def sample(self, features, max_length=30):
        """
        Given image features, use greedy decoding to predict the image caption.

        Inputs:
         - features: image features, of shape (N, D)
         - max_length: maximum possible caption length

        Returns:
         - captions: captions for each example, of shape (N, max_length)
        """
        # with torch.no_grad():
        #     features = torch.Tensor(features)
        #     N = features.shape[0]

        #     # Create an empty captions tensor (where all tokens are NULL).
        #     captions = self._null * np.ones((N, max_length), dtype=np.int32)

        #     # Create a partial caption, with only the start token.
        #     partial_caption = self._start * np.ones(N, dtype=np.int32)
        #     partial_caption = torch.LongTensor(partial_caption)
        #     # [N] -> [N, 1]
        #     partial_caption = partial_caption.unsqueeze(1)

        #     for t in range(max_length):

        #         # Predict the next token (ignoring all other time steps).
        #         output_logits = self.forward(features, partial_caption)
        #         output_logits = output_logits[:, -1, :]

        #         # Choose the most likely word ID from the vocabulary.
        #         # [N, V] -> [N]
        #         word = torch.argmax(output_logits, axis=1)

        #         # Update our overall caption and our current partial caption.
        #         captions[:, t] = word.numpy()
        #         word = word.unsqueeze(1)
        #         partial_caption = torch.cat([partial_caption, word], dim=1)

        #     return captions


    def sample(self, features, partial_caption = None, max_length=30):
        with torch.no_grad():
            features = torch.Tensor(features)
            N = features.shape[0]
  
            if partial_caption is None:

                # Create an empty captions tensor (where all tokens are NULL).
                captions = self._null * np.ones((N, max_length), dtype=np.int32)

                # Create a partial caption, with only the start token.
                partial_caption = self._start * np.ones(N, dtype=np.int32)
                partial_caption = torch.LongTensor(partial_caption).unsqueeze(1)

                for t in range(max_length):
                    # Predict the next token (ignoring all other time steps).
                    output_logits = self.forward(features, partial_caption)
                    output_logits = output_logits[:, -1, :]

                    # Choose the most likely word ID from the vocabulary.
                    word = torch.argmax(output_logits, axis=1)



                    # Update our overall caption and our current partial caption.
                    captions[:, t] = word.numpy()
                    word = word.unsqueeze(1)
                    partial_caption = torch.cat([partial_caption, word], dim=1)

                return captions
            else:

                # Use partial caption to predict the next token
                output_logits = self.forward(features, partial_caption)
                output_logits = output_logits[:, -1, :]  # Get the logits for the last word


                return output_logits


class TransformerDecoderLayer(nn.Module):
    """
    A single layer of a Transformer decoder, to be used with TransformerDecoder.
    """
    def __init__(self, input_dim, num_heads, dim_feedforward=2048, dropout=0.1):
        """
        Construct a TransformerDecoderLayer instance.

        Inputs:
         - input_dim: Number of expected features in the input.
         - num_heads: Number of attention heads
         - dim_feedforward: Dimension of the feedforward network model.
         - dropout: The dropout value.
        """
        super().__init__()
        self.rotary_emb = RotaryEmbedding(dim = 32)

        self.self_attn = MultiHeadAttention(input_dim, num_heads, dropout)
        self.multihead_attn = MultiHeadAttention(input_dim, num_heads, dropout)
        self.linear1 = nn.Linear(input_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, input_dim)

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.norm3 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()


    def forward(self, tgt, memory, tgt_mask=None):
        """
        Pass the inputs (and mask) through the decoder layer.

        Inputs:
        - tgt: the sequence to the decoder layer, of shape (N, T, W)
        - memory: the sequence from the last layer of the encoder, of shape (N, S, D)
        - tgt_mask: the parts of the target sequence to mask, of shape (T, T)

        Returns:
        - out: the Transformer features, of shape (N, T, W)
        """
        # Perform self-attention on the target sequence (along with dropout and
        # layer norm).

#        tgt2 = self.self_attn(query=self.rotary_emb.rotate_queries_or_keys(tgt), key=self.rotary_emb.rotate_queries_or_keys(tgt), value=tgt, attn_mask=tgt_mask)
        tgt2, weight_self = self.self_attn(query=tgt, key=tgt, value=tgt, attn_mask=tgt_mask) # weight output added)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Attend to both the target sequence and the sequence from the last
        # encoder layer.
        #tgt2 = self.multihead_attn(query=self.rotary_emb.rotate_queries_or_keys(tgt), key=self.rotary_emb.rotate_queries_or_keys(memory), value=memory)
        tgt2, weight_cross = self.multihead_attn(query=tgt, key=memory, value=memory) # weight output added   
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # # Pass
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memory, tgt_mask=None):
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask)

        return output


def load_checkpoint(model, optimizer, file_path):
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss_history = checkpoint['loss_history']
    print(f"Checkpoint loaded: epoch {epoch}")
    return model, optimizer, epoch, loss_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--beam', default=True, type=bool)
    parser.add_argument('--iter', default=1000, type=bool)
    args = parser.parse_args()
    print(args.beam)


    torch.manual_seed(231)
    np.random.seed(231)
    # 데이터 로드
    data = load_coco_data(pca_features=False)

    # 모델과 옵티마이저 초기화
    transformer = CaptioningTransformer(
            word_to_idx=data['word_to_idx'],
            input_dim=data['train_features'].shape[1],
            wordvec_dim=256,
            num_heads=8,
            num_layers=6,
            max_length=30
            )
    optimizer = optim.Adam(transformer.parameters(), lr=0.0006)

    # 체크포인트 로드
    checkpoint_path = '/home/justin/workspace/cv/hw3/CV/latest.pth'
    transformer, optimizer, epoch, loss_history = load_checkpoint(transformer, optimizer, checkpoint_path)

    # 모델을 평가 모드로 전환
    transformer.eval()

    if args.beam == False:
        print("asdf")
        def generate_caption(model, feature, data):
            sample_caption = model.sample(feature, max_length=30)
            sample_caption = decode_captions(sample_caption, data['idx_to_word'])[0]
            return sample_caption


        # 예측 수행
        student_id = "2020047029"  # FIXME
        pred = []
        nice_feat = data['nice_feature']
        nice_feat = np.expand_dims(nice_feat, axis=1)

        # for i in tqdm(range(len(nice_feat))):
        for i in tqdm(range(args.iter)):
            caption = generate_caption(transformer, nice_feat[i], data)
            image_id = i + 1
            pred.append({'image_id': image_id, 'caption': caption})

        result = {"student_id": student_id, "prediction": pred}
        json.dump(result, open('pred_beamfalse.json', 'w'))

        print("Predictions saved to pred.json")

    else:



        import heapq
        import numpy as np

        def beam_search(transformer, feature, beam_width=3, max_length=30):
            start_token = [data['word_to_idx']['<START>']]
            end_token = data['word_to_idx']['<END>']

            sequences = [[start_token, 0.0]]
            while len(sequences[0][0]) < max_length:

                all_candidates = []
                for seq, score in sequences:
                    #print(seq)
                    if seq[-1] == end_token:
                        all_candidates.append((seq, score))
                        continue

                    partial_caption = torch.LongTensor([seq])
                    predictions = transformer.sample(feature, partial_caption)
                    predictions = predictions.squeeze(0)

                    for idx, p in enumerate(predictions):

                        p = max(p.item(), 1e-8)
                        candidate = [seq + [idx], score - np.log(p)]
                        all_candidates.append(candidate)

                    ordered = sorted(all_candidates, key=lambda tup: tup[1])
                    ordered = [tup for tup in ordered if 3 not in tup[0]]
                    
                    #print(ordered)

                    sequences = ordered[:beam_width] # UNK gone

                if all(seq[-1] == end_token for seq, score in sequences):
                    break
            
            #print(f"sequences : {sequences}")
            return sequences[0][0]


        def generate_caption(feature):
            sample_caption = beam_search(transformer, feature)
            #print(sample_caption)
            sample_caption = decode_captions(np.array(sample_caption), data['idx_to_word'])
            return sample_caption



        student_id = "2020047029" # FIXME
        pred = []
        nice_feat = data['nice_feature']
        nice_feat = np.expand_dims(nice_feat, axis=1)

        #for i in tqdm(range(len(nice_feat))):
        for i in tqdm(range(args.iter)):
            caption = generate_caption(nice_feat[i])
            image_id = i + 1
            pred.append({'image_id' : image_id, 'caption' : caption})

        result = {"student_id" : student_id, "prediction" : pred}
        json.dump(result, open('pred_beamtrue.json', 'w'))

    
    #print(data['word_to_idx']['<UNK>'])