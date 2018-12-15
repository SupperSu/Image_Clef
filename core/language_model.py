import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

class LSTM_decoder(nn.Module):
    def __init__(self, input_size, vis_dim, vis_num,hidden_size, concept_size, vocab_size, num_layers = 1, embed_size_concept = 300, embed_size_voc = 300):
        """Set the hyper-parameters and build the layers."""
        super(LSTM_decoder, self).__init__()
        self.vocab_size = vocab_size
        self.concept_size = concept_size # note plus one aims to pad the concepts as the longest one.
        self.vis_dim = vis_dim  # dimension of each feature map (1d)
        self.vis_num = vis_num  # num of feature maps
        self.hidden_size = hidden_size # dim of hidden state
        self.embed_size_concept = embed_size_concept
        self.input_ly = nn.Linear(concept_size, input_size)
        self.feature_ly = nn.Linear(self.vis_num * self.vis_dim, input_size)
        self.concept_dim_ly = nn.Linear(self.embed_size_concept, self.embed_size_concept)
        self.lstm = nn.LSTMCell(input_size, hidden_size, num_layers)
        self.output_ly = nn.Linear(hidden_size, vocab_size)
        # self.init_weights()
        self.E_concept = nn.Embedding(concept_size, embed_size_concept)
        self.E_voc = nn.Embedding(vocab_size, embed_size_voc)
        """
        input attention parameters
        """
        # Note: one can use Glove or Word2Vec to reduce the dimension of U.

        self.att_in_U = nn.Linear(embed_size_concept, embed_size_voc)
        self.att_in_out_ly = nn.Linear(embed_size_concept, input_size)

        """
        output attention parameters
        """
        self.linear_V = nn.Linear(embed_size_concept, hidden_size)
        self.linear_w = nn.Linear(embed_size_concept, hidden_size)
        self.att_out_out = nn.Linear(hidden_size, vocab_size)
    # def init_weights(self):
    #     """Initialize weights."""
    #     self.embed.weight.data.uniform_(-0.1, 0.1)
    #     self.linear.weight.data.uniform_(-0.1, 0.1)
    #     self.linear.bias.data.fill_(0)


    def forward(self, features, captions, concepts,lengths):
        """
        :param features: encoded picture features, batch_size * 196 * 152
        :param captions: batch_size * time_step
        :param concepts: concepts of picture[sparse matrix], batch_size * concepts_size
        :param lengths: valid lengths for each padded caption.
        :return: predicts of each time step.
        """

        batch_size, time_step = captions.data.shape
        predicts = torch.zeros(batch_size, time_step, self.vocab_size)

        # we can initialize as mean of features or view it as 196 * 152 1d feature vector
        h0, c0 = self.get_start_states(batch_size)
        word_embeddings = self.E_voc(captions)        # batch_size * time_steps * embed_size
        concepts_embeddings = self.E_concept(concepts) # batch_size * num_concepts * con_embed_size

        for t in xrange(time_step):
            batch_size = sum(i >= t for i in lengths)
            words_input = word_embeddings[:batch_size, t, :]
            if t == 0:
                xt = self.feature_ly(torch.view(batch_size, -1)) # batch * input_size
            else:
                alpha,_ = self.att_in(concepts_embeddings, words_input)
                alpha = alpha.unsqueeze(2).expand(-1, -1, self.embed_size_concept)
                weighted_sum = torch.sum(alpha * concepts_embeddings, 1).squeeze(1)
                weighted_sum = self.concept_dim_ly(weighted_sum)
                xt = self.att_in_out_ly(weighted_sum + words_input)


            h0, c0 = self.lstm_cell(xt, (h0[:batch_size, :], c0[:batch_size, :]))
            beta = self.att_out(h0, concepts_embeddings)
            # batch size, hidden_size, #concepts
            weighted_sum_out = torch.sum(beta * F.relu(concepts_embeddings), 1).squeeze(1)
            weighted_sum_out = self.linear_w(weighted_sum_out)
            outputs = self.att_out_out(weighted_sum_out)
            predicts[:batch_size, t, :] = outputs
        return outputs

    def get_start_states(self, batch_size):
        h0 = torch.zeros(batch_size, self.hidden_size)
        c0 = torch.zeros(batch_size, self.hidden_size)
        return h0, c0

    def bilinear(self, input1, input2):
        '''
        :param input1: bz * hidden_size
        :param input2: bz * num_concepts * concept_embed_size
        :return: bz * num_concepts
        '''
        output = torch.zeros([input1.shape[0], input2.shape[1]])
        for k in range(input2.shape[1]):
            piece = input2[:, k, :] # bz * concept_embed_size
            piece = piece.squeeze(1)
            tmp = self.linear_V(piece) # bz * hidden_size
            tmp = tmp * input1
            tmp = tmp.sum(1)
            output[:,k] = tmp
        return output


    def att_in(self, y_concepts, y_word):
        """
        :argument: concept embededding, word embededding.
        :return: alpha (batch_size * concept_size), context (choosen concept)
        """

        # reduce dimension by using embedding size.
        batch_size,_= y_word.shape
        y_concepts = self.att_in_U(y_concepts)
        y_concepts = torch.transpose(y_concepts, 1,2)
        print y_concepts.shape
        y_word = y_word.unsqueeze(1)
        print y_word.shape
        logit = torch.bmm(y_word, y_concepts).squeeze(1)
        alpha = F.softmax(logit, dim = 1)
        context = torch.max(alpha, 0)
        return alpha, context

    def att_out(self, hidden, y_concepts):
        """
        :argument: current time step hidden state[batch_size, hidden_size], concepts embedding[batch_size, concept_size, embedding]
        :return: beta (batch_size, concept_size), context
        """

        # ensure two feature vectors has same non-linear transforms.
        y_concepts = F.relu(y_concepts)
        att_output = self.bilinear(hidden, y_concepts)
        beta =F.softmax(att_output, dim=1)
        return beta



if __name__ == '__main__':
    # def __init__(self, input_size, vis_dim, vis_num, hidden_size, concept_size, vocab_size, num_layers=1,
    #              embed_size_concept=300, embed_size_voc=300):

    model = LSTM_decoder(15, 12, 13, 20, 15, 20,1, 30, 20)
    bz = 5
    hd = 20
    d = 30
    outputSize = 10
    L = 15
    hidden = torch.randn([bz, hd])
    embed = torch.randn([bz, L, d])
    # output = bilinear(hidden, embed)
    # print F.softmax(output, dim=1).shape
    # alpha = torch.randn([2, 4])
    # print alpha
    # conc = torch.randn([2, 4, 3])
    # p =  alpha.unsqueeze(2).expand(-1,-1, 3)
    # print torch.sum(p * conc, 1)

