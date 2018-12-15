import argparse
import torch
import torch.nn as nn
import numpy as np
import pickle
from utils.data_loader_concept import get_loader
from core import image_models, BCE_Loss_Multilabels
from core import language_model
from torchvision import transforms
from utils.build_concepts_vocab import Vocabulary
import torch.nn.functional as F
from sklearn.metrics import f1_score

def train(data_loader, model_language, model_vis, optimizer, criterion, epoch, total_epoch):
    total_step = len(data_loader)
    for i, (images, concepts) in enumerate(data_loader):
        model.zero_grad()
        predicts = model(images.cuda(0))
        p_labels = model.predict(predicts)
        p_labels = p_labels.cpu().data.numpy()
        t_labels = concepts.cpu().data.numpy()

        loss = criterion(predicts, concepts.cuda(0))
        loss.backward()
        optimizer.step()
        if (i) % 1000 == 0:
            _, ave_bf1 = batch_f1(p_labels, t_labels)
            print 'Epoch [%d/%d]: [%d/%d], loss: %5.4f, perplexity: %5.4f, batch_f1: %5.4f' % (epoch, total_epoch, i,
                                                                              total_step, loss.data[0],
                                                                              np.exp(loss.data[0]),
                                                                                               ave_bf1
                                                                              )
def main(args):
    tr_transform = transforms.Compose([
        transforms.RandomCrop(size=256, pad_if_needed=True),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    with open('./utils/concept_voc.pkl', 'rb') as f:
        voc_con = pickle.load(f)
    num_con = len(voc_con)
    model_language= language_model.LSTM_decoder()
    params = [{'params': filter(lambda p: p.requires_grad, md.parameters())}]

    # criterion = BCE_Loss_Multilabels.BCELossMultiLabels()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params, lr=0.001)
    image_root = args.image_root
    batch_size = args.batch_size
    epoch = args.num_epochs
    if torch.cuda.is_available():
        print "use gpu!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        md.cuda(0)

    train_path = args.cap_con_train
    # valid_path = args.cap_con_valid
    # valid_path = './CaptionConceptValidation.csv'
    train_data = get_loader(image_root, train_path, voc_con, tr_transform, batch_size, True, 0)
    # val_data = get_loader(image_root, valid_path, voc_con, tr_transform, batch_size, False, 1)

    for ep in xrange(epoch):
        train(train_data, md, optimizer, criterion, ep, epoch)
        # valid(val_data, md)
        torch.save(md, './checkpoints/Res_Class_%d.pth' % (ep))
    pass

#input_size, vis_dim, vis_num,hidden_size, concept_size, vocab_size, num_layers = 1, embed_size_concept = 300, embed_size_voc = 300
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data loader
    parser.add_argument('--image_root', type=str,
                        default='./data/sample_images')
    parser.add_argument('--vocab_word_path', type=str,
                        default="./utils/vocab.pkl")
    parser.add_argument('--vocab_concept', type=str,
                        default="./data/concept_voc.pkl")
    parser.add_argument('--cap_con_root', type=str,
                        default='./data/Captions-Concepts-Training.csv')

    # image model setting
    parser.add_argument('--concept_size', type=int, default=111156)
    parser.add_argument('--input_size', type=int, default=256)
    parser.add_argument('--vis_dim', type=int, default=300)
    parser.add_argument('--vis_num', type=int, default=300)
    parser.add_argument('--embed_size_concept', type=int, default=300)

    # optimizer setting
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=2)

    args = parser.parse_args()
    print args
    main(args)