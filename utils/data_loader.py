import torch
import torch.utils.data as data
import os
import nltk
from PIL import Image
import CLEF

class ClefDataset(data.Dataset):
    """CLEF Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, vocab_word, vocab_concept,root = "/data/resized_train/", csv= "./data/caption/CaptionPredictionTraining2018-Captions.csv",transform=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            csv: clef annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.clef = CLEF.CLEF(csv)
        self.ids = self.clef.ids
        self.vocab_word = vocab_word
        self.transform = transform
        self.vocab_concept = vocab_concept

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        clef = self.clef.clef   # pandas dataframe
        vocab = self.vocab_word
        vocab_concepts = self.vocab_concept
        ann_id = self.ids[index]
        caption = clef.loc[ann_id]['caption']
        concepts_whole = clef.loc[ann_id]['concepts']
        img_id = clef.loc[ann_id]['image_id']
        path = clef.loc[ann_id]['file_name'] + ".jpg"
        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        concepts = []
        concepts.extend([vocab_concepts(concept) for concept in concepts_whole])
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target, concepts

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, concepts = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths_con = [len(con) for con in concepts]
    maxLen = max(lengths_con)
    targets_con = torch.zeros(len(lengths_con), maxLen).long()
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, con in enumerate(concepts):
        end = lengths_con[i]
        targets_con[i, :end] = con[:end]

    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths, targets_con


def get_loader(root, path, vocab, vocab_con,transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    clef = ClefDataset(root=root,
                       csv= path,
                       vocab=vocab,
                       vocab_concept= vocab_con,
                       transform=transform)

    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for every iteration.
    # images: tensor of shape (batch_size, 3, 224, 224).
    # captions: tensor of shape (batch_size, padded_length).
    # lengths: list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=clef,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)


    return data_loader