"""
offer what build_vacb and data_loader needs
"""
import pandas as pd
def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

class CLEF:
    def __init__(self, caption_path = "./data/caption/Captions-Concepts-Training.csv"):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        max_len = 145
        self.clef = pd.read_csv(caption_path,  sep='\t')
        del_idx = []
        for i, caption in enumerate(self.clef['caption']):
	    if len(caption.split(" ")) > max_len:
                del_idx.append(i)
            caption = caption.replace('.', '').replace(',', '').replace("'", "").replace('"', '')
            caption = caption.replace('&', 'and').replace('(', '').replace(")", "").replace('-', ' ')
            caption = " ".join(caption.split())  # replace multiple spaces

            self.clef.set_value(i, 'caption', caption.lower())

        # delete captions if size is larger than max_length
        print "The number of captions before deletion: %d" % len(self.clef)
        self.clef = self.clef.drop(self.clef.index[del_idx])
        print "The number of captions after deletion: %d" % len(self.clef)
        self.clef['image_id'] = self.clef.index
        self.ids = self.clef.index
        # load dataset
