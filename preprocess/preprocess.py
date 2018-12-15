import pandas as pd

def processCaption(data, max_len = 145):
    del_idx = []
    for i, caption in enumerate(data['caption']):
        caption = caption.replace('.', '').replace(',', '').replace("'", "").replace('"', '')
        caption = caption.replace('&', 'and').replace('(', '').replace(")", "").replace('-', ' ')
        caption = " ".join(caption.split())  # replace multiple spaces

        data.set_value(i, 'caption', caption.lower())
        if len(caption.split(" ")) > max_len:
            del_idx.append(i)

    # delete captions if size is larger than max_length
    print "The number of captions before deletion: %d" % len(data)
    data = data.drop(data.index[del_idx])
    print "The number of captions after deletion: %d" % len(data)
    data['image_id'] = data.index
    return data

def main(caption_path):
    data = pd.read_csv(caption_path, header=None, sep='\t')
    data.columns = ['file_name', 'caption']
    processCaption(data)
