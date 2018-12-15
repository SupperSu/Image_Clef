import pandas as pd


def get_dataframe(path, max_len=145):
    data = pd.read_csv(path,  sep="\t")
    del_idx = []
    for i, caption in enumerate(data["caption"]):
        caption = caption.replace('.', '').replace(',', '').replace("'", "").replace('"', '')
        caption = caption.replace('&', 'and').replace('(', '').replace(")", "").replace('-', ' ')
        caption = " ".join(caption.split())  # replace multiple spaces
        data.set_value(i, 'caption', caption.lower())
        if max_len is not None:
            if len(caption.split(" ")) > max_len:
                del_idx.append(i)
    # delete captions if size is larger than max_length
    print "The number of captions before deletion: %d" % len(data)
    # without reset index, since get_item use index of relative index, so data.index is the abosolute index
    data = data.drop(data.index[del_idx])
    print "The number of captions after deletion: %d" % len(data)
    return data

def split_csv(path):
    df1 = get_dataframe(path, 145)
    df2 = df1.sample(int(0.1 * df1.shape[0])) # validation set
    train = pd.concat([df2, df1]).drop_duplicates(keep=False)
    train.to_csv("./CaptionConceptTraining.csv", sep = '\t', index= False, header=True)
    df2.to_csv("./CaptionConceptValidation.csv", sep = '\t', index=False, header=True)


if __name__ == '__main__':
    split_csv("./Captions-Concepts-Training.csv")