import pandas as pd
import pickle


def getIdx(rw, mapIdx):
    ans = []
    idx = [0] * 111156
    cps = rw.split(";")
    for cp in cps:
        idx[mapIdx[cp]] = 1
    ans.append(idx)
    return ans

def main():
    filePath = './ConceptDetectionTraining2018-Concepts.csv'
    headers = ['file_name', 'concepts']
    data = pd.read_csv(filePath, delimiter='\t', header=-1)
    data.columns = headers
    with open(r"concepts_index_map", "rb") as input_file:
        mapConcepts = pickle.load(input_file)

    data['concepts'] = data.apply(lambda x: getIdx(x[1], mapConcepts), axis=1)
    data.to_csv("./vectorized_concepts_data.csv", sep='\t')
if __name__ == '__main__':
    main()