import pickle
import retrieve_by_cui
def main():
    with open(r"concepts_index_map", "rb") as input_file:
        mapConcepts = pickle.load(input_file)
    rerer = retrieve_by_cui.retrieve()
    keys = mapConcepts.keys()
    names = []
    for key in keys:
        name = rerer.retrieve_by_cui(key)
        print name
        names.append(name)

    res = dict(zip(keys, names))
    pickle.dump(res, open("cui_names.p", "wb"))

def getConcept(concept):
    rever = retrieve_by_cui.retrieve()
    ans = rever.retrieve_by_cui(concept)
    return ans
if __name__ == '__main__':
    print getConcept("C0043189")