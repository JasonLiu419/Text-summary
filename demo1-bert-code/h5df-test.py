import h5py

with h5py.File(C:\Users\96208\Desktop\cd.test.h5df, 'r') as f:
    for j_str in f['dataset']:
        obj = json.loads(j_str)
        article, abstract = obj['article'], obj['abstract']
        print(article)
