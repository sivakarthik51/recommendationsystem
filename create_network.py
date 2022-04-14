import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import networkx as nx
from itertools import combinations

def group_books(chunk, edge_dict):
  for k, g in chunk.groupby('user_id'):
    books = list(g['book_id'])
    for (book1, book2) in combinations(books, 2):
        if (book1, book2) not in edge_dict:
          edge_dict[(book1, book2)] = 0
        edge_dict[(book1, book2)] += 1

df = pd.read_csv('goodreads_interactions.csv', chunksize=10000)
for i, chunk in enumerate(df):
    print(f'Printing Chunk {i}')
    print(chunk)
    break

edge_dict = {}
for chunk in df:
  group_books(chunk, edge_dict)

print(f"length of edge_dict is {len(edge_dict)}")
