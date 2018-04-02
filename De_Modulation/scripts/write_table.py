import pandas as pd

df = pd.read_csv('../data/aufgabe_e.txt', delimiter='\t')
with open('../tex/aufgabe_e.tex','w') as tf:
    tf.write(df.to_latex(index=False))