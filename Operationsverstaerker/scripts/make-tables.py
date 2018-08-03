import pandas as pd

def make_table(inputfile):
    data = pd.read_csv(inputfile, sep=" ", header='infer', index_col=False)
    outputfile = inputfile[:-4] + '.tex'
    with open(outputfile,'w') as tf:
        tf.write(data.to_latex(index=False))

if __name__ == '__main__':
    make_table('../data/a_R1_0-2k_RN_1k.txt')
    make_table('../data/a_R1_10-02k_RN_33-2k.txt')
    make_table('../data/a_R1_10-02k_RN_100k.txt')
    make_table('../data/a_R1_33-3k_RN_100k.txt')
    make_table('../data/d_R1_10-02k_C1_23-1n.txt')
    make_table('../data/e_R1_0-2k_C1_23-1n.txt')
