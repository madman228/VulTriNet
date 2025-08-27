import networkx as nx
import pydot
import numpy as np
import argparse
import os
import sent2vec
import pickle
import glob
from multiprocessing import Pool
from functools import partial
from sent2vec import Sent2vecModel

#model_path = './sard_model.bin'

def parse_options():
    parser = argparse.ArgumentParser(description='Image-based Vulnerability Detection.')
    parser.add_argument('-i', '--input', help='The path of a dir which consists of some dot_files')
    parser.add_argument('-o', '--out', help='The path of output.', required=True)
    parser.add_argument('-m', '--model', help='The path of model.', required=True)
    args = parser.parse_args()
    return args

def graph_extraction(dot):
    graph = nx.drawing.nx_pydot.read_dot(dot)
    #graph = nx.drawing.nx_agraph.read_dot(dot)
    return graph

def sentence_embedding(sentence):
    emb = sent2vec_model.embed_sentence(sentence)
    return emb[0]

def image_generation_codebert(dot):
    codebert_channel = []
    embeddings = np.load(dot)
    return embeddings

def image_generation_ast(dot):
    ast_channel = []
    embeddings = np.load(dot)
    if embeddings.ndim != 2:
        return None
    for embedding in embeddings:
        ast_channel.append(embedding)
    return ast_channel


def image_generation_pdg(dot):
    try:
        pdg = graph_extraction(dot)
        labels_dict = nx.get_node_attributes(pdg, 'label')
        labels_code = dict()
        for label, all_code in labels_dict.items():
            # code = all_code.split('code:')[1].split('\\n')[0]
            # code = all_code[all_code.index(",") + 1:-2].split('\\n')[0]
            code = all_code[all_code.index(",") + 1:-1].split('\n')[0]
            # 
            code = code.replace("static void", "void")
            labels_code[label] = code
        G = nx.DiGraph()
        G.add_nodes_from(pdg.nodes())
        G.add_edges_from(pdg.edges())
        pdg_channel = []
        for label, code in labels_code.items():
            #print(code)
            if not code or (isinstance(code, str) and not code.strip()) or (isinstance(code, list) and len(code) == 0):
                print(f"Skipping empty code for label: {label}")
                continue  
            line_vec = sentence_embedding(code)
            line_vec = np.array(line_vec)
            pdg_channel.append(line_vec)
        return pdg_channel
    except Exception as e:
        #print("pdg gg")
        return None

from PIL import Image
def write_to_pkl(dot, out, existing_files):
    dot_name = dot.split('/')[-1].split('.dot')[0]
    print(dot_name)
    if dot_name in existing_files:
        return None
    else:
        #print(dot)
        #CodeBERT-path
        path1 = ''
        #PDG-path
        path2 = ''
        #AST-path
        path3 = ''
        base_name = os.path.basename(dot).split('_')[0]  
        #build your specfic path
        dot_path1 = os.path.join(path1,'codebert_'+ dot +'_embeddings.npy')
        dot_path2 = os.path.join(path2,dot+'.dot')
        dot_path3 = os.path.join(path3,dot+'.npy')
        #print(dot_path1)
        # check or return
        if not (os.path.exists(dot_path1) and os.path.exists(dot_path2) and os.path.exists(dot_path3)):
            print(f"Missing required files:")
            return None 
        ast_channels = image_generation_ast(dot_path3)
        pdg_channels = image_generation_pdg(dot_path2)
        codebert_channels = image_generation_codebert(dot_path1)
        #print(pdg_channels)
        if pdg_channels is None or codebert_channels is None or ast_channels is None:
            print('have some problems')
            return None
        else:
            out_pkl = out + dot_name + '.pkl'
            data = [ast_channels, pdg_channels, codebert_channels]
            
            ast_len = np.array(ast_channels)
            pdg_len = np.array(pdg_channels)
            codebert_len = np.array(codebert_channels)
            
            print(ast_len.shape,pdg_len.shape,codebert_len.shape)
                    
            with open(out_pkl, 'wb') as f:
                pickle.dump(data, f)
                
def main():
    #if you like,you can use args to execute
    #args = parse_options()
    # dir_name = args.input
    # out_path = args.out
    #trained_model_path = args.model
    trained_model_path ="./data_model.bin"
    global sent2vec_model
    sent2vec_model = sent2vec.Sent2vecModel()
    sent2vec_model.load_model(trained_model_path)
    dir_name = ''
    out_path = ''
    if dir_name[-1] == '/':
        dir_name = dir_name
    else:
        dir_name += "/"
    dotfiles = glob.glob(dir_name + '*.dot')
    dotfile_names = [os.path.basename(file).split('.')[0] for file in dotfiles]
    if out_path[-1] == '/':
        out_path = out_path
    else:
        out_path += '/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    existing_files = glob.glob(out_path + "/*.pkl")
    existing_files = [f.split('.pkl')[0] for f in existing_files]
    pool = Pool(10)
    #print(dotfile_names)
    pool.map(partial(write_to_pkl, out=out_path, existing_files=existing_files), dotfile_names)
    #sent2vec_model.release_shared_mem(trained_model_path)



if __name__ == '__main__':
    main()


