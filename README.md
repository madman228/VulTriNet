# VulTriNet: A Software Vulnerability Detection Method Based on Tri-channel Network
A new vulnerability detection approach，The code implementation is mainly based on VulCNN: https://github.com/CGCL-codes/VulCNN, you can refer to it for basic implementation (especially PDG).

Software vulnerabilities represent a critical concern in the domain of cybersecurity. As vulnerability patterns become increasingly complex, the security community requires advanced detection methods to comprehensively analyze vulnerability characteristics. In recent years, some researchers take source codes as text using natural language processing (NLP) techniques. Subsequent advancements transformed programs into intermediate representations, such as program dependency graph (PDG), utilizing graph neural network (GNN) for vulnerability pattern learning. Due to the impressive performance of both approaches, we developed a hybrid analysis model that strengthens vulnerability detection capabilities by simultaneously maintaining contextual code understanding and structural relationship awareness. We proposed a novel vulnerability detection method based on a tri-channel network (VulTriNet). It integrates two graph-based and one textual code representation using three distinct methods to transform functions into multiple forms. Then, VulTriNet generates corresponding embedding vectors for these transformed representations, which are merged into a three-channel feature matrix. Finally, we design a CNN model integrated with attention mechanisms to improve the capability of detecting vulnerabilities. We applied VulTriNet and conducted experiments on multiple vulnerability datasets.

# Design of the Approach

![overview](https://github.com/madman228/VulTriNet/blob/b90d3839eb85b2462f77c9f66b102a709fe0ccd4/overview_twolayers.jpg)

This section presents **VulTriNet**, an efficient and novel model for source code vulnerability detection. As shown in Figure, VulTriNet consists of three steps: function transformation, embedding vector generation, and classification. In **Step 1**, the function transformation explains how the function is transformed. Here, the code is cleaned up to produce a text representation, which is then further normalized to generate the function's AST and PDG. In **Step 2**, embedding vector generation converts the function's representation into a vector. For the function's AST, we apply depth-first traversal (DFT) and use Word2Vec to generate the vector representation. The PDG is embedded using Sent2Vec, while the code text is processed through the CodeBERT model. In **Step 3**, we input the generated tri-channel image into the CNN for classification. The classification performs binary classification to determine whether the function contains any vulnerabilities.

# The Step to Execute this
The CodeBERT model can download from:https://huggingface.co/. <br> 
The code to generate AST can down load from:https://github.com/fabsx00/codesensor or Joern. <br> 
Here,our model is built on VulCNN(https://github.com/CGCL-codes/VulCNN),the environment is similar to it. <br> 

**Step 1: Code normalization**
Normalize the code with normalization.py (This operation will overwrite the data file, please make a backup)
``` 
python ./normalization.py -i ./data/sard
```
<br> 

**Step 2: Generate pdgs with the help of joern**
Prepare the environment refering to: joern,here we use 1.1.1000

```
# first generate .bin files
python joern_graph_gen.py  -i ./data/sard/Vul -o ./data/sard/bins/Vul -t parse
python joern_graph_gen.py  -i ./data/sard/No-Vul -o ./data/sard/bins/No-Vul -t parse 
``` 

```
# then generate pdgs (.dot files)
python joern_graph_gen.py  -i ./data/sard/bins/Vul -o ./data/sard/pdgs/Vul -t export -r pdg
python joern_graph_gen.py  -i ./data/sard/bins/No-Vul -o ./data/sard/pdgs/No-Vul -t export -r pdg
```
<br> 

**Step 3: Train a sent2vec model**
**Refer to sent2vec**
``` 
./fasttext sent2vec -input ./data/data.txt -output ./data/data_model -minCount 8 -dim 128 -epoch 9 -lr 0.2 -wordNgrams 2 -loss ns -neg 10 -thread 20 -t 0.000005 -dropoutK 4 -minCountLabel 20 -bucket 4000000 -maxVocabSize 750000 -numCheckPoints 10
``` 
<br> 

**Step 4: Generate images from the pdgs**
Generate Images from the pdgs with ImageGeneration.py, this step will output a .pkl file for each .dot file.
``` 
python ImageGeneration.py -i ./data/sard/pdgs/Vul -o ./data/sard/outputs/Vul -m ./data/data_model.bin
python ImageGeneration.py -i ./data/sard/pdgs/No-Vul -o ./data/sard/outputs/No-Vul  -m ./data/data_model.bin
```

<br> 

**Step 5: Integrate the data and divide the training and testing datasets**
Integrate the data and divide the training and testing datasets

```
# n denotes the number of kfold, i.e., n=10 then the training set and test set are divided according to 9:1 and 10 sets of experiments will be performed
python split_data.py -i ./data/sard/outputs -o ./data/sard/pkl -n 5
```
<br> 

**Step 6: Train with CNN**
``` 
python main.py -i ./data/sard/pkl
``` 
