# Self-training for Knowledge Representation Models



## Dependencies

We are using a 3090 GPU 24GB with Torch version 1.12. Please refer to the requirements.txt file for detailed versions of the virtual environment.

- Compatible with PyTorch 1.12 and Python 3.7.
- Dependencies can be installed through requirements.txt.
- Commands for reproducing in reported results on ConvE:

    1.The training model was run on the WN18RR dataset as follows:
    
    nohup python run.py -data WN18RR -encoder Embed -decoder ConvE -num_filters 200 -embed_dim 200  -decoder_feat_drop 0.2 -decoder_hid_drop 0.3 -batch_size 256 -gpu 0 -smooth 0.05 -lr 0.003 -ker_sz 7 >/dev/null 2>wn1.log &
    
    2.The training model was run on the FB15k-237 dataset as follows:
    
    nohup python run.py -encoder Embed -decoder ConvE -num_filters 200 -embed_dim 200  -decoder_feat_drop 0.2 -decoder_hid_drop 0.3 -batch_size 256 -gpu 0 -smooth 0.1 -lr 0.0001 -ker_sz 7 >/dev/null 2>f1.log &
    
    For model prediction：
    
    python run.py -encoder Embed -decoder ConvE -predict -pretrain_path [pretrain model path] 


The TransE model：

    nohup python transe.py

The parameters for obtaining the best model through training will be a record in the log file.