## GIAAD-- Graph Injection Adversarial Attack & Defense 

Dataset extracted from **KDD Cup 2020 ML2 Track(https://www.biendata.xyz/competition/kddcup_2020_formal)**

The dataset is extracted from **Aminer(https://www.aminer.cn/)**, using articles as nodes, their titles' word embeddings as node features, and the citation relationships as adjacency. The dataset contains 659,574 nodes and 2,878,577 links, with each node equipped with a 100-dimensional node feature. The labels of the first 609,574 nodes (indexed 0..609,574) are released for training, while the labels of the rest 50,000 nodes are for evaluation. The defender must provide a model that is trained on the graph with labels of nodes in the training set, that gives robust performance against injection adversarial attacks on test set. The original dataset is the "dset" package. 

For an attack, the attacker needs to provide a set of injection nodes to attack the model, the number of injected nodes shall not exceed 500, and the degrees of each node shall not exceed 100. The injected nodes can be linked with nodes that are already on the graph, or other injection nodes. Injection nodes shall be equipped with 100-dimensional node features, and the absolute value of the features shall not exceed 2. We provide 28 best submmitted attacks in the "submission" package.

We provide the top 12 defense models by competitors, 
adversaries, cccn, ntt docomo labs, daftstone, dminer, idvl, msu-psu-dse, neutrino, simongeisler, speit, tsail, u1234x1234, they're top defense models which achieves the highest average scores against various attacks. Most of these defenses involve a filter process which filters out nodes that are likely to be injection nodes (e.g. having features with absolute values larger then 0.4, or having more than 90 degree), incorporates a feature normalization process, or adopt sampling when doing node aggregations. 

The original submissions are under Docker virtual environment.  To enable usage for people with difficulty to access docker, and also unify the working environment, **we remove all docker-related issues in the released codes and unify the environments of their codes.**

Basically, to run the code, you need to install the following packages. Make sure the cuda version and pytorch version of all packages are consistent. 

``numpy``
``scipy``
``flask``
``xgboost``
``pandas``
``sklearn``
``tqdm``
``torch``
``scikit-learn``
``torch_geometric``
``dgl``
``joblib``
``tf_slim``


If you still have problems, the following condition is confirmed to work well (assuming you have fix the dgl bug):

``python 3.7.3 cuda 10.1``

``numpy=1.16.4``

``networkx=2.3``

``torch=1.6.0+cu101``

``torch-cluster=1.5.7``

``torch-geometric=1.6.1``

``torch-scatter=2.0.5``

``torch-sparse=0.6.7``

``torch-spline-conv=1.2.0``

``torchvision=0.7.0+cu101``

``tf-slim=1.1.0``

``tensorflow=2.0.0``

``dgl-cu101=0.5.0``

``joblib=0.13.2``

``xgboost=1.2.0``

``scikit-learn=0.21.2``

``rdflib=4.2.2``

``scipy=1.5.0``

``pandas=0.24.2``

Notice a bug in dgl package:
dgl/nn/pytorch/conv/sgconv.py", line 167, 
    "if not self._allow_zero_in_degree:" this will yield you an error. 
    
To fix this, you shall change that to "if True:" 
    
Once you setup the environments, you can start your trial.

To attack, you shall create a package under ``submissions`` package with an ``adj.pkl`` representing the injection nodes' links and ``features.npy`` representing their features. We already provide the submission attack files. For example, to evaluate on the submissions of speit, all you need is to unzip it.

``cd submissions`` 

``unzip -d speit speit.zip ``

``cd ..``

``run.py`` performs a way to evaluate attacks over defense models, or defend over attack submissions. To run evaluate the attack submission of ``speit``, you can run

``python run.py --gpu 0 --mode attack --apaths speit``

You can also build up your own attacks, put it in a directory under ``submissions`` directory and evaluate your own attack on the defense models.

We show two scores, the **average score** which is the average accuracy on the test set of the attack over all defenders, and the **attack score** is the average accuracy over 3 best defenders. An attack shall generalize to all defenders so the evaluation is based on the average of 3 best defends against the attack. 

We also support evaluation of defense models. To run defend on a certain model, for example, ``speit``, run 

``python run.py --gpu 0 --mode defend --evaluation speit``

Like attacks, we also show two scores for defense: the **average score** which is the average accuracy on the test set of the defense model over all attacks,(you shall unzip all of the attacks in ``submissions`` first) and the **defend score**  is the average accuracy over 3 best attack. A defend shall be robust to generalize on all sorts of attacks so the evaluation is based on the average of 3 best attacks against the defender.

**This dataset is credit to biendata and Tsinghua-KEG team, Xinyu Guan, Xu Zou and Yicheng Zhao.**

**We welcome new submissions of attacks or defense models. If you're interested in submitting new attack/defend methods, or cooperate research on this topic, please contact** zoux18@mails.tsinghua.edu.cn 
