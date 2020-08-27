## GIAAD-- Graph Injection Adversarial Attack & Defense 

Dataset extracted from **KDD Cup 2020 ML2 Track(https://www.biendata.xyz/competition/kddcup_2020_formal)**

The dataset is extracted from **Aminer(https://www.aminer.cn/)**, using articles as nodes, their titles' word embeddings as node features, and the citation relationships as adjacency. The dataset contains 659,574 nodes and 2,878,577 links, with each node equipped with a 100-dimensional node feature. The labels of the first 609,574 nodes (indexed 0..609,574) are released for training, while the labels of the rest 50,000 nodes are for evaluation. The defender must provide a model that is trained on the graph with labels of nodes in the training set, that gives robust performance against injection adversarial attacks on test set. The original dataset is the "dset" package. 

For an attack, the attacker needs to provide a set of injection nodes to attack the model, the number of injected nodes shall not exceed 500, and the degrees of each node shall not exceed 100. The injected nodes can be linked with nodes that are already on the graph, or other injection nodes. Injection nodes shall be equipped with 100-dimensional node features, and the absolute value of the features shall not exceed 2. The submmitted attacks are in the "submission" package.

We provide the top 12 defense models by competitors, 
adversaries, cccn, ntt docomo labs, daftstone, dminer, idvl, msu-psu-dse, neutrino, simongeisler, speit, tsail, u1234x1234, they're top defense models which achieves the highest average scores against various attacks. Most of these defenses involve a filter process which filters out nodes that are likely to be injection nodes (e.g. having features with absolute values larger then 0.4, or having more than 90 degree), incorporates a feature normalization process, or adopt sampling when doing node aggregations. 

The original submissions are under Docker virtual environment.  **Docker (https://www.docker.com/legal/docker-terms-service)** currently does not allow people or entities on the U.S. Department of Treasury's List of Specially Designated Nationals or the U.S. Department of Commerce's Entity List to use.  **We remove all docker-related issues in the released codes and unify the environments.**

The codes are under the following environments
python 3.7.3 cuda 10.1
numpy=1.16.4
networkx=2.3
torch=1.6.0+cu101
torch-cluster=1.5.7
torch-geometric=1.6.1
torch-scatter=2.0.5
torch-sparse=0.6.7
torch-spline-conv=1.2.0
torchvision=0.7.0+cu101
tf-slim=1.1.0
tensorflow=2.0.0
dgl-cu101=0.5.0
joblib=0.13.2
xgboost=1.2.0
scikit-learn=0.21.2
rdflib=4.2.2
scipy=1.5.0
pandas=0.24.2


Once you setup the environments, you can start your trial.

To attack, you shall create a package under ``submissions`` package with an ``adj.pkl`` representing the injection nodes' links and ``features.npy`` representing their features. We already provide the submission attack files. For example, to evaluate on the submissions of speit, all you need is to unzip it.

``cd submissions
unzip -d speit speit.zip
cd ..``

``run.py`` performs a way to evaluate attacks over defense models, or defend over attack submissions. To run evaluate the attack submission of ``speit``, you 
