Code for “A Multitask Graph Neural Network Framework for AMES Mutagenicity
Prediction” Abigail E. Teitgen, Eugenia Ulzurrun, Nuria E. Campillo, Eduardo R. Hernandez

Summary of code: 

- DataBase_AMES: contains preprocessed XYZ files with 3D coordinates for each molecule, used as input to graph_maker.py to generate graphs
- data.csv: dataset used for all model training and analysis
- graph_maker_sample.yml: input script for generating graphs
- train_sample.yml: input script for training model (to run, specify —input_file (.yml format) and —output_dir)
- graph_maker.py: main script for generating graphs (to run, specify .yml input file)
- GNN_MTL_GPU.py: main script for running GNN MTL model
- GNN_explainer_analysis_final.py: main script for running GNN explainer analysis (to run, specify —input_file (.yml format) and —output_dir)