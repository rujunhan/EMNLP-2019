**Author**: Rujun Han

**Date**: Nov 22nd, 2019

**Title**: Codebase for EMNLP 2019 Paper: [Joint Event and Temporal Relation Extraction with Shared Representations and Structured Prediction](https://www.aclweb.org/anthology/D19-1041.pdf) 

1. Data processinng. We have preprocessed TB-Dense and MATRES raw data using internal NLP tools at the Information Sciences Institute. These .pickle files are saved in data fold. Download glove.6B.50d.txt into other/ folder.
2. Featurize data. Run featurize_data.py and context_aggregator.py sequentially. Two folders are created: all_joint/ and all_context/. all_context contains the final files used in the model.
3. Local Model: run joint_model.py
4. Global Model: save a pipeline_joint model object from step 3 and then run joint_model_global.py.


### Code Structure (joint_model.py)

Main() --> [NNClassifier].train_epoch()

[NNClassifier].train_epoch() --> [NNClassifier]._train()

-------------------------------> [NNClassifier].predict()


1. Singletask Model. Set args.relation_weights = 0 to train event module; then set args.entity_weights = 0 to train a relation module; use both saved modules to train a pipeline end-to-end model.
> python code/joint_model.py --relation_weights 0 --relation_weights 1.0 --data_type "tbd" --batch 4 --model --"singletask/pipeline" --epoch 10
> python code/joint_model.py --relation_weights 1.0 --entity_weights 0 --data_type "tbd" --batch 4 --model --"singletask/pipeline" --epoch 10
2. Multitask Model. Set args.pipe_epoch = 1000,  args.eval_gold = True to train with gold relations only; set args.eval_gold = False to train with candidate relations generated by event module.
> python code/joint_model.py --relation_weights 1.0 --entity_weights 1.0 --data_type "tbd" --batch 4 --model --"multitask/pipeline" --eval_gold True --pipe_epoch 1000 --epoch 10
> python code/joint_model.py --relation_weights 1.0 --entity_weights 1.0 --data_type "tbd" --batch 4 --model --"multitask/pipeline" --eval_gold False --pipe_epoch 1000 --epoch 10
3. Pipeline Joint Model. Set args.pipe_epoch < args.epochs and set args.eval_gold = False to train with candidate relations generated by event module. Our paper used the output model in this step as local model.
> python code/joint_model.py --relation_weights 1.0 --entity_weights 1.0 --data_type "tbd" --batch 4 --model --"multitask/pipeline" --eval_gold False --pipe_epoch 5 --epoch 10
4. Global model. Install [Gurobi](https://www.gurobi.com/documentation/) package and run joint_model_global.py
> python code/joint_model_global.py --relation_weights 1.0 --entity_weights 1.0 --data_type "tbd" --batch 4 --model --"multitask/pipeline" --eval_gold False --pipe_epoch 5 --epoch 5