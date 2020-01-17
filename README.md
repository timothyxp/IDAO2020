all launch files in directory bin
all configs in configuration dir
for each config we have experiment name -
 directory with its name creating in data directory
 and there we store all data
 
lets see how we can create something 

you can create your own model in pipeline.models 
and extends from ModelBase class in pipeline.models.base 
and do all in abstract methods

you can create your own data builder in pipeline.data 
ans extends from DataBuilderBase class in pipeline.data.base

after you should add your class in configuration 

and you should put train.csv and test.csv 
in data/{experiment_name} folder

after you can launch something like
`PYTHONPATH=. python3 bin/{process} /path/to/config`

or `export PYTHONPATH=.` and after use without it

for example if we want build training data so we should enter this

`python3 bin/build_training_data.py configuration/base.py`

info and higher logs write to console
all application logs write to debug.log
and all logs write to all_debug.log

you can watch it like `tail -f debug.log` 