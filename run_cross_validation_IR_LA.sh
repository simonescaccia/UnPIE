training_step_1=IR
comma=,
training_step_2=LA
dataset=pie
python train_test.py 1 ${training_step_1} ${dataset} 5 1
python train_test.py 1 ${training_step_2} ${dataset} 5 1
python train_test.py 1 ${training_step_1} ${dataset} 5 2
python train_test.py 1 ${training_step_2} ${dataset} 5 2
python train_test.py 1 ${training_step_1} ${dataset} 5 3
python train_test.py 1 ${training_step_2} ${dataset} 5 3
python train_test.py 1 ${training_step_1} ${dataset} 5 4
python train_test.py 1 ${training_step_2} ${dataset} 5 4
python train_test.py 1 ${training_step_1} ${dataset} 5 5
python train_test.py 1 ${training_step_2} ${dataset} 5 5
python plot_creator.py ${training_step_1}${comma}${training_step_2} 5 1
python plot_creator.py ${training_step_1}${comma}${training_step_2} 5 2
python plot_creator.py ${training_step_1}${comma}${training_step_2} 5 3
python plot_creator.py ${training_step_1}${comma}${training_step_2} 5 4
python plot_creator.py ${training_step_1}${comma}${training_step_2} 5 5

