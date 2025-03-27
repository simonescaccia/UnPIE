training_step_1=IR
training_step_2=LA
dataset=pie
python train_test.py 1 ${training_step_1} ${dataset}
python train_test.py 1 ${training_step_2} ${dataset}
python plot_creator.py ${training_step_1} ${training_step_2}

