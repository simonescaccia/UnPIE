training_step_1=IR
comma=,
training_step_2=LA
dataset=psi
python train_test.py 1 ${training_step_1} ${dataset}
python train_test.py 1 ${training_step_2} ${dataset}
python plot_creator.py ${training_step_1}${comma}${training_step_2}

