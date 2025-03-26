training_step_1=IR
dataset=psi
python train_test.py 1 ${training_step_1} ${dataset}
python plot_creator.py ${training_step_1} ${dataset}