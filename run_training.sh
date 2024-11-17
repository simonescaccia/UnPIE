# training_step_1=IR
# python train_test.py 0 ${training_step_1}

training_step_2=LA
python3.10 train_test.py 1 ${training_step_2}

# python plot_creator.py ${training_step_1} ${training_step_2}
python3.10 plot_creator.py ${training_step_2}

