training_step_1=IR
training_step_2=LA
python train_test.py 1 ${training_step_1}
python train_test.py 1 ${training_step_2}
python plot_creator.py ${training_step_1} ${training_step_2}

# training_step_2=LA
# python train_test.py 1 ${training_step_2}
# python plot_creator.py ${training_step_2}

# training_step_1=IR
# python train_test.py 1 ${training_step_1}
# python plot_creator.py ${training_step_1}

