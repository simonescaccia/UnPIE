training_step_1=vd_3dresnet_IR
training_step_2=vd_3dresnet
python train_test.py 2 ${training_step_2}

python plot_creator.py ${training_step_1} ${training_step_2}