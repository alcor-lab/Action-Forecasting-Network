#[TENSORFLOW]
TF_CPP_MIN_LOG_LEVEL = '3'
inter_op_parallelism_threads=7
allow_growth = True

#[Train]
c3d_ucf_weights = "sports1m_finetuning_ucf101.model"
Batch_size = 20
frames_per_step = 6
window_size = 1
load_previous_weigth = True
load_pretrained_weigth = True
model_filename = './checkpoint/Net_weigths.model'
tot_steps = 1000000
processes = 20
tasks = 10
reuse_HSM = True
Action = True

#[Network]
learning_rate = 0.1
gradient_clipping_norm = 1.0
c3d_dropout = 0.6
preLstm_dropout = 0.6
Lstm_dropout = 0.6
input_channels = 7
pre_size = 1024
C3d_Output_features = 256
lstm_units = C3d_Output_features

#[Batch]
out_H = 112
out_W = 112
hidden_states_dim = lstm_units
current_accuracy = 0
snow_ball = False
snow_ball_step_count = 0
snow_ball_per_class = 2000
snow_ball_classes = 3
op_input_width = 368
op_input_height = 368
show_pic = False

#[Dataset]
validation_fraction = 0.2

#[Annotation]
rebuild = True
limit_classes = False
classes_to_use = ['milk', 'caffe']
ocado_annotation = 'dataset/ocado.json'
breakfast_annotation = 'dataset/breakfast.json'
acitivityNet_annotation = 'dataset/activity_net.v1-3.min.json'
breakfast_path = 'dataset/Video/BreakfastII_15fps_qvga_sync'
dataset = 'Breakfast'
breakfast_fps = 15
