import board
import busio
import RPi.GPIO as GPIO
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
from functions_EMG import *

# Paul Xing Winter 2023

# i2c setup
i2c = busio.I2C(board.SCL, board.SDA)  # I2C interface initialisation
ads = ADS.ADS1115(i2c,data_rate=860,mode= 256)
           # ADC instance, I2C communication with external ads1115 module
...                 # maximal data rate value
...                 # single-shot mode for max frequency
chan1 = AnalogIn(ads, ADS.P0) # single ended mode for chan 1
chan2 = AnalogIn(ads, ADS.P1)         # single ended mode for chan 2

# buzzer setup
GPIO.setup(12,GPIO.OUT)
GPIO.output(12, 0)
buzzer=GPIO.PWM(12,10)


#training sequence
#training_label_acq = np.asarray([0,0,1,0,0,0,-1,0,0,0,1,0,0,0,-1,0,0,0,1,0,0,0,1,0,0,0,-1,0,0,0,-1,0,0,0]) # 1 = flexion, -1 = extension

training_label = np.asarray([0,1,0,0,-1,0,0,1,0,0,-1,0,0,1,0,0,1,0,0,-1,0,0,-1,0,0,1,0,0,-1,0]) # 1 = flexion, -1 = extension

# general sampling parameters
ntile = 20
window_size = 50
# evaluation window length (samples)
number_window_training = len(training_label )*ntile           # number of windows to sample for training data set
number_window_testing = 500             # number of windows to sample for online testing


#adjust sequence vector size
#training_label_acq =training_label_acq[:,np.newaxis]
#training_label_acq  = np.tile(training_label_acq ,(1,ntile)).flatten()
training_label = training_label[:,np.newaxis]
training_label  = np.tile(training_label ,(1,ntile)).flatten()


# EMG data acquisition for training - online mode
training_file_name = create_new_sampling_file("data/", "training_acquisition")
acquire_training_dataset(chan1, chan2, window_size, number_window_training, training_file_name,training_label )

# buzzer.start(50)
# visual validation of data acquisition
visualize_sampling(training_file_name,window_size,training_label )
# buzzer.stop()
# classifier training (offline training)
classifier = train_classifier(training_file_name, window_size,training_label,max_iter=1000, eta=1, mu=0 )

# buzzer control (online test)
testing_file_name = create_new_sampling_file("data/", "testing_acquisition")
final_labels = test_classifier(classifier, chan1, chan2, window_size, number_window_testing, buzzer, testing_file_name)


buzzer.stop