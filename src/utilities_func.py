import numpy as np
import math, copy
import os
import pandas
from scipy.io.wavfile import read, write
from scipy.fftpack import fft
from scipy.signal import iirfilter, butter, filtfilt, lfilter
from shutil import copyfile
from librosa.effects import split


tol = 1e-14    # threshold used to compute phase

INT16_FAC = (2**15)-1
INT32_FAC = (2**31)-1
INT64_FAC = (2**63)-1
norm_fact = {'int16':INT16_FAC, 'int32':INT32_FAC, 'int64':INT64_FAC,'float32':1.0,'float64':1.0}

def isPower2(num):
    #taken from Xavier Serra's sms tools
    """
    Check if num is power of two
    """
    return ((num & (num - 1)) == 0) and num > 0

def wavread(file_name):
    #taken from Xavier Serra's sms tools
    '''
    read wav file and converts it from int16 to float32
    '''
    sr, samples = read(file_name)
    samples = np.float32(samples)/norm_fact[samples.dtype.name] #float conversion

    return sr, samples

def wavwrite(y, fs, filename):
    #taken from Xavier Serra's sms tools
    """
    Write a sound file from an array with the sound and the sampling rate
    y: floating point array of one dimension, fs: sampling rate
    filename: name of file to create
    """
    x = copy.deepcopy(y)                         # copy array
    x *= INT16_FAC                               # scaling floating point -1 to 1 range signal to int16 range
    x = np.int16(x)                              # converting to int16 type
    write(filename, fs, x)

def dftAnal(x, w, N):
    #taken from Xavier Serra's sms tools
	"""
	Analysis of a signal using the discrete Fourier transform
	x: input signal, w: analysis window, N: FFT size
	returns mX, pX: magnitude and phase spectrum
	"""

	if not(isPower2(N)):                                 # raise error if N not a power of two
		raise ValueError("FFT size (N) is not a power of 2")

	if (w.size > N):                                        # raise error if window size bigger than fft size
		raise ValueError("Window size (M) is bigger than FFT size")

	hN = (N/2)+1                                            # size of positive spectrum, it includes sample 0
	hM1 = int(math.floor((w.size+1)/2))                     # half analysis window size by rounding
	hM2 = int(math.floor(w.size/2))                         # half analysis window size by floor
	fftbuffer = np.zeros(N)                                 # initialize buffer for FFT
	w = w / sum(w)                                          # normalize analysis window
	xw = x*w                                                # window the input sound
	fftbuffer[:hM1] = xw[hM2:]                              # zero-phase window in fftbuffer
	fftbuffer[-hM2:] = xw[:hM2]
	X = fft(fftbuffer)                                      # compute FFT
	absX = abs(X[:hN])                                      # compute ansolute value of positive side
	absX[absX<np.finfo(float).eps] = np.finfo(float).eps    # if zeros add epsilon to handle log
	mX = 20 * np.log10(absX)                                # magnitude spectrum of positive frequencies in dB
	X[:hN].real[np.abs(X[:hN].real) < tol] = 0.0            # for phase calculation set to 0 the small values
	X[:hN].imag[np.abs(X[:hN].imag) < tol] = 0.0            # for phase calculation set to 0 the small values
	pX = np.unwrap(np.angle(X[:hN]))                        # unwrapped phase spectrum of positive frequencies

	return mX, pX

def stftAnal(x, w, N, H) :
    #taken from Xavier Serra's sms tools
	"""
	Analysis of a sound using the short-time Fourier transform
	x: input array sound, w: analysis window, N: FFT size, H: hop size
	returns xmX, xpX: magnitude and phase spectra
	"""
	if (H <= 0):                                   # raise error if hop size 0 or negative
		raise ValueError("Hop size (H) smaller or equal to 0")

	M = w.size                                      # size of analysis window
	hM1 = int(math.floor((M+1)/2))                  # half analysis window size by rounding
	hM2 = int(math.floor(M/2))                      # half analysis window size by floor
	x = np.append(np.zeros(hM2),x)                  # add zeros at beginning to center first window at sample 0
	x = np.append(x,np.zeros(hM2))                  # add zeros at the end to analyze last sample
	pin = hM1                                       # initialize sound pointer in middle of analysis window
	pend = x.size-hM1                               # last sample to start a frame
	w = w / sum(w)                                  # normalize analysis window
	while pin<=pend:                                # while sound pointer is smaller than last sample
		x1 = x[pin-hM1:pin+hM2]                       # select one frame of input sound
		mX, pX = dftAnal(x1, w, N)                # compute dft
		if pin == hM1:                                # if first frame create output arrays
			xmX = np.array([mX])
			xpX = np.array([pX])
		else:                                         # append output to existing array
			xmX = np.vstack((xmX,np.array([mX])))
			xpX = np.vstack((xpX,np.array([pX])))
		pin += H                                      # advance sound pointer

	return xmX, xpX

def find_longer_audio(input_folder):
    '''
    look for all .wav files in a folder and
    return the duration (in samples) of the longest one
    '''
    contents = os.listdir(input_folder)
    file_sizes = []
    for file in contents:
        if file[-3:] == "wav": #selects just wav files
            file_name = input_folder + '/' + file   #construct file_name string
            try:
                sr, samples = wavread(file_name)  #read audio file
                #samples = strip_silence(samples)
                file_sizes.append(len(samples))
            except ValueError:
                pass
    max_file_length = max(file_sizes)

    return max_file_length

def onehot(value, range):
    '''
    int to one hot vector conversion
    '''
    one_hot = np.zeros(range)
    one_hot[value] = 1

    return one_hot

def strip_silence(input_vector, threshold=35):
    split_vec = split(input_vector, top_db = threshold)
    onset = split_vec[0][0]
    offset = split_vec[-1][-1]
    cut = input_vector[onset:offset]

    return cut

def preemphasis(input_vector, fs):
    '''
    2 simple high pass FIR filters in cascade to emphasize high frequencies
    and cut unwanted low-frequencies
    '''
    #first gentle high pass
    alpha=0.5
    present = input_vector
    zero = [0]
    past = input_vector[:-1]
    past = np.concatenate([zero,past])
    past = np.multiply(past, alpha)
    filtered1 = np.subtract(present,past)
    #second 30 hz high pass
    fc = 100.  # Cut-off frequency of the filter
    w = fc / (fs / 2.) # Normalize the frequency
    b, a = butter(8, w, 'high')
    output = filtfilt(b, a, filtered1)

    return output

def CCC(y_true, y_pred):
    '''
    Lin's Concordance correlation coefficient: https://en.wikipedia.org/wiki/Concordance_correlation_coefficient
    Accepting tensors as input

    '''

    import keras.backend as K
    # covariance between y_true and y_pred
    N = K.int_shape(y_pred)[-1]
    s_xy = 1.0 / (N - 1.0 + K.epsilon()) * K.sum((y_true - K.mean(y_true)) * (y_pred - K.mean(y_pred)))
    # means
    x_m = K.mean(y_true)
    y_m = K.mean(y_pred)
    # variances
    s_x_sq = K.var(y_true)
    s_y_sq = K.var(y_pred)

    # condordance correlation coefficient
    ccc = (2.0*s_xy) / (s_x_sq + s_y_sq + (x_m-y_m)**2)

    return ccc


def find_mean_std(input_folder):
    annotations = os.listdir(input_folder)
    sequence = []
    for datapoint in annotations:
        annotation_file = input_folder + '/' + datapoint
        ann = pandas.read_csv(annotation_file)
        ann = ann.values.T[0]
        sequence = np.concatenate((sequence,ann))
    mean = np.mean(sequence)
    std = np.std(sequence)
    return mean, std

def f_trick(input_sequence, ref_mean, ref_std):
    mean = np.mean(input_sequence)
    std = np.std(input_sequence)
    num = np.multiply(ref_std, np.subtract(input_sequence, mean))
    output = np.divide(num, std)
    output = np.add(output, ref_mean)

    return output

def gen_fake_annotations(frames_count, output_folder):
    with open(frames_count) as f:
        content = f.readlines()
    for line in content:
        split = line.split('-')
        name = split[0].replace(' ', '')
        file_name = output_folder + '/' + name
        len = int(split[-1].split(' ')[1])
        valence = np.zeros(len)
        temp_dict = {'valence':valence}
        temp_df = pandas.DataFrame(data=temp_dict)
        temp_df.to_csv(file_name, index=False)

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_lowpass_filter_bidirectional(data, cutoff=0.1, fs=25, order=1):
    y_first_pass = butter_lowpass_filter(data[::-1].flatten(), cutoff, fs, order)
    y_second_pass = butter_lowpass_filter(y_first_pass[::-1].flatten(), cutoff, fs, order)
    return y_second_pass


def notch_filter(band, cutoff, ripple, rs, sr=44100, order=2, filter_type='cheby2'):
    #creates chebyshev polynomials for a notch filter with given parameters
    nyq  = sr/2.0
    low  = cutoff - band/2.0
    high = cutoff + band/2.0
    low  = low/nyq
    high = high/nyq
    w0 = cutoff/(sr/2)
    a, b = iirfilter(order, [low, high], rp=ripple, rs=rs, btype='bandstop', analog=False, ftype=filter_type)

    return a, b

def subsample_mean(input_vector, chunk_size = 10):
    '''
    subsample a vector computing the mean value for every chunk
    '''
    #filter
    input_vector = butter_lowpass_filter(input_vector, cutoff=0.4, fs=25, order=8)
    if len(input_vector) % chunk_size != 0:
        raise ValueError('chunk_size must be divisor of len(input_vector)')
    pointer = 0
    out_vector = []
    while pointer != len(input_vector):
        chunk = input_vector[pointer:pointer+chunk_size]
        chunk_mean = np.mean(chunk)
        out_vector.append(chunk_mean)
        pointer += chunk_size
    out_vector = np.array(out_vector)
    return out_vector

def subsample_dataset(input_dataset, chunk_size):
    '''
    apply subsample_mean to a dataset
    '''
    out_matrix = []
    for data_point in input_dataset:
        sub_sampled = subsample_mean(data_point, chunk_size)
        out_matrix.append(sub_sampled)
    out_matrix = np.array(out_matrix)
    return out_matrix

def split_folder(input, out1, out2, perc):
    contents = os.listdir(input)
    length = len(contents)
    split = int(length * perc)
    for i in range(length):
        in_file = input + '/' + contents[i]
        if i <= split:
            out_file = out1 + '/' + contents[i]
        else:
            out_file = out2 + '/' + contents[i]
        copyfile(in_file, out_file)

def save_data(dataloader, epoch, gen_figs_path, gen_sounds_path, save_figs, save_sounds):
    data_gen = []
    data_truth = []
    if save_figs or save_sounds:
        if epoch % save_items_epochs == 0: #save only every n epochs
            #create folders
            if save_figs:
                curr_figs_path_train = os.path.join(gen_figs_path, 'training' , 'epoch_'+str(epoch))
                curr_figs_path_test = os.path.join(gen_figs_path, 'test' , 'epoch_'+str(epoch))
                if not os.path.exists(curr_figs_path_train):
                    os.makedirs(curr_figs_path_train)
                if not os.path.exists(curr_figs_path_test):
                    os.makedirs(curr_figs_path_test)
            if save_sounds:
                curr_sounds_path_train = os.path.join(gen_sounds_path, 'training' , 'epoch_'+str(epoch))
                curr_sounds_path_test = os.path.join(gen_sounds_path, 'test' , 'epoch_'+str(epoch))
                curr_orig_path_training = os.path.join(gen_sounds_path, 'training', 'originals')
                curr_orig_path_test = os.path.join(gen_sounds_path, 'test', 'originals')
                if not os.path.exists(curr_sounds_path_train):
                    os.makedirs(curr_sounds_path_train)
                if not os.path.exists(curr_sounds_path_test):
                    os.makedirs(curr_sounds_path_test)
                if not os.path.exists(curr_orig_path_training):
                    os.makedirs(curr_orig_path_training)
                if not os.path.exists(curr_orig_path_test):
                    os.makedirs(curr_orig_path_test)


            for i, (sounds, truth) in enumerate(dataloader):
                if len(figs_truth) <= save_figs_n:
                    sounds = sounds.to(device)
                    truth = truth.numpy()
                    #compute predictions
                    if use_complete_net:
                        outputs, mu, logvar = model(sounds)
                    else:
                        mu, logvar = encoder(sounds)
                        z = reparametrize(mu, logvar)
                        outputs = decoder(z)
                    outputs = outputs.cpu().numpy()
                    #concatenate predictions
                    for single_sound in outputs:
                        if features_type == 'waveform':
                            data_gen.append(single_sound)
                        elif features_type == 'spectrum':
                            data_gen.append(single_sound.reshape(single_sound.shape[-2], single_sound.shape[-1]))
                    for single_sound in truth:
                        if features_type == 'waveform':
                            data_gen.append(single_sound)
                        elif features_type == 'spectrum':
                            data_gen.append(single_sound.reshape(single_sound.shape[-2], single_sound.shape[-1]))
                    if save_sounds:
                        for single_sound in outputs:
                            tr_preds.append(single_sound)
                else:
                    break
            #save items
            for i in range(save_items_n):
                if save_figs:
                    fig_name = 'gen_' + str(i) + '.png'
                    fig_path = os.path.join(curr_figs_path_train, fig_name)
                    plt.subplot(211)
                    plt.pcolormesh(data_gen[i].T)
                    plt.title('gen')
                    plt.subplot(212)
                    plt.pcolormesh(data_truth[i].T)
                    plt.title('original')
                    plt.savefig(fig_path)
                    plt.close()
                if save_sounds:
                    #generated
                    sound_name = 'gen_' + str(i) + '.wav'
                    sound_path = os.path.join(curr_sounds_path_train, sound_name)
                    sound = data_gen[i]
                    sound = sound.flatten()
                    sound = np.divide(sound, np.max(sound))
                    sound = np.multiply(sound, 0.8)
                    uf.wavwrite(sound, SR, sound_path)
                    #originals only for epoch 0
                    if epoch = 0:
                        orig_name = 'orig_' + str(i) + '.wav'
                        orig_path = os.path.join(curr_orig_path_training, sound_name)
                        orig = data_truth[i]
                        orig = orig.flatten()
                        orig = np.divide(orig, np.max(orig))
                        orig = np.multiply(orig, 0.8)
                        uf.wavwrite(orig, SR, orig_path)
            print ('')
            if save_figs:
                print ('Generated figures saved')
            if save_sounds:
                print ('Generated sounds saved')
