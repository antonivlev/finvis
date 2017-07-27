# exec(open("C:/PythonStuff/my_har.py").read())

exec(open("C:/PythonStuff/make_wins.py").read())
# import numpy as np
# import tensorflow as tf
# import random
# from numpy.fft import fft, ifft

# data (num_ex, win_width, chans)
# labels (num_ex, 2)
def plot_data(data, labels, walk_f_name, null_f_name, start=0, fin=15):
	'''
	stuff 
	'''
	i = 1
	num_chans = data.shape[2]

	walk_indices = indices_of_label(labels, 1)
	print("walk indices: ", walk_indices)
	null_indices = indices_of_label(labels, 2)
	print("null indices: ", null_indices)


	for example in range(start, fin):
		walk_ind = walk_indices[ random.randint(0, len(walk_indices)-1) ]
		for chan in range(0, num_chans):
			plt.ylim(-10, 30)
			plt.subplot(fin-start, num_chans, i)
			# plt.axis('off')
			plt.plot(data[walk_ind, :, chan], linewidth=0.3)
			# plt.plot(fft_sequence(data[walk_ind, :, chan]), linewidth=0.3)
			
			i += 1
	
	plt.savefig(walk_f_name, bbox_inches='tight', dpi=1000)
	plt.show()

	i = 1
	plt.clf()
	plt.cla()
	for example in range(start, fin):
		null_ind = null_indices[ random.randint(0, len(null_indices)-1) ]
		for chan in range(0, num_chans):
			plt.ylim(-10, 30)
			plt.subplot(fin-start, num_chans, i)
			#plt.axis('off')
			plt.plot(data[null_ind, :, chan], linewidth=0.3)
			#plt.plot(fft_sequence(data[null_ind, :, chan]), linewidth=0.3)
			i += 1
	
	plt.savefig(null_f_name, bbox_inches='tight', dpi=1000)
	plt.show()



# ======================== #
# 		 DATA PREP         #
# ======================== #
data_dir = "C:/Datasets/Gavin0/"
file_list = [
	# "Myo_Accel_14_12_16.txt",
	# "Myo_Gyro_14_12_16.txt",
	"Phone_Accel_14_12_16.txt",
	"Phone_Gyro_14_12_16.txt",
	"Watch_Accel_14_12_16.txt",
	"Watch_Gyro_14_12_16.txt"
]


# covariance matrix phone accel
#	x 			y 			  z
#x  2.48976486, -0.87812394,  0.12187754,
#y -0.87812394,  5.29026461,  0.93694656,
#z  0.12187754,  0.93694656,  2.48387574


pax = np.array(file_to_dict(data_dir+file_list[0])["x"])
pay = np.array(file_to_dict(data_dir+file_list[0])["y"])
paz = np.array(file_to_dict(data_dir+file_list[0])["z"])

def get_normal(mu, sigma, num_samples, rng):
	a = np.zeros(num_samples)
	xvals = np.linspace(rng[0], rng[1], num_samples)
	for i in range(0, num_samples):
		x = xvals[i]
		a[i] = 1/np.sqrt(2*np.pi*sigma**2) * np.e**-((x-mu)**2/(2*sigma**2))
	return xvals, a


data, all_labels = make_data_label_arrays(data_dir, file_list)
#data, labels = redist_data(data, labels)

print("final data shapes: ")
print("\tdata: ", data.shape)
print("\tall_labels: ", all_labels.shape)

data, reduced_labels = keep_labels(data, all_labels, [1, 2])
print("new data shapes: ")
print("\tdata: ", data.shape)
print("\treduced_labels: ", reduced_labels.shape)

feat_vecs = get_feature_vecs(data)
print("feat_vecs shape: ")
print("\t", feat_vecs.shape)

rfeat_vecs = reduce_feature_vecs(feat_vecs)
print("reduced vecs:")
print("\t", rfeat_vecs.shape)

plot_feature_vecs(rfeat_vecs, reduced_labels)