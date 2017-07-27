def fft_sequence(seq):
	L = len(seq)
	Y = fft(seq)
	P2 = abs(Y/L)
	P1 = P2[1 : int(L/2+1)]
	P1[1 : -2] = 2*P1[1 : -2]

	#f = np.arange(0, int(L/2))/L
	return P1
# Testing fft_sequence
# t = np.arange(0, 1000)/1000
# S = 0.7*sin(2*pi*50*t) + sin(2*pi*120*t)

# plt.plot(S)
# plt.show()
# x, y = fft_sequence(S)
# plt.plot(y)
# plt.show()



def file_to_dict(in_file_path):
	'''
	Takes path to file of format 

	x0 y0 z0 t0
	x1 y1 z1 t1
	...

	returns {
		"x":[...],
		"y":[...],
		"z":[...],
		"t":[...]	
	}
	'''
	f = open(in_file_path, "r")
	contents = f.readlines()
	n = len(contents)

	out = {"x":[], "y":[], "z":[], "t":[]}

	for line in contents:
		split_line = line.split()
		if (float(split_line[3]) != 0):
			out["x"].append( float(split_line[0]) )
			out["y"].append( float(split_line[1]) )
			out["z"].append( float(split_line[2]) )
			out["t"].append( float(split_line[3]) )
	f.close()
	return out

'''
Takes path to file of format

start_t finish_t label
...

returns [
	[start, finish, integer_label],
	...
]
'''
def labels_to_list(in_file_path):
	f = open(in_file_path, "r")
	contents = f.readlines()
	n = len(contents)

	out = []

	for line in contents:
		split_line = line.split()
		out.append([
			float(split_line[0]),
			float(split_line[1]),
			float(split_line[2])
		])
	f.close()
	return out

# Linear interpolation
# a = [x0, x, x1]
# b = [y0,    y1]
# Returns  y  between y0 and y1
def lin_inter(a, b):
	bigA = a[2] - a[0]
	leftA = a[1] - a[0]
	bigB = b[1] - b[0]
	return b[0] + bigB*(leftA/bigA)


# Given two sequences, samples any number of points from them.
# Linearly interpolates for points between samples 
#
# xvals - list of time stamps
# yvals - list of sensor values
# both are ordered, e.g. 5th timetsamp belongs with 5th sensor value
#
# startx - the xvalue (time) we want to start sampling from. Does not have to be contained in xvals.
# endx - (time) sample until this value
# if there is no data for a sample xvalue, outputs zero
#
# num_vals - number of values in sample
#
# returns tuple (sampled_ys, sampled_xs, time between samples)
def get_sample(xvals, yvals, startx, endx, num_vals):
	step = (endx - startx)/num_vals
	outy = []
	outx = []
	x = startx
	i = 0

	try:
		while x <= endx:
			if x > xvals[i]:
				i = i+1
			elif x <= xvals[i]:
				if i == 0:
					outy.append(yvals[i])
				else:
					outy.append( lin_inter( 
						[xvals[i-1], x, xvals[i]], 
						[yvals[i-1], yvals[i]] 
					) )
				outx.append(x)
				x += step
	except IndexError:
		print("index error occured!")
		print("---xvals yvals length: ", len(xvals))
		print("---i: ", i)
		print("---x: ", x)
		print("---step: ", step)
		print("---endx + step: ", endx+step)
	return outy, outx, step


'''
Turns each window into a feature vec. Features:
    Average in each dimension [1 * n_dim]
    Standard deviation in each dimension [1 * n_dim]
    Average Absolute Difference  [1 * n_dim]
    Average Resultant Acceleration [1 * n_dim?]

data - array shape (num_examples, win_width, n_dim)

returns: array shape (num_examples, num_features)
'''
def get_feature_vecs(data):
	num_dims = data.shape[2]
	print("num dims in vec: ", num_dims)
	# print("in get_feature_vecs")
	num_examples = data.shape[0]
	feat_vecs = np.zeros(shape=(num_examples, num_dims*4))
	for i in range(0, num_examples):
		example = data[i]
		vec = np.zeros(shape=(num_dims*4))
		for dim in range(num_dims):
			# print("\texample[: , ",dim,"]", example[:, dim])
			vec[dim*4  ] = np.average(example[:, dim])
			vec[dim*4+1] = np.std(example[:, dim])
			vec[dim*4+2] = absolute_diff(example[:, dim])
			vec[dim*4+3] = np.sum(example[:, dim])
		
		feat_vecs[i] = vec

	return feat_vecs

'''
feat vecs - (num_examples, num_features)
'''
new_basis = 0
f_vecs = 0
new_data = 0
covar = 0
gpca = 0
def reduce_feature_vecs(feat_vecs):
	pca = PCA(n_components=3)
	pca.fit(feat_vecs)
	out = np.matmul(pca.components_, feat_vecs.transpose())

	global new_basis, f_vecs, new_data, covar, gpca 
	new_basis = pca.components_
	f_vecs = feat_vecs
	new_data = out.transpose()
	covar = np.matmul(new_data, new_data.transpose())
	gpca = pca

	return out.transpose()

def plot_feature_vecs(feat_vecs, labels):
	# plt.subplot(122, projection="3d")
	# ax = plt.gca()
	ax = plt.figure().add_subplot(111, projection='3d')

	data = feat_vecs.transpose()
	xs1, ys1, zs1 = [], [], []	
	xs2, ys2, zs2 = [], [], []
	xs3, ys3, zs3 = [], [], []
	for i in range(len(labels)):
		if (labels[i].nonzero()[0][0] == 1):
			xs1.append(data[0][i])
			ys1.append(data[1][i])
			zs1.append(data[2][i])
		elif (labels[i].nonzero()[0][0] == 2):
			xs2.append(data[0][i])
			ys2.append(data[1][i])
			zs2.append(data[2][i])
		else:
			xs3.append(data[0][i])
			ys3.append(data[1][i])
			zs3.append(data[2][i])

	ax.scatter(xs1, ys1, zs1, c="b", marker="o")
	# xs = np.array(xs2+xs1)
	# ys = np.array(ys2+ys1)
	# zs = np.array(zs2+zs1)
	ax.scatter(xs2, ys2, zs2, c="r", marker="o")
	ax.scatter(xs3, ys3, zs3, c="g", marker="o")

	print("n points: ", len(feat_vecs))
	plt.show()

	# def randrange(n, vmin, vmax):
	#     '''
	#     Helper function to make an array of random numbers having shape (n, )
	#     with each number distributed Uniform(vmin, vmax).
	#     '''
	#     return (vmax - vmin)*np.random.rand(n) + vmin

	# fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d')

	# n = 500

	# # For each set of style and range settings, plot n random points in the box
	# # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
	# for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', 'o', -30, -5)]:
	#     xs = randrange(n, 23, 32)
	#     ys = randrange(n, 0, 100)
	#     zs = randrange(n, zlow, zhigh)
	#     ax.scatter(xs, ys, zs, c=c, marker=m)

	# ax.set_xlabel('X Label')
	# ax.set_ylabel('Y Label')
	# ax.set_zlabel('Z Label')

	# plt.show()


def absolute_diff(seq):
	s = 0
	for x1 in seq:
		for x2 in seq:
			s += (x1 - x2)
	return s / len(seq)**2


'''
Removes any labels not in label_list from data

data - array shaped(num_examples, win_width, dim)
label_list - integer labels like [1, 2, 4]

returns data, labels with only label_list labels
'''
def keep_labels(data, labels, label_list):
	indices = []
	for label in label_list:
		indices = indices + indices_of_label(labels, label)
	#print(indices)
	new_data = np.array([data[i] for i in indices])
	new_labels = np.array([labels[i] for i in indices]) 
	return new_data, new_labels 

def indices_of_label(labels, in_label):
	out = [];
	i = 0
	for one_hot_label in labels:
		integer_label = one_hot_label.nonzero()[0][0]
		if (integer_label) == in_label:
			out.append(i)
		i += 1
	return out





# Helper
def which_region(win_start, win_end, labeled_regions):
	out = None
	for region in labeled_regions:
		if (region[0] <= win_start <= region[1]) or (region[0] <= win_end <= region[1]):
			out = region[2]
	return out


def int_to_onehot(x, total=10):
	out = np.zeros(10)
	out[x] = 1
	return out

# labels (num_examples, 10)
# label - integer label
def how_many_with_label(labels, in_label):
	count = 0
	for one_hot_label in labels:
		integer_label = one_hot_label.nonzero()[0][0]
		if (integer_label) == in_label:
			count += 1
	return count

# Splits a dataset into windows
# data - [
# 	 file name: {
# 		 x : [...]
# 		 y : [...]
# 		 z : [...]
# 	 }, 
# 	 ...	
# 	 t : [...]
# ]
# specify window width either in readings or in msecs
# step - the number of msecs window is shifted by
#
# returns shaped_array(num_examples, win_width, dim) and 
# bounds - list of time values for each window's edge values 
# 	[145345, 145350]
#	...
#	len = num_windows 
def split_into_windows(data, msecs_width=None, readings_width=None, step=None):
	half_width = int(msecs_width/step/2)	
	win_width = half_width*2
	num_examples = int(len(data[0]["x"])/half_width)-1

	shaped_array = np.zeros(shape=(num_examples, win_width, len(data)*3))

	# d = file dictionary	
	channel = 0
	for d in data: 
		for dim in ["x", "y", "z"]:
			win_bounds = []	
			all_values = d[dim]						
			all_windows = []
			
			# Indices of readings to go in window
			left_bound = 0
			right_bound = win_width

			while right_bound <= len(all_values):
				window = np.array(all_values[left_bound:right_bound])
				all_windows.append(window)
				
				win_bounds.append(np.array( [d["t"][left_bound], d["t"][right_bound]] ))

				left_bound += half_width
				right_bound += half_width
			
			d[dim] = np.array(all_windows)
			
			shaped_array[:, :, channel] = np.array(all_windows)
			channel += 1

	win_bounds = np.array(win_bounds)

	return shaped_array, win_bounds

# data (num_examples, win_width, dims)
# bounds (num_examples, 2) (necessary to determine if window is touching labeled region)
# labels - labeled regions
# 	[start, finish, integer_label]
# 	...
# 
# returns labels shape=(num_examples, 10) - every window is labeled with a one hot label
def make_labels_array(data, bounds, labels):
	print("\tmake_labels_array: ")
	labels_array = np.zeros(shape=(data.shape[0], 10))
	
	for win_i in range(data.shape[0]):
		win_label = which_region(bounds[win_i][0], bounds[win_i][1], labels)
		if (win_label != None):
			# if (win_label == 1):
			# 	plt.axvspan(bounds[win_i][0], bounds[win_i][1], facecolor="b", alpha=0.2)
			# else:
			# 	plt.axvspan(bounds[win_i][0], bounds[win_i][1], facecolor="r", alpha=0.2)

			#print("\t\t\twin_label: ", win_label)
			labels_array[win_i, :] = int_to_onehot(int(win_label))
		else:
			#plt.axvspan(bounds[win_i][0], bounds[win_i][1], facecolor="r", alpha=0.2)
			labels_array[win_i, :] = int_to_onehot(0)

	return labels_array

def write_windows_to_text(data, out_dir):
	for i in range(len(data)):
		fname_labels = file_list[i][:len(file_list[i])-12]+"_labels"+".txt"
		for dim in ["x", "y", "z"]:
			fname_data = file_list[i][:len(file_list[i])-12]+dim+".txt"

			fdata = open(out_dir+"/"+fname_data, "w")
			flabels = open(out_dir+"/"+fname_labels, "w")

			fdata_test = open(out_dir+"/test/"+fname_data, "w")
			flabels_test = open(out_dir+"/test/"+fname_data, "w")

			win_index = 1
			for window in data[i][dim]:
				if (win_index < 3200):
					for value in window[0]:
						fdata.write(str(value)+" ")
					fdata.write("\n")

					flabels.write(str(window[3])+"\n")
				else:
					for value in window[0]:
						fdata_test.write(str(value)+" ")
					fdata_test.write("\n")

					flabels_test.write(str(window[3])+"\n")
				win_index += 1

			fdata_test.close()
			flabels_test.close()
			flabels.close()
			fdata.close()

def get_labeled_regions_sum(labeled_regions):
	s = 0
	for region in labeled_regions:
		s += region[1] - region[0]
		#print("\tadding: ", region[1] - region[0])
	return s


'''
data_dir -directory with data files
file_list -list of file names to process

returns
	data - shape(num_examples, win_width, dim)
	labels - shape(num_examples, 10)
'''
the_bounds = 0
def make_data_label_arrays(data_dir, file_list):
	data = []
	labeled_regions = labels_to_list("C:/Datasets/Gavin0/labels_lots.txt")

	print("converting files to dicts")
	for file_name in file_list: 
		data.append( file_to_dict(data_dir+file_name) )

	print("sampling data")
	for d in data:
		for dim in ["x", "y", "z"]:
			d[dim], new_tvals, step = get_sample(
				startx=1481717804900, 
				endx=1481721960265, 
				num_vals=200000, 
				xvals=d["t"],
				yvals=d[dim]
			)
		d["t"] = new_tvals	

	print("plotting data")	
	# plt.plot( data[0]["t"], -np.array(data[0]["y"]) )
	# for region in labeled_regions:
	# 	plt.axvspan(region[0], region[1], facecolor="g", alpha=0.2)


	trace = go.Scatter(
	    x = np.array(data[0]["t"]),
	    y = -np.array(data[0]["y"])
	)
	pdata = [trace]
	plotly.offline.iplot(pdata)


	print("splitting into windows")
	data, bounds = split_into_windows(data=data, msecs_width=2000, step=step)
	print("\tdata shape: ", data.shape)

	print("making labels")
	labels = make_labels_array(data, bounds, labeled_regions)
	print("\tlabels shape: ", labels.shape)
	print("done reading data")

	return data, labels

# data shape (num_exmaples, win_wdith, num_chans)
# labels shape (num_examples, num_classes)
def redist_data(data, labels, dist={(1, 0):0.4, (0, 1):0.6}):
	unique_labels = [list(x) for x in set(tuple(x) for x in labels)]
	occur = {}
	index_lists = {}

	for label in unique_labels:
		label_indices = list(set(np.where(labels==label)[0]))
		num_labels = int(len(label_indices))
		occur[tuple(label)] = num_labels
		index_lists[tuple(label)] = label_indices

	min_key = min(occur.keys(), key=(lambda key : occur[key]))
	
	n = int(occur[min_key]/dist[min_key])
	new_data = np.zeros(shape=(n, data.shape[1], data.shape[2]))
	new_labels = np.zeros(shape=(n, labels.shape[1]))

	label_start = 0
	for label in unique_labels:
		num_examples = int(n*dist[tuple(label)])
		#print("lbl num: ", label, num_examples)
		
		for i in range(0, num_examples):
			win_index = index_lists[tuple(label)][i]
			
			new_data[label_start+i, :, :] = data[win_index, :, :]
			new_labels[label_start+i, :] = labels[win_index, :]

		label_start += num_examples

	return new_data, new_labels


def split_data(data, labels):
	unique_labels = [list(x) for x in set(tuple(x) for x in labels)]
	train_n = int(data.shape[0]*0.7)
	test_n = data.shape[0] - train_n

	train_data = np.zeros(shape=(train_n, data.shape[1], data.shape[2]))
	train_labels = np.zeros(shape=(train_n, labels.shape[1]))

	test_data = np.zeros(shape=(test_n, data.shape[1], data.shape[2]))
	test_labels = np.zeros(shape=(test_n, labels.shape[1]))

	# print("\tin split_data, shapes before for label: ")
	# print("\t\ttrain_data: ", train_data.shape)
	# print("\t\ttrain_labels: ", train_labels.shape)
	# print("\t\ttest_data: ", test_data.shape)
	# print("\t\ttest_labels: ", test_labels.shape)

	train_i = 0
	test_i = 0
	for label in unique_labels:
		label_indices = np.where(labels==label)[0]
		num_labels = int(len(label_indices)/2)
		
		label_train_n = int(num_labels*0.7)
		label_test_n = num_labels - label_train_n

		# print("\t\tin for, label: ", label)
		# print("\t\t\tnum_labels: ", num_labels)
		# print("\t\t\tlabel_train_n: ", label_train_n)
		# print("\t\t\tlabel_test_n: ", label_test_n)

		# fill up training and test sets proportionately
		for i in range(0, label_train_n):
			train_data[train_i+i, :, :] = data[label_indices[i]] 
			train_labels[train_i+i, :] = label

		for i in range(0, label_test_n):
			test_data[test_i+i, :, :] = data[label_indices[i+label_train_n]] 
			test_labels[test_i+i, :] = label

		train_i += label_train_n
		test_i += label_test_n

	return train_data, train_labels, test_data, test_labels



def get_batch(data, labels):
    batch_x = np.zeros( shape=(100, 1, data.shape[2], data.shape[3]) )
    batch_y = np.zeros( shape=(100, labels.shape[1]) )
    indices = [random.randint(0, len(labels)-1) for _ in range(100)]

    #print("\tnum unique indices: ", len(set(indices)))

    for i in range(100):
        batch_x[i] = data[indices[i], :, :, :]
        batch_y[i] = labels[indices[i], :]

    return batch_x, batch_y