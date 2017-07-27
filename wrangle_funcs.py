exec(open("C:/PythonStuff/imports.py").read())


def file_to_dict(in_file_path):
	'''
	Takes path to file of format 

	x0 y0 z0 t0
	x1 y1 z1 t1
	...

	returns {
		"x":[x0, x1, ...],
		"y":[y0, y1, ...],
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

def lin_inter(a, b):
	'''
	Linear interpolation
	a = [x0, x, x1]
	b = [y0,    y1]
	Returns  y  between y0 and y1
	'''
	bigA = a[2] - a[0]
	leftA = a[1] - a[0]
	bigB = b[1] - b[0]
	return b[0] + bigB*(leftA/bigA)


def get_sample(xvals, yvals, startx, endx, num_vals):
	'''
	Given two sequences, samples any number of points from them.
	Linearly interpolates for points between samples 

	xvals - list of time stamps
	yvals - list of sensor values
	both are ordered, e.g. 5th timetsamp belongs with 5th sensor value

	startx - the xvalue (time) we want to start sampling from. Does not have to be contained in xvals.
	endx - (time) sample until this value
		if there is no data for a sample xvalue, outputs zero

	num_vals - number of values in sample

	returns tuple (sampled_ys, sampled_xs, time between samples)
	'''
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


def make_data(data_dir, file_list):
	'''
	data_dir -directory with data files
	file_list -list of file names to process

	returns
		data - shape(num_examples, win_width, dim)
	'''
	data = []

	print("converting files to dicts...", end="")
	for file_name in file_list: 
		data.append( file_to_dict(data_dir+file_name) )
	print("done")

	print("sampling data...", end="")
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
	print("done")	

	print("plotting data...", end="")	
	pdata = []
	fig = tools.make_subplots(rows=12, cols=1)
	row = 1
	for d in data:
		for dim in ["x", "y", "z"]:			
			trace = go.Scatter(
			    	x = np.array(d["t"]),
			    	y = -np.array(d[dim])
			)
			if (row <= 12): fig.append_trace(trace, row, 1)
			row += 1
	
	plotly.offline.iplot(fig)
	print("done")

	return data