import matplotlib.pyplot as plt

def show_img(X, Y, index) :
	imgplot = plt.imshow(X.iloc[index].reshape(96, 96), cmap=plt.cm.gray)
	plt.plot((Y.iloc[index], Y.iloc[index]), (0, 96))