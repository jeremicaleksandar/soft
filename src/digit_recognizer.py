import cv2

import numpy as np
from random import randint

SZ = 20
CLASS_N = 10

def split2d(img, cell_size, flatten=True):
    h, w = img.shape[:2]
    sx, sy = cell_size
    cells = [np.hsplit(row, w//sx) for row in np.vsplit(img, h//sy)]
    cells = np.array(cells)
    if flatten:
        cells = cells.reshape(-1, sy, sx)
    return cells

def load_digits(fn):
    digits_img = cv2.imread(fn, 0)
    digits = split2d(digits_img, (SZ, SZ))
    labels = np.repeat(np.arange(CLASS_N), len(digits)/CLASS_N)
    return digits, labels

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

def svmInit(C=12.5, gamma=0.50625):
  model = cv2.ml.SVM_create()
  model.setGamma(gamma)
  model.setC(C)
  model.setKernel(cv2.ml.SVM_RBF)
  model.setType(cv2.ml.SVM_C_SVC)
  
  return model

def svmTrain(model, samples, responses):
  model.train(samples, cv2.ml.ROW_SAMPLE, responses)
  return model

def svmPredict(model, samples):
  return model.predict(samples)[1].ravel()

def preprocess_simple(digits):
    return np.float32(digits).reshape(-1, SZ*SZ) / 255.0


def get_hog() : 
    winSize = (20,20)
    blockSize = (8,8)
    blockStride = (4,4)
    cellSize = (8,8)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradient = True

    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradient)

    return hog
    affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR


#ovo vraca ves istreniran model
def getTrainedModel():

    #print('Loading digits from digits.png ... ')
    # Load data.
    digits, labels = load_digits('digits.png')

    #print('Shuffle data ... ')
    # Shuffle data
    rand = np.random.RandomState(10)
    shuffle = rand.permutation(len(digits))
    digits, labels = digits[shuffle], labels[shuffle]
    
    #print('Deskew images ... ')
    digits_deskewed = list(map(deskew, digits))
	
    #print('Defining HoG parameters ...')
    # HoG feature descriptor
    hog = get_hog();

    #print('Calculating HoG descriptor for every image ... ')
    hog_descriptors = []
    for img in digits_deskewed:
        hog_descriptors.append(hog.compute(img))
    hog_descriptors = np.squeeze(hog_descriptors)
    # print('VELICINA : ', len(hog_descriptors))
    # print('JEDAN TAKI : ', len(hog_descriptors[0]))
    # print('JEDAN TAKI : ', len(hog_descriptors[1]))
    # print('JEDAN TAKI : ', len(hog_descriptors[3]))
    # print('JEDAN TAKI : ', len(hog_descriptors[30]))
    # print('JEDAN TAKI : ', len(hog_descriptors[51]))
    # print('JEDAN TAKI : ', len(hog_descriptors[68]))
    # print('A U NJEMU : ', hog_descriptors[0])
	
    #print('Spliting data into training (90%) and test set (10%)... ')
    train_n=int(0.9*len(hog_descriptors))
    digits_train, digits_test = np.split(digits_deskewed, [train_n])
    hog_descriptors_train, hog_descriptors_test = np.split(hog_descriptors, [train_n])
    labels_train, labels_test = np.split(labels, [train_n])
    
	
    # cv2.imwrite('provera/provera-0.png', digits_deskewed[40])
    # cv2.imwrite('provera/provera-1.png', digits_deskewed[622])
    # cv2.imwrite('provera/provera-2.png', digits_deskewed[151])
    # cv2.imwrite('provera/provera-3.png', digits_deskewed[90])
    # cv2.imwrite('provera/provera-4.png', digits_deskewed[822])
    # cv2.imwrite('provera/provera-5.png', digits_deskewed[256])
    # cv2.imwrite('provera/provera-6.png', digits_deskewed[904])
    # cv2.imwrite('provera/provera-7.png', digits_deskewed[998])
    # cv2.imwrite('provera/provera-8.png', digits_deskewed[967])
    # cv2.imwrite('provera/provera-9.png', digits_deskewed[311])
    # cv2.imwrite('provera/provera-10.png', digits_deskewed[487])
    # cv2.imwrite('provera/provera-11.png', digits_deskewed[666])
    # cv2.imwrite('provera/provera-12.png', digits_deskewed[765])
    # cv2.imwrite('provera/provera-13.png', digits_deskewed[451])
    # cv2.imwrite('provera/provera-14.png', digits_deskewed[512])
    # cv2.imwrite('provera/provera-15.png', digits_deskewed[411])
    # cv2.imwrite('provera/provera-16.png', digits_deskewed[35])
    # cv2.imwrite('provera/provera-17.png', digits_deskewed[98])
    # cv2.imwrite('provera/provera-18.png', digits_deskewed[800])
    # cv2.imwrite('provera/provera-19.png', digits_deskewed[700])
    # cv2.imwrite('provera/provera-20.png', digits_deskewed[500])
    # cv2.imwrite('provera/provera-21.png', digits_deskewed[966])
    # cv2.imwrite('provera/provera-22.png', digits_deskewed[961])
    # cv2.imwrite('provera/provera-23.png', digits_deskewed[962])
	
    
    #print('Training SVM model ...')
    model = svmInit()
    return svmTrain(model, hog_descriptors_train, labels_train)

	
def whatsThis(model, image):
	height, width = image.shape[:2]
	if height < 20 or width < 20:
		return -1
	image = cv2.resize(image,(20, 20), interpolation = cv2.INTER_CUBIC)
	
	imagesave = image.copy()
	
	dunno = []
	dunno.append(image)
	images = np.array(dunno)
	images = images.reshape(-1, 20, 20)

	digits_deskewed = list(map(deskew, images))
		
	hog = get_hog();
	
	hog_descriptors = []
	for img in digits_deskewed:
		hog_descriptors.append(hog.compute(img))
	hog_descriptors[0] = np.squeeze(hog_descriptors[0])
	# print('VELICINA : ', len(hog_descriptors))
	# print('JEDAN TAKI : ', len(hog_descriptors[0]))
	# print('A U NJEMU : ', hog_descriptors[0])
	
	predictions = model.predict(np.array(hog_descriptors))[1].ravel()
	
	#if int(predictions) == 2:
	#	cv2.imwrite('zasto_se_ovo_desava/'+str(randint(0, 1000))+'.png', imagesave)
	
	return int(predictions[0])
	