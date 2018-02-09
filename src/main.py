import cv2
import numpy as np
from copy import copy, deepcopy
import math
from random import randint
import digit_recognizer
import intersect
from sklearn import datasets, linear_model
import sys

VIDEO_NAME = "video-0.avi";

def getLines(img):
	# gray = cv2.cvtcolor(img,cv2.color_bgr2gray)
	# edges = cv2.canny(gray,50,150,aperturesize = 3)

	# lines = cv2.houghlines(edges,1,np.pi/180,200)
	# print(lines[1])
	# for line in lines:
		# for rho,theta in line:
			# a = np.cos(theta)
			# b = np.sin(theta)
			# x0 = a*rho
			# y0 = b*rho
			# x1 = int(x0 + 1000*(-b))
			# y1 = int(y0 + 1000*(a))
			# x2 = int(x0 - 1000*(-b))
			# y2 = int(y0 - 1000*(a))

			# cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

	# cv2.imshow('soft - '+VIDEO_NAME, img)
	# cv2.waitkey(0) 

	# # return
	
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	# mask_white = cv2.inRange(hsv, np.array([0,240,240]), np.array([179, 255, 255]))
	# dilation = cv2.dilate(mask_white, np.ones((2,2), np.uint8), iterations=3)
	# erosion = cv2.erode(dilation, np.ones((2,2), np.uint8), iterations=3)
	
	#TESTING (zapravo ovo radi, ostavi)
	mask_green = cv2.inRange(hsv, np.array([50,200,200]), np.array([70, 255, 255]))
	mask_blue = cv2.inRange(hsv, np.array([110,200,200]), np.array([130, 255, 255]))
	# cv2.imshow('soft - '+VIDEO_NAME, mask_green)
	# cv2.waitKey(0) 
	#cv2.imshow('soft - '+VIDEO_NAME, mask_blue)
	#cv2.waitKey(0) 	
	
	# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
	# edges = cv2.Canny(gray, 10, 150)
	# dilation = cv2.dilate(edges, np.ones((2,2), np.uint8), iterations=4)
	# erosion = cv2.erode(dilation, np.ones((2,2), np.uint8), iterations=5)

#	cv2.imshow('soft - '+VIDEO_NAME, erosion)
#	cv2.waitKey(0) 
	minLineLength = 100
	maxLineGap = 10
	
	zelene = cv2.HoughLinesP(mask_green, 1, np.pi/180, 100, minLineLength, maxLineGap)
	plave = cv2.HoughLinesP(mask_blue, 1, np.pi/180, 100, minLineLength, maxLineGap)
	
	#nadji najduze
	plava = plave[0][0];
	zelena = (0,0,0,0)
	#ovo samo trazi najduzu, ali moze se desiti da linija bude presecena na dve
	#zato ne valja (video-3 na primer)
	# for line in plave:
		# for x1,y1,x2,y2 in line:
			# cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
			# if math.hypot(x1 - x2, y1 - y2) > math.hypot(plava[0] - plava[2], plava[1] - plava[3]):
				# plava = (x1, y1, x2, y2);
	# for line in zelene:
		# for x1,y1,x2,y2 in line:
			# if math.hypot(x1 - x2, y1 - y2) > math.hypot(zelena[0] - zelena[2], zelena[1] - zelena[3]):
				# zelena = (x1, y1, x2, y2);
	minx = 1000;
	maxx = 0;	
	miny = 1000;
	maxy = 0;
	for line in plave:
		for x1,y1,x2,y2 in line:
			if x1 < minx:
				minx = x1
			if x2 < minx:
				minx = x2
			if x1 > maxx:
				maxx = x1
			if x2 > maxx:
				maxx = x2
			if y1 < miny:
				miny = y1
			if y2 < miny:
				miny = y2
			if y1 > maxy:
				maxy = y1
			if y2 > maxy:
				maxy = y2
	plava = (minx, maxy, maxx, miny)
	
	minx = 1000;
	maxx = 0;	
	miny = 1000;
	maxy = 0;
	for line in zelene:
		for x1,y1,x2,y2 in line:
			if x1 < minx:
				minx = x1
			if x2 < minx:
				minx = x2
			if x1 > maxx:
				maxx = x1
			if x2 > maxx:
				maxx = x2
			if y1 < miny:
				miny = y1
			if y2 < miny:
				miny = y2
			if y1 > maxy:
				maxy = y1
			if y2 > maxy:
				maxy = y2
	zelena = (minx, maxy, maxx, miny)
			
	
	# cv2.line(img, (plava[0], plava[1]), (plava[2], plava[3]), (150, 150, 220), 1)
	# cv2.line(img, (zelena[0], zelena[1]), (zelena[2], zelena[3]), (150, 150, 220), 1)
	# cv2.imshow('soft - '+VIDEO_NAME, img)
	# cv2.waitKey(0)
	
	return plava, zelena

def prosiri(plava):
	
	produzetak = 8 #produzetak
	
	k_p = (plava[3]-plava[1])/(plava[2]-plava[0])
	#y = k_p * x
	x_p = int(produzetak / math.sqrt(1+k_p*k_p))
	y_p = int(k_p * x_p)
	plava1 = (plava[0]-x_p, plava[1]-y_p, plava[2]+x_p, plava[3]+y_p)
	
	return plava1

#ne koristi ovo
def podigni(plava):
	za_koliko = 15
	
	k_p = (plava[3]-plava[1])/(plava[2]-plava[0])
	#y = k_p * x
	x_p = int(za_koliko / math.sqrt(1+k_p*k_p))
	y_p = int(k_p * x_p)
	plava1 = (plava[0]-x_p, plava[1]+y_p, plava[2]-x_p, plava[3]+y_p)
	
	return plava1

def getCleanImage(image):
	#sklanja linije, vraca monohormatsku sliku, posterizovanu

	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	mask_green = cv2.inRange(hsv, np.array([50,70,50]), np.array([120, 255, 255]))
	mask_blue = cv2.inRange(hsv, np.array([110,200,50]), np.array([130, 255, 255]))
	img_save = deepcopy(image)
	img_save[mask_green > 0] = 0
	img_save[mask_blue > 0] = 0
	#cv2.imshow("Maska", img_save)
	#cv2.waitKey(0)
	#^ ovo je izbacilo i tackice doduse (98%)
	
	#erozija za one tackice
	#kernel = np.ones((2,2),np.uint8)
	#erosion = cv2.erode(img_save,kernel,iterations = 1)
	
	#edged = cv2.Canny(img_save, 10, 250)
	#cv2.imshow("Edges", edged)
	#cv2.waitKey(0) 

	#applying closing function 
	#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
	#closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
	#cv2.imshow("Closed", closed)
	#cv2.waitKey(0)
	
	gray = cv2.cvtColor(img_save, cv2.COLOR_BGR2GRAY);
	#kernel = np.ones((2,2),np.uint8)
	#erosion = cv2.erode(gray,kernel,iterations = 1)
	#dilation = cv2.dilate(gray,kernel,iterations = 3) #ove 3 dilatacije ce pomeriri sve u desno dole malo, obrati paznju kasnije
	#gray = dilation
	#ovo sad je posterizacija (ove 2 linije ispod)
	#iako izgleda kao sa imamo samo crno i belo, zapravo
	#ima nekih artefakta (ovo to resava)
	#gray[gray >= 128]= 255
	#gray[gray < 128] = 0
	#v2.imshow("erozija", gray)
	#cv2.waitKey(0)
	return gray

def getBoundingBoxes(image):

	#da izbaci plavu i zelenu liniju
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	mask_green = cv2.inRange(hsv, np.array([50,70,50]), np.array([120, 255, 255]))
	mask_blue = cv2.inRange(hsv, np.array([110,200,50]), np.array([130, 255, 255]))
	img_save = deepcopy(image)
	img_save[mask_green > 0] = 0
	img_save[mask_blue > 0] = 0
	#cv2.imshow("Maska", img_save)
	#cv2.waitKey(0)
	#^ ovo je izbacilo i tackice doduse (98%)
	
	#erozija za one tackice
	#kernel = np.ones((2,2),np.uint8)
	#erosion = cv2.erode(img_save,kernel,iterations = 1)
	
	#edged = cv2.Canny(img_save, 10, 250)
	#cv2.imshow("Edges", edged)
	#cv2.waitKey(0) 

	#applying closing function 
	#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
	#closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
	#cv2.imshow("Closed", closed)
	#cv2.waitKey(0)
	
	gray = cv2.cvtColor(img_save, cv2.COLOR_BGR2GRAY);
	kernel = np.ones((2,2),np.uint8)
	#erosion = cv2.erode(gray,kernel,iterations = 1)
	dilation = cv2.dilate(gray,kernel,iterations = 3) #ove 3 dilatacije ce pomeriri sve u desno dole malo, obrati paznju kasnije
	gray = dilation
	#ovo sad je posterizacija (ove 2 linije ispod)
	#iako izgleda kao sa imamo samo crno i belo, zapravo
	#ima nekih artefakta (ovo to resava)
	gray[gray >= 128]= 255
	gray[gray < 128] = 0
	#cv2.imshow("erozija", gray)
	#cv2.waitKey(0)
	#finding_contours (od closed-a)
	(_, cnts, _) = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	 

	idx = 0
	image2 = deepcopy(image)
	ret = [];
	for c in cnts:
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)
		cv2.drawContours(image2, [approx], -1, (0, 255, 0), 1)
		
		#image cropping
		x,y,w,h = cv2.boundingRect(c)
		centarx = x + w//2;
		centary = y + h//2;
		if w<10 and h<10:
			continue #preskoci ove male, ostale upisi u ret listu
		else:
			idx+=1
			new_img=image[y:y+h,x:x+w]
			#cv2.imwrite(str(idx) + '.png', new_img)
			#ili ovako???
			cv2.rectangle(image2, (centarx-15, centary-15), (centarx+15, centary+15), (140,140,240), 1)
			ret.append(([centarx-15-1, centary-15-1-1], [w,h]))
			#minus 1 zbog one 3 dilatacije (malo je pomerila sve)
			#ono minus 1 kod y je da spusti malo dole jer izgleda da u learning
			#setu cifre zaista jesu malo spustene (blize donjoj ivici)
	
	#u retu su sad sve cifre koje smo povatali
	#ALI
	#mozda ima suma menju tim slicicama
	
	#cv2.imshow("konture", image2)
	#cv2.waitKey(0)
	
	praviret = []
	
	#sad ide pravi purge
	#za svaku ovu, pogledaj kakva je u orig slici
	#ako su konture u orig slici manje od 5x5, onda je garant to sum
	for (r, wh) in ret:
		#cifra_slika = image[r[1]:r[1]+30, r[0]:r[0]+30]
		cifra_slika = image[r[1]+5:r[1]+25, r[0]+5:r[0]+25]
		#ovi +5 na levu i -5 na desnu ivicu da bi slicica
		#koji vadimo bila jos manja i da ova provera bude jos striktija
		
		if cifra_slika.shape[0] < 20 or cifra_slika.shape[1] < 20:
			#ovo je nesto uz ivicu, ignorisi ga sada
			#(ako je gore, eventualno ce se spustiti pa cemo
			#ga u nekom narednom frejmu uhvatiti)
			#(ako je dole, onda nam i ne treba)
			continue
		
		edged = cv2.Canny(cifra_slika, 10, 250)
		#gray = cv2.cvtColor(cifra_slika, cv2.COLOR_BGR2GRAY);
		closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
		(_, cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		
		broj_tackica = 0
		for c in cnts:
			peri = cv2.arcLength(c, True)
			x,y,w,h = cv2.boundingRect(c)
			cv2.rectangle(image2, (r[0]+5+x, r[1]+5+y), (r[0]+5+x+w, r[1]+5+y+h), (255,100,100), 1)
			
			if w < 6 and h < 6: 
				#aha, ovo je sum (tackica)
				cv2.rectangle(image2, (r[0]+5+x, r[1]+5+y), (r[0]+5+x+w, r[1]+5+y+h), (0,0,255), 1)
				broj_tackica += 1
		if broj_tackica == len(cnts):
			#pa ovo su sve tackice na slicici
			cv2.rectangle(image2, (r[0], r[1]), (r[0]+30, r[1]+30), (0,0,250), 1)
			continue
		else:
			#okej, ima nesto sem tackica (broj_tackica < len(c))
			praviret.append((r, wh))
	
	#cv2.imshow("Output", image2)
	#cv2.waitKey(0)
	return praviret

def getValue(s): #samo nadje najbolje ocenjenu procenu
	ret_key = 0
	for k, v in s['votes'].items():
		if v > s['votes'][ret_key]:
			ret_key = k
	return ret_key
	
def getSecondValue(s):
	best = getValue(s)
	
	if not best == 0:
		ret_key = 0
	else:
		ret_key = 1
		
	for k, v in s['votes'].items():
		if not k == best:
			if v > s['votes'][ret_key]:
				ret_key = k
	
	return ret_key	
	
def getThirdValue(s):
	best = getValue(s)
	second_best = getSecondValue(s)
	
	if best == 0 or second_best == 0:
		if best == 1 or second_best == 1:
			ret_key = 2
		else:
			ret_key = 1
	else:
		ret_key = 0
		
	for k, v in s['votes'].items():
		if not k == best and not k == second_best:
			if v > s['votes'][ret_key]:
				ret_key = k
	
	return ret_key
	
# def getValueG(s): #vadi geometrijsku sredinu ocene iz 'votes' dicta u jednom objektu iz 'strings' liste, vraca je kao int
	# votecount = 0
	# totalvotesum = 0
	# for k, v in s['votes'].items():
		# votecount += v
		# totalvotesum += k*v
	# return int(round(totalvotesum/votecount))

if __name__ == '__main__' :
	
	if len(sys.argv) != 2:
		print('Input video name is missing; loading default - ', VIDEO_NAME)
	else:
		VIDEO_NAME = sys.argv[1]
		print('Loading video - ', VIDEO_NAME)
	
	cv2.namedWindow('soft - '+VIDEO_NAME)
	
	camera = cv2.VideoCapture(VIDEO_NAME)
	
	ok, image=camera.read()
	if not ok:
		print('Failed to read video')
		exit()
	
	plava, zelena = getLines(image)
	plava, zelena = prosiri(plava), prosiri(zelena)
	
	print('training svm model...')
	svmmodel = digit_recognizer.getTrainedModel();
	print('done')
	
	suma = 0;
	
	#putanje svake cifre s koordinatama puta (na svakih 10 frejmova)
	strings = {}
	prosli_korak = {} #pozicije svih do sad vidjenih cifara, gde su bile u proslom frejmu
	proslo_frejmova = 0 #ovo se resetuje na 0 na svakih 10 frejmova
	
	#da pocinje da proverava intersection tek nekih 30-ak frejmova
	#od pocetka filma;
	#u video-5.avi npr ima jedna sestica koja na pocetku ima gornje-levo
	#teme iza linije i odmah za 10 frejmova ispod te linije, iako je sama
	#od pocetka bila ispod linije, pa da ne bi takvi slucajevi pravili problem
	prvih_20ak_frejmova = 0
	
	#pravi frame count ,zapravo da broji frejmove
	frame_count = 0;
	
	#kljuc je  kljuc iz strings liste, vrednost je par (y1,y2), opisuje poc i kraj (regresija)
	izgubljene = {}
	
	regr = linear_model.LinearRegression()
	
	while camera.isOpened():
	
		frame_count += 1;
	
		if prvih_20ak_frejmova < 20:
			prvih_20ak_frejmova += 1;
	
		savelastimage  = deepcopy(image) #cisto da imamo kad se zavrsi video
		ok, image=camera.read()
		if not ok:
			print('no image to read')
			image = savelastimage
			break
	
		kopija = deepcopy(image); #doduse bez linija
		hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		mask_white = cv2.inRange(hsv, np.array([0,240,240]), np.array([179, 255, 255]))
		kopija[mask_white > 0] = 0;
		cleanimagesave = cv2.cvtColor(kopija, cv2.COLOR_BGR2GRAY)
	
		boundingBoxes = getBoundingBoxes(image)
		#zapravo samo pozicije, uvek je velicine 20x20
		for bb in boundingBoxes:
			x, y = bb[0]
			w, h = bb[1]
			
			cv2.rectangle(image, (x, y), (x+30, y+30), (120,120,240), 1)
			
			vecpostojala = 0
			for key, value in prosli_korak.items():
				xx, yy = value['pos']
				#proverava blizinu da utvrdi da se radi o istoj tacki (da, znam), i pri tome ne gleda izgubljene tacke
				#jer bi se tako najverovatnije vezao na pogresnu nit (kad je guzva)
				if math.hypot(xx - x, yy - y) < 14 and not key in izgubljene and not strings[key]['gone_for_good']: #not gone_for_good da se ne vezuje za neku koja je izasla van
				
					#GONE_FOR_GOOD deo je usko grlo, ako bude problema vrlo moguce da je ovde greska
					#(kad se u dnu ekrana nagomilaju cifre i jos je tu nisko linija, moze da bude problema)
				
					#(probaj 20 za opustenije, 10 za striktnije)
					#znaci to je ta tacka, samo iz proslog frejma
					#(u prosli_korak se nalaze [kljuc:value] elementi
					#koji opisuju poslednju poziciju svih cifara;
					#kljuc u tom dictu je isti onaj u strings dictu)

					# slicica = cleanimagesave[y:y+30, x:x+30]
					# predicted = digit_recognizer.whatsThis(svmmodel, slicica)
					# if predicted == -1:
						# continue
					# if predicted == getValue(strings[key]) or predicted == getSecondValue(strings[key]) or predicted == getThirdValue(strings[key]):
					
					
					vecpostojala = 1
					savedkey = key;
					break
					
			if (vecpostojala):
			
				#na svakih 10 frejmova cemo zabeleziti gde je bila cifra
				if proslo_frejmova == 10:
					strings[savedkey]['list'].append((x, y));
					
					if prvih_20ak_frejmova >= 20:
						#aj i da vidimo da li je presekla neku od linija
						prosla = strings[savedkey]['list'][len(strings[savedkey]['list'])-2]
						
						#TESTING
						#kad se cifra podvude pod liniju pa moze da bude ocitana kao da je presla preko nje;
						#samo proverava koef. ugla i da li smo pri cosku linije, pa koriguje duzinu linije
						#pre provere preseka (da provera bude malo strozija)
						l = math.hypot(prosla[0]-x, prosla[1]-y)
						slope = -1
						if not prosla[0] == x: 
							slope = (prosla[1]-y)/(prosla[0]-x)
						kor1, kor2 = 0, 0
						if l > 10:
							if slope > 1 and math.hypot(x-plava[2], y-plava[3]) <= 15:
								kor1 = 5
							if slope < 1 and math.hypot(x-plava[0], y-plava[1]) <= 15:
								kor2 = 5
						#kor1, kor2 = 0, 0
						
						seku_se_plava = intersect.da_li_se_seku(np.array([prosla[0], prosla[1]]), np.array([x, y]), np.array([plava[0]+kor2, plava[1]]), np.array([plava[2]-kor1, plava[3]]))
						seku_se_zelena = intersect.da_li_se_seku(np.array([prosla[0], prosla[1]]), np.array([x, y]), np.array([zelena[0], zelena[1]]), np.array([zelena[2], zelena[3]]))
						#cv2.line(image, (prosla[0], prosla[1]), (x, y), (250, 0, 255), 2)
						#cv2.line(image, (linije[0][0], linije[0][1]), (linije[0][2], linije[0][3]), (0, 250, 255), 2)
						#cv2.line(image, (linije[1][0], linije[1][1]), (linije[1][2], linije[1][3]), (0, 250, 255), 2)
						#cv2.imshow('soft - '+VIDEO_NAME, image)
						#cv2.waitKey(0)
						if seku_se_plava == 1:
							suma += getValue(strings[savedkey])
							print('+', str(getValue(strings[savedkey])))
							strings[savedkey]['prosao_plavu'] = True
#							cv2.line(image, (x, y), (prosla[0], prosla[1]), (0, 0, 255), 1)
#							cv2.imshow('soft - '+VIDEO_NAME, image)
#							cv2.waitKey(0)
						if seku_se_zelena == 1:
							suma -= getValue(strings[savedkey])
							print('-', str(getValue(strings[savedkey])))
							strings[savedkey]['prosao_zelenu'] = True
#							cv2.line(image, (x, y), (prosla[0], prosla[1]), (0, 0, 255), 1)
#							cv2.imshow('soft - '+VIDEO_NAME, image)
#							cv2.waitKey(0)
						
					
				
				#TODO: zasto u svakom frejmu prepoznajemo cifru??? (sto ne na svakih 10?)
				
				#TEST
				#ideja: na svakih deset frejmova 'resetuj' expected_dimensions
				#i to radi SAMO ako su dovoljno velike (da ne bi neke prekrivene cifre bile evaluirane previse puta)
				#(ovo bi trebalo da resi problem kada je pogresna cifra zakucana i ne dozvoljava predikciju nove zato
				#sto je negde ranije expected_dimensions kvadrat bio prevelik)
				#if proslo_frejmova == 10 and w >= 14 and h >= 14:
				#	strings[savedkey]['expected_dimensions'] = (w, h);
				
				if w >= strings[savedkey]['expected_dimensions'][0]-1 and h >= strings[savedkey]['expected_dimensions'][1]-1: #-1 da ima lufta za gresku
					strings[savedkey]['expected_dimensions'] = (w, h);
					
					slicica = cleanimagesave[y:y+30, x:x+30]
					predicted = digit_recognizer.whatsThis(svmmodel, slicica)
					if not predicted == -1:
						strings[savedkey]['votes'][predicted] += 1
					
				cv2.putText(image, str(getValue(strings[savedkey])), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1, cv2.LINE_AA)
				cv2.putText(image, str(strings[savedkey]['votes'][getValue(strings[savedkey])]), (x+20, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 70), 1, cv2.LINE_AA)
				cv2.putText(image, str(getSecondValue(strings[savedkey])), (x, y+38), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 60), 1, cv2.LINE_AA)
				cv2.putText(image, str(getThirdValue(strings[savedkey])), (x, y+46), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (90, 90, 60), 1, cv2.LINE_AA)
				cv2.putText(image, str(strings[savedkey]['votes'][getSecondValue(strings[savedkey])]), (x+20, y+38), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 90), 1, cv2.LINE_AA)
				cv2.putText(image, str(strings[savedkey]['votes'][getThirdValue(strings[savedkey])]), (x+20, y+46), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (90, 90, 90), 1, cv2.LINE_AA)
				#jos i broj glasova za najjacu
				
				
				#u SVAKOM frejmu cemo beleziti gde se cifra sada nalazi
				#da bi sa time poredili u narednom frejmu
				prosli_korak[savedkey] = {'pos' : (x,y), 'last_update' : frame_count}
				#last_update = rb frejma kad je poslednju put ova cifra updateovana (posle prodji kroz 'prosli_korak' i vidi koga smo izgubili u ovom frejmu)
				
				#zapisi i koliko dugo vec putuje, u strings listi
				#(NOTE: to NIJE broj frejmova u kojima su zabelezne tacke
				#u toj listi, vec totani broj frejmova koji je ova cifra
				#putovala)
				strings[savedkey]['been_traveling_for'] += 1;
				
			else: #nova
				
				#ako je negde van leve i gornje ivice, i nije prvi frejm, onda to zapravo nije nova
				#50px (iako je vec na 30 vidi) je da ima malo lufta, da se pojave nove ako se preklapaju
				if x > 59 and y > 59 and not frame_count == 1:
					#ovo nije nova, vec neka od starih se ponovo pojavila
					#nadji je
					
					#ovo ce nam trebati da je nadjemo
					#slicica = cleanimagesave[y:y+30, x:x+30]
					#predicted = digit_recognizer.whatsThis(svmmodel, slicica)
					
					pronadjena = False
					
					for kk, vv in izgubljene.items():
						#TODO
						xx1 = strings[kk]['list'][0][0]
						yy1 = izgubljene[kk][0]
						xx2 = strings[kk]['list'][-1][0]
						yy2 = izgubljene[kk][1]
						
						if xx2 > x and yy2 > y:
							#ova izgubljena je izgubljena ispred ove koju sad posmatramo
							#znaci to sigurno nije ta
							continue
						
						k = (yy2-yy1)/(xx2-xx1)
						n = yy1 - k*xx1
						
						predikcija_pozicije = k * x + n
						
						
								
						if abs(predikcija_pozicije - y) < 30:
						
							#linija dobijena regresijom izgubljene putanje, od (xx1,yy1) do (xx2,yy2)
							cv2.line(image, (xx1, yy1), (xx2, yy2), (255, 0, 255), 1)
							cv2.rectangle(image, (x-2, predikcija_pozicije-2), (x+2, predikcija_pozicije+2), (0, 0, 255), 1)
							#cv2.imshow('soft - '+VIDEO_NAME, image)
							#cv2.waitKey(0)
							#okej, jeste joj na putuje
							#a da li je dovoljno daleko?? (TODO:?)
							
							#OKEJ, TO JE TA, ZAPISI JE (ZASAD)
							pronadjena = True
							strings[kk]['list'].append((x, y));
							
							if w >= strings[kk]['expected_dimensions'][0]-1 and h >= strings[kk]['expected_dimensions'][1]-1:
								strings[kk]['expected_dimensions'] = (w, h);
								
								slicica = cleanimagesave[y:y+30, x:x+30]
								predicted = digit_recognizer.whatsThis(svmmodel, slicica)
								if not predicted == -1:
									strings[kk]['votes'][predicted] += 1
							#proverio sam dimenzije : ako je manjih dimenzija od ocekivanih, onda nesto nije u redu;
							#vrv je zaklonjena nekom drugom (u tom slucaju, ne radi predikciju uopste)
							
							cv2.putText(image, str(getValue(strings[kk])), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1, cv2.LINE_AA)
							cv2.putText(image, str(strings[kk]['votes'][getValue(strings[kk])]), (x+20, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 70), 1, cv2.LINE_AA)
							cv2.putText(image, str(getSecondValue(strings[kk])), (x, y+38), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 60), 1, cv2.LINE_AA)
							cv2.putText(image, str(getThirdValue(strings[kk])), (x, y+46), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (90, 90, 60), 1, cv2.LINE_AA)
							cv2.putText(image, str(strings[kk]['votes'][getSecondValue(strings[kk])]), (x+20, y+38), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 90), 1, cv2.LINE_AA)
							cv2.putText(image, str(strings[kk]['votes'][getThirdValue(strings[kk])]), (x+20, y+46), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (90, 90, 90), 1, cv2.LINE_AA)
							prosli_korak[kk] = {'pos' : (x,y), 'last_update' : frame_count}
							strings[kk]['been_traveling_for'] += 1;
							
							if prvih_20ak_frejmova >= 20:
								#aj i da vidimo da li je presekla neku od linija
								#-2 jer hocemo predposlednju (poslednja je onaj x,y sto si malo pre dodao u listu)
								prosla = strings[kk]['list'][len(strings[kk]['list'])-2]
								seku_se_plava = intersect.da_li_se_seku(np.array([prosla[0], prosla[1]]), np.array([x, y]), np.array([plava[0], plava[1]]), np.array([plava[2], plava[3]]))
								seku_se_zelena = intersect.da_li_se_seku(np.array([prosla[0], prosla[1]]), np.array([x, y]), np.array([zelena[0], zelena[1]]), np.array([zelena[2], zelena[3]]))
								if seku_se_plava == 1:
									suma += getValue(strings[kk])
									print('+', str(getValue(strings[kk])))
									strings[kk]['prosao_plavu'] = True
								if seku_se_zelena == 1:
									suma -= getValue(strings[kk])
									print('-', str(getValue(strings[kk])))
									strings[kk]['prosao_zelenu'] = True
							
							#okej , break i onda je izbaci iz izgubljenih
							break
					
					if pronadjena:
						del izgubljene[kk]
						
						
						
						
				else:
					#ovo zaista jeste nova

					#prvo izvuci tu cifru (sliku)
					slicica = cleanimagesave[y:y+30, x:x+30]
					
					predicted = digit_recognizer.whatsThis(svmmodel, slicica)
					
					if predicted == -1: #kako se ovo uopste desava???
						continue
					
					key = len(strings)
					strings[key] = {'list' : [], 'been_traveling_for' : 1, 'votes' : {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 0:0}, 'expected_dimensions' : (w,h), 'prosao_plavu' : False, 'prosao_zelenu' : False, 'gone_for_good' : False}
					strings[key]['votes'][predicted] += 1;
					#been_traveling_for = broj frejmova koliko dugo putuje
					#(da bi od toga posle racunao brzinu)
					strings[key]['list'].append((x, y))
					cv2.putText(image, str(getValue(strings[key])), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1, cv2.LINE_AA)
					
					#opet, i za nju sacuvaj gde se sada nalazi, trebace u sledecem frejmu
					prosli_korak[key] = {'pos' : (x,y), 'last_update' : frame_count}
		
		
		
		#sad da vidimo koje smo izgubili u ovom frejmu
		for k, v in prosli_korak.items():
			#ovo za length liste samo da izbegnemo onaj sum
			#ako je poslednji put updateovana u proslom frejmu (a ne u ovom sada), onda smo je izgubili
			if len(strings[k]['list']) > 2 and v['last_update'] == frame_count-1:
				#ovu smo izgubili
				
				if v['pos'][0] > (640-40) or v['pos'][1] > (480-40):
					#ova je izgubljena jer je izasla van ekrana, nemoj je ni beleziti,
					#svejedno se na nju nista nece povezati
					#(usput, -40: 30x30 su dimenzije slicice, i jos 10px lufta jer se pomera)
					
					#MOMENAT
					
					#mozda je presekla liniju bas pre nego sto je nestala, proveri
					
					#(klasicna provera, da li string sece liniju)
					x, y = v['pos'] #sto je isto kao strings[k]['list'][-1] (tj. poslednja pozicija)
					px, py = strings[k]['list'][-2] #predposlednja pozicija 
					
					if not strings[k]['prosao_plavu']:
						seku_se_plava = intersect.da_li_se_seku(np.array( [x, y] ), np.array( [px, py] ), np.array( [plava[0], plava[1]] ), np.array( [plava[2], plava[3]] ));
						if seku_se_plava:
							suma += getValue(strings[k])
							print('+', getValue(strings[k]))
							strings[k]['prosao_plavu'] = True
					if not strings[k]['prosao_zelenu']:
						seku_se_zelena = intersect.da_li_se_seku(np.array( [x, y] ), np.array( [px, py] ), np.array( [zelena[0], zelena[1]] ), np.array( [zelena[2], zelena[3]] ));
						if seku_se_zelena:
							suma -= getValue(strings[k])
							print('-', getValue(strings[k]))
							strings[k]['prosao_zelenu'] = True
					
					#(ovo je provera sa gornjom i levom ivicom (kao onda sto se radi na kraju))
					x, y = v['pos']
					seku_se_plava1 = intersect.da_li_se_seku(np.array( [x, y] ), np.array( [x, y+30] ), np.array( [plava[0], plava[1]] ), np.array( [plava[2], plava[3]] ));
					seku_se_plava2 = intersect.da_li_se_seku(np.array( [x, y] ), np.array( [x+30, y] ), np.array( [plava[0], plava[1]] ), np.array( [plava[2], plava[3]] ));
					seku_se_zelena1 = intersect.da_li_se_seku(np.array( [x, y] ), np.array( [x, y+30] ), np.array( [zelena[0], zelena[1]] ), np.array( [zelena[2], zelena[3]] ));
					seku_se_zelena2 = intersect.da_li_se_seku(np.array( [x, y] ), np.array( [x+30, y] ), np.array( [zelena[0], zelena[1]] ), np.array( [zelena[2], zelena[3]] ));
					
					if seku_se_plava1 and seku_se_plava2:
						suma += getValue(strings[k])
						print('+', getValue(strings[k]))
						strings[k]['prosao_plavu'] = True
					if seku_se_zelena1 and seku_se_zelena2:
						suma -= getValue(strings[k])
						print('-', getValue(strings[k]))
						strings[k]['prosao_zelenu'] = True
					
					#okej, otpisujemo ovu, sledeca...
					strings[k]['gone_for_good'] = True
					continue
				
				putanja = strings[k]['list']
				X = []
				Y = []
				for xx, yy in putanja:
					X.append([xx])
					Y.append(yy)
				regr.fit(X, Y)
				y1 = regr.predict(putanja[0][0])
				y2 = regr.predict(putanja[-1][0])
				izgubljene[k] = (y1, y2) #zabelezi da je izgubljena,
				#sa koordinatama y1 i y2, pocetak i kraj
				
				#cv2.line(image, (putanja[0][0], y1), (putanja[-1][0], y2), (255, 0, 255), 1)
				#cv2.imshow('soft - '+VIDEO_NAME, image)
				#cv2.waitKey(0)
				
				#TODO: sta sad???
				#aj probamo negde kasnije je pronaci i povezani na neku
				#novo nadjenu
	
		#linije posle kojih ne prihvatamo 'nove' (sem u nekom extremnom slucaju???(???????))
		cv2.line(image, (0, 59), (640, 59), (50, 100, 100), 1)
		cv2.line(image, (59, 0), (59, 480), (50, 100, 100), 1)
		#linije posle kojih ne prihvatamo 'izgubljene' (jer su izasle van, i niko se nece povezati na njih svejedno)
		cv2.line(image, (640-40, 0), (640-40, 480), (0, 0, 100), 1)
		cv2.line(image, (0, 480-40), (640, 480-40), (0, 0, 100), 1)
		#progress bar
		cv2.line(image, (0, 475), (int((640*frame_count)/1200), 475), (255, 255, 255), 2)
	
		cv2.line(image, (plava[0], plava[1]), (plava[2], plava[3]), (0, 255, 255), 1)
		cv2.line(image, (zelena[0], zelena[1]), (zelena[2], zelena[3]), (255, 0, 255), 1)
		cv2.putText(image, str(suma), (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
		cv2.imshow('soft - '+VIDEO_NAME, image)
		k = cv2.waitKey(1)
		if k == 27 : break # esc pressed
		
		#if frame_count > 490:
		#	cv2.waitKey(0)
		
		proslo_frejmova = proslo_frejmova + 1
		if (proslo_frejmova > 10):
			proslo_frejmova = 0;
	
	# for k, v in strings.items():
		# print('-----------------')
		# print('CIFRA ', k)
		# print('-----------------')
		# for pos in v:
			# print(pos[0], pos[1])
	
	#E SAD
	#mozda je neka cifra presla preko linije ali nije presla skroz
	#pre nego sto se video zavrsio (na primer poslednja dvojka u video-5)
	#proveri jos u tom poslednjem frejmu da nema neka cifra koja ima
	#bounding box sa gornjim levim temenom iznad linije a ostala 2
	#najbliza temena ispod linije
	for bb in boundingBoxes:
		x, y = bb[0]
		#w, h = bb[1] #ne treba ti ovo sad
	
		vecpostojala = 0
	
		for key, value in prosli_korak.items():
			xx, yy = value['pos']
			if math.hypot(xx - x, yy - y) < 15:
				vecpostojala = 1
				savedkey = key;
				break
	
		seku_se_plava1 = intersect.da_li_se_seku(np.array( [x, y] ), np.array( [x, y+30] ), np.array( [plava[0], plava[1]] ), np.array( [plava[2], plava[3]] ));
		seku_se_plava2 = intersect.da_li_se_seku(np.array( [x, y] ), np.array( [x+30, y] ), np.array( [plava[0], plava[1]] ), np.array( [plava[2], plava[3]] ));
		seku_se_zelena1 = intersect.da_li_se_seku(np.array( [x, y] ), np.array( [x, y+30] ), np.array( [zelena[0], zelena[1]] ), np.array( [zelena[2], zelena[3]] ));
		seku_se_zelena2 = intersect.da_li_se_seku(np.array( [x, y] ), np.array( [x+30, y] ), np.array( [zelena[0], zelena[1]] ), np.array( [zelena[2], zelena[3]] ));
		if seku_se_plava1 and seku_se_plava2:
			cv2.rectangle(image, (x, y), (x+30, y+30), (0,0,254), 1)
			slicica = cleanimagesave[y:y+30, x:x+30]
			predicted = digit_recognizer.whatsThis(svmmodel, slicica)
			
			if predicted == -1:
				continue
				
			if vecpostojala == 1:
				strings[savedkey]['votes'][predicted] += 1;
				predicted = getValue(strings[savedkey]);
			
			suma += predicted
			print('+', predicted)
			strings[savedkey]['prosao_plavu'] = True
		if seku_se_zelena1 and seku_se_zelena2: #ovde je ranije bilo OR, greskom (a da li je greska?)
			cv2.rectangle(image, (x, y), (x+30, y+30), (0,0,254), 1)
			slicica = cleanimagesave[y:y+30, x:x+30]
			predicted = digit_recognizer.whatsThis(svmmodel, slicica)
			
			if predicted == -1:
				continue
			
			if vecpostojala == 1:
				strings[savedkey]['votes'][predicted] += 1;
				predicted = getValue(strings[savedkey]);
			
			suma -= predicted
			print('-', predicted)
			strings[savedkey]['prosao_zelenu'] = True
	
	cv2.putText(image, '####', (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2, cv2.LINE_AA)
	cv2.putText(image, '####', (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2, cv2.LINE_AA)
	cv2.putText(image, 'finalna suma: '+str(suma), (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
			
			
			
	
	
	print('Suma: ', suma)
	print('frame count: ', frame_count)
	
	i = 0;
	#bg za legendu
	cv2.line(image, (0, 470), (640, 470), (0,0,0), 15)
	for k,v in strings.items():
		color = (randint(0, 255), randint(0, 255), randint(0, 255));
		if (len(v['list']) < 2): #TODO: privremeno resenje, da preskoci sum
			continue
		for index , pos in enumerate(v['list']):
			#cv2.rectangle(image, (pos[0], pos[1]), (pos[0]+1, pos[1]+1), color, 1)
			if not index == 0:
				cv2.line(image, (pos[0], pos[1]), (v['list'][index-1][0], v['list'][index-1][1]), color, 1)
		#legenda
		cv2.putText(image, str(getValue(v)), (5+i*6, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)
		i = i+1
	cv2.imshow('soft - '+VIDEO_NAME, image)
	k = cv2.waitKey(0)