Pocetak programa je u main.py fajlu, te se moze pokrenuti iz komandne linije, kao svaki drugi python program, komandom:

py .\main.py .\video-9.avi

Poslednji paramtear u komandi je putanja do video zapisa koji ce biti ucitan, i ovaj parametar je opcion (u slucaju da nije naveden, bice ucitan snimak cija je putanja upisana u polju VIDEO_NAME, u main.py fajlu).
Ukoliko se snimak nalazi van src foldera, putanju do njega navesti u gore navedenoj komandi, a ne preko VIDEO_NAME promenljive (iz nekog razloga nece uspeti da ucita).

Po pokretanju, potrebno je sacekati par sekundi da se SVM model obuci zadatim trening setom (digits.png), posle cega se pocinju prikazivati obradjeni frejmovi snimka.
Rozi kvadrati su tu da oznace prepoznate cifre. Iznad kvadrata je velikim fontom ispisana prepoznata cifra (ona sa najvise glasova), i a malim fontom pored nje stoji broj glasova za tu cifru. Ispod kvadrata, malim fontom ispisane, stoje druga i treca najbolje izglasana cifra, i pored obe (desno) je i broj glasova za svaku o njih.
U gornjem levom uglu je ispisana suma. U dnu prozora postoj i progress bar, koji prikazuje koliki je deo snimka do sad obradjen.
Po zavrsetku rada program iscrtava putanje svih prepoznatih cifara i ispisuje konacnu sumu u gornjem levom uglu.

Za pokretanje programa su potrebni:
Python v3.7, sa bibliotekama:
- numpy-1.14.0+mkl
- opencv_contrib_python-3.3.0.10
- scikit_learn-0.19.1
- scipy-1.0.0

Sve potrebne biblioteke se mogu skinuti preko pip-a (pip install ime_biblioteke), ili sa stranice:
https://www.lfd.uci.edu/~gohlke/pythonlibs/
I potom ih instalirati preko pip-a (pip install ime_biblioteke.whl).

Usput, vise puta sam rucno prebrojavao, i prilicno sam siguran da u 'video-3.avi' konacna suma treba da bude -57 a ne -64, kako je izracunato u res.txt fajlu sa tacnim rezultatima. Za proveru tacnosti sam koristio izmenjen res.txt fajl (sa vrednoscu -57 za video-3.avi), ali ostavio sam i originalni res.txt fajl, u 'test' folderu (pod imenom 'res - ORIGINAL (s greskom za video-3)').