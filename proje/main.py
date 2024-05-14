import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
#kütüphaneler yüklendi 

veri= pd.read_csv("./mushroom_cleaned.csv")

ozellikler = veri.drop("class", axis=1)

degerler = veri["class"]

ozellikler_egitim, ozellikler_test, degerler_egitim, degerler_test = train_test_split(ozellikler, degerler, test_size=0.2)

model = DecisionTreeClassifier()
#
model.fit(ozellikler_egitim, degerler_egitim)

score = model.score(ozellikler_test, degerler_test)
print(score)
cap_diameter = int(input("Şapka çapi(mm)"))
cap_shape = int(input("şapka şekli(0:konveks 1:düz 2:çukur"))
gill_attachment = int(input("yaprak eki(0: serbest 1:yapışık"))
gill_color = int(input("yaprak rengi 0:kahverengi 1:pembe 2:koyu kahverengi"))
stem_height = float(input("gövde uzunluğu"))
stem_width = int(input("gövde genişliği"))
stem_color = int(input("gövde rengi 0:beyaz 1:kahverengi 2:siyah"))
season = int(input("mevsim 0:ilkbahar 1:yaz 2:sonbahar 3:kiş"))

yeni_veri = pd.DataFrame([[cap_diameter,cap_shape,gill_attachment,gill_color,stem_height,stem_width,stem_color,season]],
                         columns=["cap-diameter", "cap-shape", "gill-attachment", "gill-color", "stem-height", "stem-width", "stem-color", "season"])
sonuc = model.predict(yeni_veri)

degerler =  {1 : "zehirli", 0 : "temiz"}

print("sonuc" , degerler.get(sonuc[0]))