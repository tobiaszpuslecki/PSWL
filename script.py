"""
https://gist.github.com/xehivs/e15d850cf728a7ed1e0d9edbfd2f694b
Zadania z laboratorium PSW

1. Przygotuj kolekcję co najmniej 4 obrazów (mogą być to obrazy rzeczywiste, syntetyczne, wielowidmowe, nadwidmowe, absolutnie dowolne).
2. Dokonaj selekcji co najmniej trzech metryk oceny metod klasteryzacji.
3. Dokonaj selekcji kilku indukcyjnych metod klasteryzacji.
4. Zrealizuj pętlę walidacji krzyżowej 5x2, która pozwala na poprawną ocenę jakości metod rozpoznawania.
5. Przygotuj krótki opis rezultatów osiągniętych przez wybraną grupę metod na wybranej grupie obrazów.
6. Jeżeli odnajdziesz inną metodę oceny jakości modeli klasteryzacji, która pozwala na wykorzystanie metod transdukcyjnych, nie krępuj się wykorzystać jej w miejsce proponowanego podejścia.
!!! Za ocenę jakości klasteryzacji metryką dokładności, pomimo tego, że zadanie nie jest oceniane, przewidywana jest ocena niedostateczna.

Opis:

A B [X Y Z]

A - numer zdjęcia
B - numer metody (KMeans/MeanShift/MiniBatchKMeans)
X - ocena metryki fowlkes_mallows_score
Y - ocena metryki normalized_mutual_info_score
Z - ocena metryki rand_score

KMeans - k-najbliższych średnich
MeanShift - "algorytm iteracyjnie przypisuje każdy punkt danych w kierunku najbliższego centroida klastra,
    a kierunek do najbliższego centroida klastra jest określany przez miejsce, w którym znajduje się większość punktów w pobliżu.
    Tak więc w każdej iteracji każdy punkt danych będzie się zbliżał do miejsca, w którym znajduje się najwięcej punktów, co prowadzi do centrum klastra."
MiniBatchKMeans - "wykorzystują małe losowe partie danych o stałym rozmiarze, dzięki czemu mogą być przechowywane w pamięci. W każdej iteracji uzyskuje się
    nową losową próbkę ze zbioru danych i wykorzystuje się ją do aktualizacji klastrów, a następnie powtarza się to aż do osiągnięcia zbieżności."

fowlkes_mallows_score - "Mierzy podobieństwo dwóch skupień zbioru punktów.
normalized_mutual_info_score - "Znormalizowana informacja wzajemna pomiędzy dwoma skupieniami. Znormalizowana do <0 (brak wzajemnej informacji) ; 1 (doskonała korelacja)>"
rand_score - "oblicza miarę podobieństwa pomiędzy dwoma klastrami poprzez rozważenie wszystkich par próbek i policzenie par, które są przypisane do tych samych
    lub różnych klastrów w przewidywanych i prawdziwych klastrach."

Najlepsze wyniki dla kMeans, następnie MeanShift i MiniBatchKMeans.
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import disk
from sklearn.cluster import KMeans, DBSCAN, MeanShift
from sklearn.metrics import rand_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import fowlkes_mallows_score, normalized_mutual_info_score, rand_score
from sklearn.cluster import KMeans, MeanShift, MiniBatchKMeans
from sklearn.base import clone

number_of_disks = 3
disk_size = 16
number_of_images = 4
image_size = 64
n_splits=5
n_repeats=2

rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)
metrics = [fowlkes_mallows_score, normalized_mutual_info_score, rand_score]
methods = [KMeans(n_clusters=number_of_disks), MeanShift(bandwidth=.00001), MiniBatchKMeans(n_clusters=number_of_disks),]
results = np.zeros((number_of_images,n_splits*n_repeats,len(methods),len(metrics)))

fig, axs = plt.subplots(number_of_images,len(methods)+2,figsize=(8,30))
axs[0][0].set_title('Image')
axs[0][1].set_title('GroundTruth')
axs[0][2].set_title('KMeans')
axs[0][3].set_title('MeanShift')
axs[0][4].set_title('MiniBatchKMeans')

for i in range(number_of_images):
    image = np.zeros((image_size,image_size,3)).astype(np.uint8)
    ground_truth = np.zeros((image_size,image_size)).astype(np.uint8)

    for d in range(number_of_disks):
        rr, cc = disk((np.random.randint(disk_size, image_size-disk_size),np.random.randint(disk_size, image_size-disk_size)), disk_size)
        rand_value = np.random.randint(0,255)
        image[rr,cc,d] += rand_value
        ground_truth[rr,cc] += rand_value

    noise = np.random.normal(0, 32, size=image.shape)
    image = np.clip(image+noise, 0, 255).astype(np.uint8)

    X = image.reshape(-1,3)
    aa, bb, = np.meshgrid(np.linspace(0,1,image.shape[0]),
                          np.linspace(0,1,image.shape[1]))
    aa = aa.reshape(-1)
    bb = bb.reshape(-1)

    X = np.concatenate((X,
                        aa[:,np.newaxis],
                        bb[:,np.newaxis]), axis=1).astype(float)

    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)

    y = ground_truth.reshape(-1)

    for fold, (train_index, test_index) in enumerate(rskf.split(X, y)):
        for method_id, method in enumerate(methods):
            prediction = clone(method).fit(X[train_index], y[train_index]).predict(X[test_index])
            for metric_id, metric in enumerate(metrics):
                results[i, fold, method_id, metric_id] = metric(y[test_index], prediction)
            print(i, method_id, results[i, fold, method_id])

    predictions_by_method = []
    for methods_id, method in enumerate(methods):
        predictions_by_method.append(clone(method).fit_predict(X,y))

    axs[i][0].imshow(image)
    axs[i][1].imshow(ground_truth)
    axs[i][2].imshow(predictions_by_method[0].reshape(ground_truth.shape))
    axs[i][3].imshow(predictions_by_method[1].reshape(ground_truth.shape))
    axs[i][4].imshow(predictions_by_method[2].reshape(ground_truth.shape))

plt.tight_layout()
plt.savefig('final.png')
