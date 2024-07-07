Path finders - program porównujący działanie algorytmów Dijkstra oraz A*

INSTRUKCJA:
Po lewej stronie znajduje się Mapa, po prawej interfejs
Można rysować myszką po mapie, interaktywnie, lewym przyciskiem, a prawym można kasować !!!.

PIERWSZY NODE JEST STARTEM, DRUGI CELEM
Kolejne mogą być wybierane przez interfejs po prawej
Kolejne odcienie żółtego - ścieżki
CZARNY - Bariera


PRZYCISKI CZERWONY I ZIELONY:
DWL - zapisywanie stworzonej mapy do \maps
UPL - pobieranie i wczytywanie gotowych map ze sprawozdania

WŁĄCZANIE ALGORYTMU:
lewy alt - A*
prawy alt - Dijkstra

znane bugi:
Nie działa resetowanie - proszę o wczytywanie pobranych map z UPL ( deepcopy to zło ), proszę o zamykanie i ponowne włączenie programu

ZMIANE WYBORU MAPY PROSZĘ ROBIĆ PRZEZ KOD, w funkcji main() !!! w przedziale od 1 do 4
gdy liczba > 4 zostanie wpisana, można stworzyc nową mape, ale już nie można jej wczytać, bo takowa nie istnieje, prosze tego nie robić
