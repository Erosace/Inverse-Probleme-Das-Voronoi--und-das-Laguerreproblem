# Inverse-Probleme-Das-Voronoi--und-das-Laguerreproblem

In diesen drei Programmen geht es darum, das direkte und inverse Voronoi- und Laguerreproblem zu lösen.


# Direkte Problem (Calculating_Voronoi_or_Laguerre_Tessellation.py):

Beide direkte Problem können der Definition nach mit dem selben Algortihmus gelöst werden.
Dafür kann die Methode create_tessellation oder display_given_tessellation verwendet werden. Wichtig ist dabei, dass im letzten Aufruf der Methode block=True gesetzt wird. Beispielhafte Befehle sind am Ende der Datei angehängt.


# Inverses Vornoiproblem (Inverting_Voronoi_Tesselation.py):

Anhand dieser Datei kann das inverse Voronoiproblem gelöst werden. Dazu wird die main Methode verwendet.  Sollte bekannt sein, dass das gegebene Mosaik kein Voronoimosaik oder aber fehlerbehaftet ist, so ist es sinnvoll, den Approximationsalgorithmus mit main(approx=True) zu verwenden. Beispielhafte Befehle sind am Ende der Datei zu finden.

# Inverses Laguerreproblem (Invertin_Laguerre_Tessellation.py):

Anhand dieses Problems kann das inverse Laguerreproblem gelöst werden. Dafür stehen zwei Algorithmen zur Verfügung, generating_points und approximated_generating_points. In der ersten Methode wird eine Menge an generierenden gewichteten Punkten berechnet, welche in ein gegebenes Mosaik münden. Im zweiten Algorithmus wird eine Menge an generierenden gewichteten Punkten mit minimalem positiven Radius berechnet.


Die Ergebnisse der inversen Probleme lassen sich anhand der Visualisierungsmethoden der Datei Calculating_Voronoi_or_Laguerre_Tessellation.py überprüfen. Des Weiteren sind detaillierte Informationen in den einzelnen Funktionen hinterlegt. Außerdem kann es von Nöten sein, den approximativen oder optimierenden Algortihmus entsprechend des gegebenen Mosaiks anzupassen.

Copyright:
