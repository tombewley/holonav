from GM import GridMap

gme = GridMap.GridMapEnv(gridMap=None, workingDir="maps")
gme.load("maps", "sample.json")

print(gme)