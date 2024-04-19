import math

import pygame

pygame.init()
width, length = 1300, 800
screen = pygame.display.set_mode((width, length))
clock = pygame.time.Clock()
running = True
dt = clock.tick(60) / 1000
snapDist = 20





def DrawWalls(walls):
    for i in range(len(walls) - 1):
        pygame.draw.line(screen, "black", walls[i], walls[i+1], 2)
def DrawCheckPoints(cp):
    for i in range(len(cp)):
        if len(cp[i]) == 2:
            pygame.draw.line(screen, "green", cp[i][0], cp[i][1], 1)


def CreateCheckPoints(checkPoints, pressedDown, current):
    stop = False
    if not pygame.mouse.get_pressed()[0]:
        pressedDown = False
    if pygame.mouse.get_pressed()[0] and not pressedDown:
        current.append(pygame.mouse.get_pos())
        if len(current) == 2:
            checkPoints.append(current)
            current = []
        pressedDown = True
    return checkPoints, pressedDown, current



def CreateWall(wall, pressedDown):
    stop = False
    if not pygame.mouse.get_pressed()[0]:
        pressedDown = False
    if pygame.mouse.get_pressed()[0] and not pressedDown:

        if len(wall) > 2:

            distFromStart = math.sqrt(
                math.pow(wall[0][0] - pygame.mouse.get_pos()[0], 2) + math.pow(wall[0][1] - pygame.mouse.get_pos()[1],
                                                                               2))
            if distFromStart > snapDist:
                wall.append(pygame.mouse.get_pos())
            else:
                wall.append(wall[0])
                stop = True

        else:
            wall.append(pygame.mouse.get_pos())
        pressedDown = True

    DrawWalls(wall)

    return wall, stop, pressedDown


checkPoints = [[], False, []]
outerTrack = [[], False, False]
innerTrack = [[], False, True]
startPos = 0
startDone = False
cpDone = False
trackPrinted = False

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill("white")

    if not outerTrack[1]:
        outerTrack = CreateWall(outerTrack[0], outerTrack[2])
    else:
        DrawWalls(outerTrack[0])
        if not innerTrack[1]:
            innerTrack = CreateWall(innerTrack[0], innerTrack[2])
        else:
            DrawWalls(innerTrack[0])
            if not startDone:
                if pygame.key.get_pressed()[pygame.K_g]:
                    startPos = pygame.mouse.get_pos()
                    startDone = True
            else:
                pygame.draw.circle(screen, 'green', startPos, 3)
                if pygame.key.get_pressed()[pygame.K_c]:
                    cpDone = True
                DrawCheckPoints(checkPoints[0])
                if not cpDone:
                    checkPoints = CreateCheckPoints(checkPoints[0], checkPoints[1], checkPoints[2])
                else:
                    if not trackPrinted:
                        fullTrack = [outerTrack[0], innerTrack[0], startPos, checkPoints[0]]
                        f = open("tracks.txt", "w")
                        f.write(str(fullTrack))
                        print(fullTrack)
                        trackPrinted = True












    pygame.display.flip()
    dt = clock.tick(60) / 1000

pygame.quit()
