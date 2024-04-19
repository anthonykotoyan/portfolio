import math
import time

from nn import NeuralNetwork as nn
import pygame

pygame.init()
width, length = 1300, 800
screen = pygame.display.set_mode((width, length))
clock = pygame.time.Clock()
running = True
dt = clock.tick(60) / 1000

carImage = pygame.image.load('car.png').convert()

track = [[(119, 55), (322, 44), (516, 41), (835, 45), (1021, 68), (1127, 113), (1185, 175), (1213, 316), (1167, 579),
          (1061, 706), (886, 741), (660, 735), (327, 731), (101, 727), (11, 513), (19, 387), (57, 108), (119, 55)],
         [(227, 249), (233, 477), (355, 564), (535, 605), (701, 598), (932, 585), (1041, 477), (1052, 285), (947, 182),
          (705, 155), (527, 146), (361, 148), (284, 165), (227, 249)], (706, 675),
         [[(732, 590), (728, 756)], [(759, 592), (749, 749)], [(791, 593), (775, 744)], [(815, 593), (802, 747)],
          [(838, 591), (834, 743)], [(863, 590), (876, 743)], [(893, 585), (916, 737)], [(921, 585), (957, 733)],
          [(939, 579), (1041, 715)], [(957, 565), (1084, 685)], [(976, 533), (1123, 641)], [(1011, 503), (1166, 591)],
          [(1033, 466), (1186, 514)], [(1043, 421), (1197, 461)], [(1038, 385), (1209, 400)],
          [(1042, 339), (1214, 353)], [(1036, 300), (1213, 299)], [(1024, 271), (1198, 204)], [(995, 237), (1142, 117)],
          [(969, 213), (1090, 85)], [(947, 194), (979, 63)], [(927, 187), (925, 58)], [(894, 181), (887, 46)],
          [(819, 172), (831, 41)], [(788, 162), (791, 39)], [(757, 167), (754, 41)], [(706, 155), (697, 39)],
          [(657, 155), (639, 38)], [(614, 155), (595, 40)], [(563, 137), (552, 34)], [(531, 145), (514, 45)],
          [(502, 155), (471, 27)], [(455, 150), (422, 33)], [(409, 153), (385, 44)], [(367, 152), (341, 41)],
          [(325, 161), (307, 49)], [(286, 169), (221, 58)], [(269, 205), (94, 89)], [(237, 233), (71, 161)],
          [(227, 260), (61, 203)], [(233, 295), (53, 253)], [(239, 356), (48, 309)], [(242, 387), (30, 367)],
          [(241, 441), (14, 426)], [(245, 477), (8, 503)], [(278, 502), (34, 548)], [(311, 529), (74, 623)],
          [(332, 550), (89, 693)], [(361, 564), (173, 715)], [(397, 577), (257, 727)], [(403, 578), (327, 728)],
          [(427, 581), (397, 716)], [(461, 593), (460, 734)], [(499, 593), (503, 729)], [(555, 603), (567, 729)],
          [(627, 606), (634, 726)], [(675, 599), (683, 734)]]]


def sign(x):
    if x != 0:
        return abs(x) / x
    return 0


def line_intersection(a, b, c, d):
    d1 = (
            (a[0] - b[0]) * (c[1] - d[1]) - (a[1] - b[1]) * (c[0] - d[0]))
    d2 = (
            (a[0] - b[0]) * (c[1] - d[1]) - (a[1] - b[1]) * (c[0] - d[0]))
    if d1 == 0 or d2 == 0:
        return [False, []]
    else:
        t = ((a[0] - c[0]) * (c[1] - d[1]) - (a[1] - c[1]) * (c[0] - d[0])) / d1
        u = ((a[0] - c[0]) * (a[1] - b[1]) - (a[1] - c[1]) * (a[0] - b[0])) / d2

    # check if line actually intersect
    if 0 <= t <= 1 and 0 <= u <= 1:
        return [True, (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))]
    else:
        return [False, []]


def BoxCollision(corners, line):
    for i in range(4):
        if i == 3:
            j = -1
        else:
            j = i
        boxLine = [corners[j], corners[j + 1]]
        collisionInfo = line_intersection(line[0], line[1], boxLine[0], boxLine[1])
        if collisionInfo[0]:
            return True

    return False


def DrawCheckPoints(cp):
    for i in range(len(cp)):
        if len(cp[i]) == 2:
            pygame.draw.line(screen, "green", cp[i][0], cp[i][1], 1)


def DrawTrack(track, drawCP):
    for wall in range(2):
        for i in range(len(track[wall]) - 1):
            pygame.draw.line(screen, "black", track[wall][i], track[wall][i + 1], 2)
    pygame.draw.circle(screen, 'green', track[2], 3)
    if drawCP:
        DrawCheckPoints(track[3])


class Agent:
    maxSpeed = 6
    driftLength = 4
    maxTurnSpeed = 2.5
    driftFriction = 10
    friction = 0.4
    baseTSValue = -.3
    ShiftUpTS = 1.1
    driftSpeed = 1
    minDriftAngle = 20
    normalSlip = .2
    handBrakeSlip = normalSlip * 3
    errorCorrectionStrength = 15
    gen = 0
    genLength = 5
    startTime = time.time()
    boxSize = 2.5

    numBestAgents = 2
    mutateFactor = .5
    mutationChance = .25
    allAgents = []
    allMaxCP = 0
    nnBest = 0

    def __init__(self):
        self.image = pygame.image.load('car.png').convert_alpha()
        self.size = 3
        self.image = pygame.transform.scale(self.image, (self.size * 10, self.size * 10))
        self.acc = 3
        self.turnSpeed = 2.8
        self.slip = Agent.normalSlip
        self.dir = 0
        self.angle = 0
        self.vel = pygame.Vector2(0, 0)
        self.pos = pygame.Vector2(track[2][0], track[2][1])
        self.speed = 0
        self.car = self.image.get_rect(center=(self.pos.x, self.pos.y))
        self.hbColor = "green"
        self.carCorners = [(self.pos.x - self.size * Agent.boxSize, self.pos.y - self.size * Agent.boxSize),
                           (self.pos.x + self.size * Agent.boxSize, self.pos.y - self.size * Agent.boxSize),
                           (self.pos.x + self.size * Agent.boxSize, self.pos.y + self.size * Agent.boxSize),
                           (self.pos.x - self.size * Agent.boxSize, self.pos.y + self.size * Agent.boxSize)]
        self.angles = [-90, -37.5, 0, 37.5, 90]
        self.vision = []
        self.totalCP = 0
        self.nextCP = 0
        self.runAgent = True

        layers = [len(self.angles) + 1, 3, 5]
        self.nn = nn(layers)
        self.nn.randomize()
        Agent.allAgents.append(self)

    def Vision(self):

        positions = []
        endPositions = []
        rayDist = 100000
        dists = []
        for i in range(len(self.angles)):
            pos = pygame.Vector2(math.cos(math.radians(self.angle + self.angles[i])) * rayDist + self.pos.x,
                                 math.sin(math.radians(self.angle + self.angles[i])) * rayDist + self.pos.y)
            positions.append(pos)
            endPos = pos
            wallDists = []
            for walls in range(2):
                for wall in range(len(track[walls]) - 1):
                    intersection = line_intersection(self.pos, pos, track[walls][wall], track[walls][wall + 1])
                    if intersection[0]:
                        wallDist = math.dist(self.pos, pygame.Vector2(intersection[1][0], intersection[1][1]))
                        wallDists.append(wallDist)

                        if wallDist == min(wallDists):
                            endPos = pygame.Vector2(intersection[1][0], intersection[1][1])
            dists.append(min(wallDists))
            endPositions.append(endPos)
        return [dists, endPositions]

    def DrawVision(self, endPositions):
        for i in range(len(endPositions)):
            pygame.draw.line(screen, "purple", self.pos, endPositions[i], 1)
            pygame.draw.circle(screen, "orange", endPositions[i], 5)

    def HitBox(self, draw):
        if draw:
            for i in range(4):
                if i == 3:
                    j = -1
                else:
                    j = i

                pygame.draw.line(screen, self.hbColor, self.carCorners[j], self.carCorners[j + 1], 2)

    def DrawCar(self):

        self.car = self.image.get_rect(center=(self.pos.x, self.pos.y))
        rotated_image = pygame.transform.rotate(self.image, -self.angle)
        # Update the car's rectangle to the new rotated image's rectangle
        self.car = rotated_image.get_rect(center=self.car.center)
        # Draw the rotated image onto the screen
        screen.blit(rotated_image, self.car)

    def DriftTrail(self, dist, wid):
        driftAngle = abs(((self.angle - self.dir + 540) % 360 - 180))
        if driftAngle >= Agent.minDriftAngle and abs(self.speed) > Agent.driftSpeed:
            endPos = pygame.Vector2(
                sign(self.speed) * math.cos(math.radians(self.dir)) * dist + self.pos.x + self.size * 2,
                sign(self.speed) * math.sin(math.radians(self.dir)) * dist + self.pos.y)
            pygame.draw.line(screen, (30, 30, 30), self.pos, endPos, wid)
            endPos = pygame.Vector2(
                sign(self.speed) * math.cos(math.radians(self.dir)) * dist + self.pos.x - self.size * 2,
                sign(self.speed) * math.sin(math.radians(self.dir)) * dist + self.pos.y)
            pygame.draw.line(screen, (30, 30, 30), self.pos, endPos, wid)

    def DrawAngle(self, dist, draw):
        if draw:
            endPos = pygame.Vector2(math.cos(math.radians(self.dir)) * dist + self.pos.x,
                                    math.sin(math.radians(self.dir)) * dist + self.pos.y)
            pygame.draw.line(screen, "blue", self.pos, endPos, self.size)
            endPos = pygame.Vector2(math.cos(math.radians(self.angle)) * dist + self.pos.x,
                                    math.sin(math.radians(self.angle)) * dist + self.pos.y)
            pygame.draw.line(screen, "red", self.pos, endPos, self.size)

    def ApplyVelocity(self):
        self.carCorners = [(self.pos.x - self.size * Agent.boxSize, self.pos.y - self.size * Agent.boxSize),
                           (self.pos.x + self.size * Agent.boxSize, self.pos.y - self.size * Agent.boxSize),
                           (self.pos.x + self.size * Agent.boxSize, self.pos.y + self.size * Agent.boxSize),
                           (self.pos.x - self.size * Agent.boxSize, self.pos.y + self.size * Agent.boxSize)]
        self.vision = self.Vision()

        self.pos.x += self.vel.x
        self.pos.y += self.vel.y

    def ResetAgent(self):
        self.angle = 0
        self.dir = 0
        self.speed = 0
        self.totalCP = 0
        self.nextCP = 0
        self.runAgent = True
        self.pos = pygame.Vector2(track[2][0], track[2][1])

    def AgentDeath(self):
        self.runAgent = False

    def TrackCollisions(self):
        for walls in range(2):
            for wall in range(len(track[walls]) - 1):
                coll = BoxCollision(self.carCorners, (track[walls][wall], track[walls][wall + 1]))
                if coll:
                    self.AgentDeath()

    def TrackCheckpoints(self):
        coll = BoxCollision(self.carCorners, track[3][self.nextCP])
        pygame.draw.line(screen, "green", track[3][self.nextCP][0], track[3][self.nextCP][1], 1)
        if coll:
            self.totalCP += 1
            if self.nextCP == len(track[3]) - 1:
                self.nextCP = 0
            else:
                self.nextCP += 1

    def BorderCollisions(self):
        if 0 >= self.pos.x:
            self.AgentDeath()

        if self.pos.x >= width:
            self.AgentDeath()

        if 0 >= self.pos.y:
            self.AgentDeath()

        if self.pos.y >= length:
            self.AgentDeath()

    def ApplyDirection(self):
        if abs(self.speed) >= Agent.maxSpeed:
            self.speed = Agent.maxSpeed * sign(self.speed)
        self.vel.x = -math.cos(math.radians(self.dir)) * self.speed
        self.vel.y = -math.sin(math.radians(self.dir)) * self.speed

    def Controls(self, outputs):
        angleError = ((self.angle - self.dir + 540) % 360 - 180)
        if outputs[0]:
            self.speed -= self.acc * dt
        if abs(self.speed) > Agent.driftSpeed:
            self.speed += Agent.driftFriction * dt * (abs(angleError) / 180) * -sign(self.speed)
        self.speed += Agent.friction * -sign(self.speed) * dt
        if outputs[4]:
            self.slip = Agent.handBrakeSlip
            if self.speed < 0:
                self.speed += self.acc * dt
            else:
                self.speed = 0
        else:
            self.slip = Agent.normalSlip

        if outputs[1]:
            if self.speed < 0:
                self.speed += self.acc * dt
            else:
                self.speed = 0
        if outputs[2]:
            # angle error changes positively
            self.angle += self.turnSpeed
            if self.speed != 0:
                self.dir += self.turnSpeed * (1 - self.slip)
            else:
                self.dir = self.angle
            if angleError < 0:
                self.dir += sign(angleError) * Agent.errorCorrectionStrength * (abs(angleError) / 180)


        elif outputs[3]:
            # angle error changes negatively
            self.angle -= self.turnSpeed
            if self.speed != 0:
                self.dir -= self.turnSpeed * (1 - self.slip)
            else:
                self.dir = self.angle
            if angleError > 0:
                self.dir += sign(angleError) * Agent.errorCorrectionStrength * (abs(angleError) / 180)


        else:
            self.dir += sign(angleError) * (Agent.driftLength * .25)
        b = 4 * Agent.maxTurnSpeed / Agent.maxSpeed ** 2
        self.turnSpeed = -b * (Agent.baseTSValue + abs(self.speed)) * (
                Agent.baseTSValue - Agent.maxSpeed + abs(self.speed)) + Agent.ShiftUpTS

    @staticmethod
    def RestartGen():
        Agent.startTime = time.time()
        allCP = []
        bestAgents = []
        randomizeAll = False
        cpThreshold = 3

        for i in range(len(Agent.allAgents)):
            allCP.append(Agent.allAgents[i].totalCP)
        for agent in range(Agent.numBestAgents):
            maxCP = max(allCP)
            bestAgents.append(allCP.index(maxCP))
            if maxCP> Agent.allMaxCP:
                Agent.nnBest = allCP.index(maxCP)
            allCP.insert(allCP.index(maxCP), 0)
            allCP.remove(maxCP)
            if maxCP <= cpThreshold:
                randomizeAll = True
        for i in range(len(Agent.allAgents)):
            if not randomizeAll:
                for j in range(len(bestAgents)):
                    if i == bestAgents[j]:
                        Agent.allAgents[i].ResetAgent()
                        break
                else:
                    Agent.allAgents[i].nn = Agent.allAgents[bestAgents[0]].nn
                    Agent.allAgents[i].nn.mutate(Agent.mutationChance, Agent.mutateFactor)
                    Agent.allAgents[i].ResetAgent()
            else:
                Agent.allAgents[i].nn.randomize()
                Agent.allAgents[i].ResetAgent()
    @staticmethod
    def ManageGen():


        if time.time() - Agent.startTime > Agent.genLength:
            Agent.RestartGen()
            Agent.gen += 1
            Agent.startTime = time.time()

    def RunAgent(self):
        nnInputs = self.vision[0]
        nnInputs.append(self.speed)
        nnOutputs = self.nn.run(nnInputs, nn.Tanh)
        agentInput = []
        for i in range(len(nnOutputs)):
            if nnOutputs[i] > 0:
                agentInput.append(True)
            else:
                agentInput.append(False)

        self.Controls(agentInput)

    @staticmethod
    def UpdateAgents():
        Agent.ManageGen()
        deadAgents = 0
        for i in range(len(Agent.allAgents)):
            if Agent.allAgents[i].runAgent:
                Agent.allAgents[i].ApplyVelocity()
                Agent.allAgents[i].ApplyDirection()
                Agent.allAgents[i].RunAgent()

                Agent.allAgents[i].TrackCheckpoints()
                Agent.allAgents[i].TrackCollisions()
                Agent.allAgents[i].BorderCollisions()

                Agent.allAgents[i].DriftTrail(10 * Agent.allAgents[i].size, 4 * Agent.allAgents[i].size)
                Agent.allAgents[i].DrawAngle(50, False)
                Agent.allAgents[i].DrawCar()

            else:
                deadAgents += 1
        if deadAgents >= len(Agent.allAgents):
            Agent.RestartGen()
        allCP = []
        for i in range(len(Agent.allAgents)):
            allCP.append(Agent.allAgents[i].totalCP)
        Agent.allAgents[allCP.index(max(allCP))].HitBox(True)
        Agent.allAgents[allCP.index(max(allCP))].DrawVision(Agent.allAgents[allCP.index(max(allCP))].vision[1])


numAgents = 10
for i in range(numAgents):
    Agent()


class Car:
    maxSpeed = 6
    driftLength = 4
    maxTurnSpeed = 2.5
    driftFriction = 10
    friction = 0.4
    baseTSValue = -.3
    ShiftUpTS = 1.1
    driftSpeed = 1
    minDriftAngle = 20
    normalSlip = .2
    handBrakeSlip = normalSlip * 3
    errorCorrectionStrength = 15

    boxSize = 2.5

    def __init__(self):
        self.image = pygame.image.load('car.png').convert_alpha()
        self.size = 3
        self.image = pygame.transform.scale(self.image, (self.size * 10, self.size * 10))
        self.acc = 3
        self.turnSpeed = 2.8
        self.slip = Car.normalSlip
        self.dir = 0
        self.angle = 0
        self.vel = pygame.Vector2(0, 0)
        self.pos = pygame.Vector2(track[2][0], track[2][1])
        self.speed = 0
        self.car = self.image.get_rect(center=(self.pos.x, self.pos.y))
        self.hbColor = "green"
        self.carCorners = []
        self.vision = []

    def Vision(self):
        angles = [-80, -45, -15, -5, 0, 5, 15, 45, 80]
        positions = []
        endPositions = []
        rayDist = 100000
        dists = []
        for i in range(len(angles)):
            pos = pygame.Vector2(math.cos(math.radians(self.angle + angles[i])) * rayDist + self.pos.x,
                                 math.sin(math.radians(self.angle + angles[i])) * rayDist + self.pos.y)
            positions.append(pos)
            endPos = pos
            wallDists = []
            for walls in range(2):
                for wall in range(len(track[walls]) - 1):
                    intersection = line_intersection(self.pos, pos, track[walls][wall], track[walls][wall + 1])
                    if intersection[0]:
                        wallDist = math.dist(self.pos, pygame.Vector2(intersection[1][0], intersection[1][1]))
                        wallDists.append(wallDist)

                        if wallDist == min(wallDists):
                            endPos = pygame.Vector2(intersection[1][0], intersection[1][1])
            dists.append(min(wallDists))
            endPositions.append(endPos)
        return [dists, endPositions]

    def DrawVision(self, endPositions):
        for i in range(len(endPositions)):
            pygame.draw.line(screen, "purple", self.pos, endPositions[i], 1)
            pygame.draw.circle(screen, "orange", endPositions[i], 5)

    def HitBox(self, draw):
        if draw:
            for i in range(4):
                if i == 3:
                    j = -1
                else:
                    j = i

                pygame.draw.line(screen, self.hbColor, self.carCorners[j], self.carCorners[j + 1], 2)

    def DrawCar(self):

        self.car = self.image.get_rect(center=(self.pos.x, self.pos.y))
        rotated_image = pygame.transform.rotate(self.image, -self.angle)
        # Update the car's rectangle to the new rotated image's rectangle
        self.car = rotated_image.get_rect(center=self.car.center)
        # Draw the rotated image onto the screen
        screen.blit(rotated_image, self.car)

    def DriftTrail(self, dist, wid):
        driftAngle = abs(((self.angle - self.dir + 540) % 360 - 180))
        if driftAngle >= Car.minDriftAngle and abs(self.speed) > Car.driftSpeed:
            endPos = pygame.Vector2(
                sign(self.speed) * math.cos(math.radians(self.dir)) * dist + self.pos.x + self.size * 2,
                sign(self.speed) * math.sin(math.radians(self.dir)) * dist + self.pos.y)
            pygame.draw.line(screen, (30, 30, 30), self.pos, endPos, wid)
            endPos = pygame.Vector2(
                sign(self.speed) * math.cos(math.radians(self.dir)) * dist + self.pos.x - self.size * 2,
                sign(self.speed) * math.sin(math.radians(self.dir)) * dist + self.pos.y)
            pygame.draw.line(screen, (30, 30, 30), self.pos, endPos, wid)

    def DrawAngle(self, dist, draw):
        if draw:
            endPos = pygame.Vector2(math.cos(math.radians(self.dir)) * dist + self.pos.x,
                                    math.sin(math.radians(self.dir)) * dist + self.pos.y)
            pygame.draw.line(screen, "blue", self.pos, endPos, self.size)
            endPos = pygame.Vector2(math.cos(math.radians(self.angle)) * dist + self.pos.x,
                                    math.sin(math.radians(self.angle)) * dist + self.pos.y)
            pygame.draw.line(screen, "red", self.pos, endPos, self.size)

    def ApplyVelocity(self):
        self.carCorners = [(self.pos.x - self.size * Car.boxSize, self.pos.y - self.size * Car.boxSize),
                           (self.pos.x + self.size * Car.boxSize, self.pos.y - self.size * Car.boxSize),
                           (self.pos.x + self.size * Car.boxSize, self.pos.y + self.size * Car.boxSize),
                           (self.pos.x - self.size * Car.boxSize, self.pos.y + self.size * Car.boxSize)]
        self.vision = self.Vision()

        self.pos.x += self.vel.x
        self.pos.y += self.vel.y

    def ResetPos(self):
        self.angle = 0
        self.dir = 0
        self.speed = 0
        self.pos = pygame.Vector2(track[2][0], track[2][1])

    def TrackCollisions(self):
        for walls in range(2):
            for wall in range(len(track[walls]) - 1):
                coll = BoxCollision(self.carCorners, (track[walls][wall], track[walls][wall + 1]))
                if coll:
                    self.ResetPos()

    def BorderCollisions(self):
        if 0 >= self.pos.x:
            self.ResetPos()

        if self.pos.x >= width:
            self.ResetPos()

        if 0 >= self.pos.y:
            self.ResetPos()

        if self.pos.y >= length:
            self.ResetPos()

    def ApplyDirection(self):
        if abs(self.speed) >= Car.maxSpeed:
            self.speed = Car.maxSpeed * sign(self.speed)
        self.vel.x = -math.cos(math.radians(self.dir)) * self.speed
        self.vel.y = -math.sin(math.radians(self.dir)) * self.speed

    def Controls(self, keys):
        angleError = ((self.angle - self.dir + 540) % 360 - 180)
        if keys[pygame.K_w]:
            self.speed -= self.acc * dt
        if abs(self.speed) > Car.driftSpeed:
            self.speed += Car.driftFriction * dt * (abs(angleError) / 180) * -sign(self.speed)
        self.speed += Car.friction * -sign(self.speed) * dt
        if keys[pygame.K_SPACE]:
            self.slip = Car.handBrakeSlip
            if self.speed < 0:
                self.speed += self.acc * dt
            else:
                self.speed = 0
        else:
            self.slip = Car.normalSlip

        if keys[pygame.K_s]:
            if self.speed < 0:
                self.speed += self.acc * dt
            else:
                self.speed = 0
        if keys[pygame.K_d]:
            # angle error changes positively
            self.angle += self.turnSpeed
            if self.speed != 0:
                self.dir += self.turnSpeed * (1 - self.slip)
            else:
                self.dir = self.angle
            if angleError < 0:
                self.dir += sign(angleError) * Car.errorCorrectionStrength * (abs(angleError) / 180)


        elif keys[pygame.K_a]:
            # angle error changes negatively
            self.angle -= self.turnSpeed
            if self.speed != 0:
                self.dir -= self.turnSpeed * (1 - self.slip)
            else:
                self.dir = self.angle
            if angleError > 0:
                self.dir += sign(angleError) * Car.errorCorrectionStrength * (abs(angleError) / 180)


        else:
            self.dir += sign(angleError) * (Car.driftLength * .25)
        b = 4 * Car.maxTurnSpeed / Car.maxSpeed ** 2
        self.turnSpeed = -b * (Car.baseTSValue + abs(self.speed)) * (
                Car.baseTSValue - Car.maxSpeed + abs(self.speed)) + Car.ShiftUpTS

    def UpdateCar(self, keys):

        self.ApplyVelocity()
        self.ApplyDirection()
        self.Controls(keys)
        self.TrackCollisions()
        self.BorderCollisions()

        # self.DrawVision(self.vision[1])
        self.DriftTrail(10 * self.size, 4 * self.size)
        self.DrawAngle(50, False)
        self.DrawCar()
        self.HitBox(False)


car1 = Car()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill("grey")
    keys = pygame.key.get_pressed()
    DrawTrack(track, False)
    # car1.UpdateCar(keys)
    Agent.UpdateAgents()

    pygame.display.flip()
    dt = clock.tick(60) / 1000

pygame.quit()
