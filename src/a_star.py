import pygame
import math
from queue import PriorityQueue


WINDOW_WIDTH = 800
UI_WIDTH = 400
WIN = pygame.display.set_mode((WINDOW_WIDTH + UI_WIDTH, WINDOW_WIDTH))
pygame.display.set_caption("Path algos - comparison")

# colors:
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 255, 0)
YELLOW_S = [(250, 249, 180),
            (209, 208, 113),
            (166, 165, 58),
            (117, 116, 21) ]
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165 ,0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)


class Spot:
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.width = width
        self.x = row * width
        self.y = col * width
        self.color = YELLOW_S[0]
        self.weight = 1
        self.neighbours = []
        self.total_rows = total_rows

    def get_pos(self):
        return self.row, self.col
     
    def is_closed(self):
        return self.color == RED
    
    def is_open(self):
        return self.color == GREEN

    def is_barrier(self):
        return self.color == BLACK

    def is_start(self):
        return self.color == ORANGE

    def is_end(self):
        return self.color == TURQUOISE

    def reset(self):
        self.color = YELLOW_S[0]

    def make_start(self):
        self.color = ORANGE
        self.weight = 1

    def make_closed(self):
        self.color = RED

    def make_open(self):
        self.color = GREEN

    def make_barrier(self):
        self.color = BLACK
        self.weight = WINDOW_WIDTH ** 2

    def make_spot_weightedw(self, weight: int):
        self.color = YELLOW_S[weight - 1]
        self.weight = weight

    def make_end(self):
        self.color = TURQUOISE
        
    def make_path(self):
        self.color = PURPLE

    # implement new logic!!!
    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))
    
    def update_neighbours(self, grid):
        pass

    def __lt__(self, other):
        return False
    
# make heuristic function


def make_grid(rows, width):
    grid = []
    gap = width // rows

    for i in range(rows):
        grid.append([])
        for j in range(rows):
            spot = Spot(i, j, gap, rows)
            grid[i].append(spot)

    return grid

def draw_grid(win, rows, width):
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, GREY, (0, i*gap), (width, i*gap))
        for j in range(rows):
            pygame.draw.line(win, GREY, (j*gap, 0), (j*gap, width))


def draw(win, grid, rows, width):
    win.fill(WHITE)

    for row in grid:
        for spot in row:
            spot.draw(win)

    draw_grid(win, rows, width)            
    pygame.display.update()

def get_clicked_pos(pos, rows, width):
    gap = width // rows
    y, x = pos

    row = y // gap
    col = x // gap
    
    return row, col

def main(win, width_PA):
    # width PA - Playable Area
    ROWS = 50
    grid = make_grid(ROWS, width_PA)

    start = None
    end = None

    run = True
    started = False
    
    while run:
        draw(win, grid, ROWS, width_PA)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            
            if started:
                continue
            
            if pygame.mouse.get_pos()[0] > WINDOW_WIDTH: 
                pass
            else:
                if pygame.mouse.get_pressed()[0]:
                    pos = pygame.mouse.get_pos()
                    row, col = get_clicked_pos(pos, ROWS, width_PA)

                    spot = grid[row][col]

                    if not start:
                        start = spot
                        start.make_start()
                    elif not end:
                        end = spot
                        end.make_end()

                    elif spot != end and spot != start:
                        spot.make_barrier()

                elif pygame.mouse.get_pressed()[2]:
                    pass
    pygame.quit()

main(WIN, WINDOW_WIDTH)