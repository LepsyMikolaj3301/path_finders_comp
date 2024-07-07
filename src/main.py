import pygame
import math
from queue import PriorityQueue
from functools import wraps
import time
import heapq
import os
import copy
from additional_functions import *
CLOCK = pygame.time.Clock()
FPS = 60
PA_WIDTH = 600
UI_WIDTH = 400



pygame.init()

WIN = pygame.display.set_mode((PA_WIDTH + UI_WIDTH, PA_WIDTH))
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
        self.weight_indice = 1
    
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
    
    def get_weight(self):
        return self.weight
    
    def get_weight_indice(self):
        return self.weight_indice

    def reset(self):
        self.color = YELLOW_S[0]
        self.weight = 1

    def make_start(self):
        self.color = ORANGE
        self.weight = 0

    def make_end(self):
        self.color = TURQUOISE
        
    def make_path(self):
        self.color = PURPLE

    def make_closed(self):
        self.color = RED

    def make_open(self):
        self.color = GREEN

    # def make_barrier(self):
        

    def make_spot_weighted(self, weight_indice):
        self.weight_indice = weight_indice
        if isinstance(weight_indice, int) and weight_indice != 0:
            # custom weights:
            weights = {1: 1,
                       2: 10,
                       3: 100,
                       4: 1000}
            
            self.color = YELLOW_S[weight_indice - 1]
            self.weight = weights[weight_indice]
        else:
            self.color = BLACK
            self.weight = PA_WIDTH ** 2

    

    # implement new logic!!!
    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))
    
    def update_neighbours(self, grid):
        self.neighbours = []
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier(): #DOWN
            self.neighbours.append(grid[self.row + 1][self.col])

        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier(): #UP
            self.neighbours.append(grid[self.row - 1][self.col])

        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier(): #RIGHT
            self.neighbours.append(grid[self.row][self.col + 1])

        if self.row > 0 and not grid[self.row][self.col - 1].is_barrier(): #LEFT
            self.neighbours.append(grid[self.row][self.col - 1])
        

    def __lt__(self, other):
        return False

# Button
class Button:
    def __init__(self, x, y, rect_width, rect_height, color, text, callback) -> None:
        button_font = pygame.font.SysFont('Corbel', 35)
        self.raw_text = text
        self.text = button_font.render(text , True , BLACK)
        self.text_width, self.text_height = button_font.size(text)
        self.color = color
        self.x = x
        self.y = y
        self.rect_width = rect_width
        self.rect_height = rect_height
        self.callback = callback
        

    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.rect_width, self.rect_height))
        win.blit(self.text, (self.x + self.rect_width/2 - self.text_width/2, self.y + self.rect_height/2 - self.text_height/2))

    def check_input(self, position):
        if self.x <= position[0] <= self.x + self.rect_width and self.y <= position[1] <= self.y + self.rect_height:
            return self.callback()
            
# TIME IT FUNCTION
def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Algo took: {total_time:.4f} seconds')
        return result
    return timeit_wrapper


def download_created_map():


    # LICZBA MAP - WYBIERAMY PRZEZ ZMIENNĄ W MAIN !!!
    # ZMIENNA GLOBALNA
    map_num = map_pick


    map_name = r"".join(["maps_", str(map_num), ".txt"])
    CURRENT_DIR = os.getcwd()
    maps_dir = CURRENT_DIR + r'\maps'

    if not os.path.exists(maps_dir):
        os.makedirs(maps_dir)
    os.chdir(maps_dir)
    # saving the current state of global grid
    if not os.path.isfile(maps_dir + '\\' + map_name):
        with open(map_name, 'a') as file:
            file.write(" ")
    with open(map_name, 'w') as file:
        
        save_grid = []
        for row in grid:
            temp = []
            line = "|"
            for spot in row:
                if spot.is_start():
                    temp.append("S")
                elif spot.is_end():
                    temp.append("E")
                elif spot.is_barrier():
                    temp.append("B")
                else:
                    temp.append(str(spot.get_weight_indice()))
            
            save_grid.append(line.join(temp))
        
        for line in save_grid:
            file.write(line + '\n')
    os.chdir(CURRENT_DIR)
    return ["downlad"]
    

def upload_created_map():

    # LICZBA MAP - WYBIERAMY PRZEZ ZMIENNĄ W MAIN!!!
    map_num = map_pick

    map_name = r"".join(["maps_", str(map_num), ".txt"])
    CURRENT_DIR = os.getcwd()
    maps_dir = CURRENT_DIR + r'\maps'
    try:
        os.chdir(maps_dir)
    except Exception("No such directory!"):
        return
    
    with open(map_name, 'r') as f:
        lines = [line.rstrip('\n') for line in f]
        temp = [line.split("|") for line in lines]

        for i, line in enumerate(temp):
            for j, item in enumerate(line):
                if item == "S":
                    grid[i][j].make_start()
                    start = grid[i][j]
                elif item == "E":
                    grid[i][j].make_end()
                    end = grid[i][j]
                elif item == "B":
                    grid[i][j].make_spot_weighted(0)
                else:
                    grid[i][j].make_spot_weighted(int(item))
    os.chdir(CURRENT_DIR)
    return ["upload", start, end]


# make heuristic function
def h(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)


def make_grid(rows, width):
    grid = []
    gap = width // rows

    for i in range(rows):
        grid.append([])
        for j in range(rows):
            spot = Spot(i, j, gap, rows)
            grid[i].append(spot)
    # create obramówka xd
    # 49 = rows - 1
    for i, spot in enumerate(grid[0]):
        grid[i][0].make_spot_weighted(0)
    for j, spot in enumerate(grid[49]):
        grid[j][49].make_spot_weighted(0)
    for i, rows in enumerate(grid):
        for j, spot in enumerate(rows):
            grid[0][j].make_spot_weighted(0)
            grid[49][j].make_spot_weighted(0)
    
    return grid

def draw_grid(win, rows, width):
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, GREY, (0, i*gap), (width, i*gap))
        for j in range(rows + 1):
            pygame.draw.line(win, GREY, (j*gap, 0), (j*gap, width))
    
def draw_current_color(win, x, y, current_weight):
    button_font = pygame.font.SysFont('Corbel',20)
    text = "Current Color:"
    text_width, text_height = button_font.size(text)
    if current_weight:
        color = YELLOW_S[current_weight - 1]
    else:
        color = BLACK
    
    pygame.draw.rect(win, color, (x + text_width + 5, y, 20, 20))
    win.blit(button_font.render(text , True , BLACK), (x, y))


def draw(win, grid, rows, width, buttons=[]):
    win.fill(WHITE)

    for row in grid:
        for spot in row:
            spot.draw(win)
    
    
    draw_grid(win, rows, width)
    draw_current_color(win, PA_WIDTH + 5, 1, CURRENT_WEIGHT)
    if buttons:
        for button in buttons:
            button.draw(win)

    pygame.display.update()



# ALGORITHMS
def reconstruct_path(came_from, current, draw):
    total_cost = 0
    while current in came_from:
        current = came_from[current]
        if current == None:
            break
        total_cost += current.get_weight()
        current.make_path()
        draw()
    return total_cost
    

@timeit
def dijkstra(draw, grid, start, end):
    queue = []
    count = 0
    heapq.heappush(queue, (0, count, start))
    cost_visited = {spot: float("inf") for row in grid for spot in row}
    cost_visited[start] = 0
    # queue_hash = {start}
    visited = {start: None}
    

    while queue:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        # cur_cost, cur_count,
        cur_cost, cur_count, cur_spot = heapq.heappop(queue)
        

        if cur_spot == end:
            # make path
            total_cost = reconstruct_path(visited, end, draw)
            print_algo_info(algo_name="Dijkstra", count=count, total_cost=total_cost)
            end.make_end()
            return True
        
        if cur_cost > cost_visited[cur_spot]:
            continue
        for neighbour in cur_spot.neighbours:
            neighbour_cost = neighbour.weight
            new_cost = cost_visited[cur_spot] = neighbour_cost
            # temp_cost = cur_cost + neighbour.weight
            # if temp_cost < cost_visited[neighbour]:
            if neighbour not in visited:
                visited[neighbour] = cur_spot
                if new_cost < cost_visited[neighbour]:
                    count += 1
                    cost_visited[neighbour] = new_cost
                    heapq.heappush(queue, (new_cost, count, neighbour))
                    neighbour.make_open()
        draw()

        if cur_spot != start:
            cur_spot.make_closed() 
    return False

@timeit
def a_star(draw, grid, start, end):
    count = 0
    
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}
    # The current shortest distence - G_score
    g_score = {spot: float("inf") for row in grid for spot in row}
    g_score[start] = 0
    # Predicted distance to end node - F_score
    f_score = {spot: float("inf") for row in grid for spot in row}
    f_score[start] = h(start.get_pos(), end.get_pos())

    # see if somethings in the set already
    open_set_hash = {start}

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        
        # get the current smallest element from the Priority que
        current = open_set.get()[2]
        open_set_hash.remove(current)

        if current == end:
            # make path
            total_cost = reconstruct_path(came_from, end, draw)
            print_algo_info(algo_name="A*", count=count, total_cost=total_cost)
            end.make_end()
            return True
        
        # consider all neighbours
        for neighbour in current.neighbours:
            # calculate their g_score
            temp_g_score = g_score[current] + neighbour.weight # dodaj weight w tym miejscu
            # if we found a better path!
            if temp_g_score < g_score[neighbour]:
                came_from[neighbour] = current
                g_score[neighbour] = temp_g_score
                f_score[neighbour] = temp_g_score + h(neighbour.get_pos(), end.get_pos())
                if neighbour not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbour], count, neighbour))
                    open_set_hash.add(neighbour)
                    neighbour.make_open()
        draw()

        if current != start:
            current.make_closed() 
    return False

    

def main(win, width_PA):
    # width PA - Playable Area
    
    global grid, ROWS, start, end, map_pick
    ROWS = 50 # NOT DYNAMIC - NOT TO CHANGE AT CURRENT STATE !!!
    grid = make_grid(ROWS, width_PA)
    

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # WHAT MAP SHOULD WE LOAD?
    map_pick = 4
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    

    # BUTTONS 
    # buttons (x, y) -> (width_PA + x' , y')
    button_width, button_height = width_PA*0.3, width_PA*0.1
    button_gap = button_height * 1.2

    buttons_UI = [  Button(width_PA + UI_WIDTH*0.2, width_PA*0.07, button_width, button_height, YELLOW_S[1], "weight 2", weight_2),
                    Button(width_PA + UI_WIDTH*0.2, width_PA*0.07 + button_gap, button_width, button_height, YELLOW_S[2], "weight 3", weight_3),
                    Button(width_PA + UI_WIDTH*0.2, width_PA*0.07 + 2*button_gap, button_width, button_height, YELLOW_S[3], "weight 4", weight_4),
                    Button(width_PA + UI_WIDTH*0.2, width_PA*0.07 + 3*button_gap, button_width, button_height, BLACK, " ", weight_barrier),
                    # buttons for dowloading and uploading maps
                    Button(width_PA + UI_WIDTH*0.2, width_PA*0.07 + 4*button_gap, button_width*0.5 - 2, button_height, RED, "DWL", callback=download_created_map),
                    Button(width_PA + UI_WIDTH*0.2 + button_width*0.5 + 2, width_PA*0.07 + 4*button_gap, button_width*0.5, button_height, GREEN, "UPL", callback=upload_created_map),
                    # Button to reset the map after algorithm
                    # Button(width_PA + UI_WIDTH*0.2, width_PA*0.07 + 5*button_gap, button_width, button_height, PURPLE, "RESET", reset_grid)
                    # this doesnt work
                    ]
    
    # global variables
    start = None
    end = None

    run = True
    started = False
    
    global CURRENT_WEIGHT
    CURRENT_WEIGHT = 1
    
    while run:
        draw(win, grid, ROWS, width_PA, buttons=buttons_UI)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            
            
            if started:
                continue
            
            if event.type == pygame.KEYDOWN:
                # backup_grid = grid
                if event.key == pygame.K_LALT and not started:
                    for row in grid:
                        for spot in row:
                            spot.update_neighbours(grid)
                    a_star(lambda: draw(win, grid, ROWS, width_PA), grid, start, end)
                if event.key == pygame.K_RALT and not started:
                    for row in grid:
                        for spot in row:
                            spot.update_neighbours(grid)
                    dijkstra(lambda: draw(win, grid, ROWS, width_PA), grid, start, end)

            
            if 0 < pygame.mouse.get_pos()[0] > PA_WIDTH:
                # The UI on the right hand side essa
                if pygame.mouse.get_pressed()[0]:
                    pos = pygame.mouse.get_pos()
                    for button in buttons_UI:
                        new_update = button.check_input(pos)
                        if new_update is not None:
                            if new_update[0] == "weight":
                                CURRENT_WEIGHT = new_update[1]
                            if new_update[0] == "upload":
                                start = new_update[1]
                                end = new_update[2]
                        # print(CURRENT_WEIGHT)
            elif 0 < pygame.mouse.get_pos()[1] < PA_WIDTH and 0 < pygame.mouse.get_pos()[0] < PA_WIDTH: 
                if pygame.mouse.get_pressed()[0]:
                    pos = pygame.mouse.get_pos()
                    row, col = get_clicked_pos(pos, ROWS, width_PA)

                    if row == 0 or row == ROWS - 1 or col == 0 or col == ROWS - 1: continue

                    spot = grid[row][col]
                    
                    if not start and spot != end:
                        start = spot
                        start.make_start()
                    elif not end and spot != start:
                        end = spot
                        end.make_end()
                    elif spot != end and spot != start:
                        spot.make_spot_weighted(CURRENT_WEIGHT)

                elif pygame.mouse.get_pressed()[2]:
                    
                    pos = pygame.mouse.get_pos()
                    row, col = get_clicked_pos(pos, ROWS, width_PA)

                    if row == 0 or row == ROWS - 1 or col == 0 or col == ROWS - 1: continue

                    spot = grid[row][col]
                    spot.make_spot_weighted(1)
                    if spot == start:
                        start = None
                    if spot == end:
                        end = None
                
                
        pygame.display.flip()
        CLOCK.tick(FPS)
    pygame.quit()

main(WIN, PA_WIDTH)