
# Button functionalities:
"""
funkcje pomocnicze, do określenia wagi po naciśnięciu przycisków

"""
def weight_1():
    return ["weight", 1]
def weight_2():
    return ["weight", 2]
def weight_3():
    return ["weight", 3]
def weight_4():
    return ["weight", 4]
def weight_barrier():
    return ["weight", 0]

def print_algo_info(algo_name="", count=0, total_cost=0):
    """
    Wypisanie wszystkich informacji potrzebnych do testów, wypisuje w command line
    """
    # PRINTING OUT SPECIFIC INFORMATIONS ABOUT THE PERFORMANCE OF THE PROGRAM
    print(f"#---# {algo_name} #---#")
    print(f"total count: {count}")
    print(f"total cost: {total_cost}")

def get_clicked_pos(pos, rows, width):
    """
    przerobienie pozycji w x i y (tych z monitora) na rzędy i kolumny
    """
    gap = width // rows
    y, x = pos

    row = y // gap
    col = x // gap
    
    return row, col
