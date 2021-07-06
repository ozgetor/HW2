import numpy as np
import random

num_iter = 300
tabu_length = 17
N = 190
neighbors = np.zeros((N, 20 + 2), dtype=int)

flow = [
        [0, 0, 5,0, 5,2,10,3,1, 5, 5, 5, 0, 0, 5, 4, 4, 0, 0, 1 ],
        [0, 0, 3,10,5,1, 5,1,2, 4, 2, 5, 0,10,10, 3, 0, 5,10, 5 ],
        [5, 3, 0,2, 0,5, 2,4,4, 5, 0, 0, 0, 5, 1, 0, 0, 5, 0, 0 ],
        [0,10, 2,0, 1,0, 5,2,1, 0,10, 2, 2, 0, 2, 1, 5, 2, 5, 5 ],
        [5, 5, 0,1, 0,5, 6,5,2, 5, 2, 0, 5, 1, 1, 1, 5, 2, 5, 1 ],
        [2, 1, 5,0, 5,0, 5,2,1, 6, 0, 0,10, 0, 2, 0, 1, 0, 1, 5 ],
        [10,5, 2,5, 6,5, 0,0,0, 0, 5,10, 2, 2, 5, 1, 2, 1, 0,10 ],
        [3, 1, 4,2, 5,2, 0,0,1, 1,10,10, 2, 0,10, 2, 5, 2, 2,10 ],
        [1, 2, 4,1, 2,1, 0,1,0, 2, 0, 3, 5, 5, 0, 5, 0, 0, 0, 2 ],
        [5,4, 5,0, 5,6, 0,1,2, 0, 5, 5, 0, 5, 1, 0, 0, 5, 5, 2 ],
        [5,2, 0,10,2,0, 5,10,0,5, 0, 5, 2, 5, 1,10, 0, 2, 2, 5 ],
        [5,5, 0,2, 0,0,10,10,3,5, 5, 0, 2,10, 5, 0, 1, 1, 2, 5 ],
        [0,0, 0,2, 5,10,2,2, 5,0, 2, 2, 0, 2, 2, 1, 0, 0, 0, 5 ],
        [0,10,5,0, 1,0, 2,0, 5,5, 5,10, 2, 0, 5, 5, 1, 5, 5, 0 ],
        [5,10,1,2, 1,2, 5,10,0,1, 1, 5, 2, 5, 0, 3, 0, 5,10,10 ],
        [4, 3,0,1, 1,0, 1,2, 5,0,10, 0, 1, 5, 3, 0, 0, 0, 2, 0 ],
        [4, 0,0,5, 5,1, 2,5, 0,0, 0, 1, 0, 1, 0, 0, 0, 5, 2, 0 ],
        [0, 5,5,2, 2,0, 1,2, 0,5, 2, 1, 0, 5, 5, 0, 5, 0, 1, 1 ],
        [0,10,0,5, 5,1, 0,2, 0,5, 2, 2, 0, 5,10, 2, 2, 1, 0, 6 ],
        [1, 5,0,5, 1,5,10,10,2,2, 5, 5, 5, 0,10, 0, 0, 1, 6, 0 ]]

distance = [
        [0,1,2,3,4,1,2,3,4, 5, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7 ],
        [1,0,1,2,3,2,1,2,3, 4, 3, 2, 3, 4, 5, 4, 3, 4, 5, 6 ],
        [2,1,0,1,2,3,2,1,2, 3, 4, 3, 2, 3, 4, 5, 4, 3, 4, 5 ],
        [3,2,1,0,1,4,3,2,1, 2, 5, 4, 3, 2, 3, 6, 5, 4, 3, 4 ],
        [4,3,2,1,0,5,4,3,2, 1, 6, 5, 4, 3, 2, 7, 6, 5, 4, 3 ],
        [1,2,3,4,5,0,1,2,3, 4, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6 ],
        [2,1,2,3,4,1,0,1,2, 3, 2, 1, 2, 3, 4, 3, 2, 3, 4, 5 ],
        [3,2,1,2,3,2,1,0,1, 2, 3, 2, 1, 2, 3, 4, 3, 2, 3, 4 ],
        [4,3,2,1,2,3,2,1,0, 1, 4, 3, 2, 1, 2, 5, 4, 3, 2, 3 ],
        [5,4,3,2,1,4,3,2,1,0, 5, 4, 3, 2, 1, 6, 5, 4, 3, 2 ],
        [2,3,4,5,6,1,2,3,4,5, 0, 1, 2, 3, 4, 1, 2, 3, 4, 5 ],
        [3,2,3,4,5,2,1,2,3,4, 1, 0, 1, 2, 3, 2, 1, 2, 3, 4 ],
        [4,3,2,3,4,3,2,1,2,3, 2, 1, 0, 1, 2, 3, 2, 1, 2, 3 ],
        [5,4,3,2,3,4,3,2,1,2, 3, 2, 1, 0, 1, 4, 3, 2, 1, 2 ],
        [6,5,4,3,2,5,4,3,2,1, 4, 3, 2, 1, 0, 5, 4, 3, 2, 1 ],
        [3,4,5,6,7,2,3,4,5,6, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4 ],
        [4,3,4,5,6,3,2,3,4,5, 2, 1, 2, 3, 4, 1, 0, 1, 2, 3 ],
        [5,4,3,4,5,4,3,2,3,4, 3, 2, 1, 2, 3, 2, 1, 0, 1, 2 ],
        [6,5,4,3,4,5,4,3,2,3, 4, 3, 2, 1, 2, 3, 2, 1, 0, 1 ],
        [7,6,5,4,3,6,5,4,3,2, 5, 4, 3, 2, 1, 4, 3, 2, 1, 0 ]]


def printer(tag, sol):
    print("%s: %s cost %s " % (tag, [item + 1 for item in sol], solution_cost(sol)))
    

def solution_cost(sol):
  cost = 0
  for i in range(len(sol)):
    for j in range(len(sol)):
        cost += distance[i][j] * flow[sol[i]][sol[j]]
  return cost


def swap_move(sol_n, idx, neighbors):
    for i in range (len(sol_n)):
        j = i + 1
        for j in range(len(sol_n)):
            if i < j:
                idx = idx + 1
                sol_n[j], sol_n[i] = sol_n[i], sol_n[j]
                neighbors[idx, : -2] = sol_n
                neighbors[idx, -2:] = [sol_n[i], sol_n[j]]
                sol_n[i], sol_n[j] = sol_n[j], sol_n[i]

def not_in_tabu(solution, tabu):
    not_found = False
    if not solution.tolist() in tabu:
        solution[0], solution[1] = solution[1], solution[0]
        if not solution.tolist() in tabu:
            not_found = True

    return not_found

def run_tabu(initial_solution, num_iter, neighbors, tabu_length, aspiration=False):
    curnt_sol = initial_solution
    best_soln = initial_solution
    TABU = []
    frequency = {}

    printer("Initial", initial_solution)
    
    while num_iter > 0:

        idx = -1
        swap_move(curnt_sol, idx, neighbors)  # komşuluk araması

        cost = np.zeros((len(neighbors)))  # komşuların cost değeri 
        for index in range(len(neighbors)):
            cost[index] = solution_cost(neighbors[index, :-2])  #  the candidate neighborsların cost hesaplaması
        rank = np.argsort(cost)  # costa göre sıralama
        neighbors = neighbors[rank]

        for j in range(N):

            not_tabu = not_in_tabu(neighbors[j, -2:], TABU)
            if (not_tabu):
                curnt_sol = neighbors[j, :-2].tolist()
                TABU.append(neighbors[j, -2:].tolist())

                if len(TABU) > tabu_length:
                    TABU = TABU[1:]

                #frequency based
                if not tuple(curnt_sol) in frequency.keys():
                    frequency[tuple(curnt_sol)] = 1 # set key penalty değerini 1 olarak ayarla

                    if solution_cost(curnt_sol) <  solution_cost(best_soln):
                        best_soln = curnt_sol

                else:

                    cur_cost= solution_cost(curnt_sol) + frequency[tuple(curnt_sol)] # frekansa göre penalize etme
                    frequency[tuple(curnt_sol)] += 1   # mevcut ziyaretin sıklığını artır

                    if cur_cost <  solution_cost(best_soln):
                        best_soln = curnt_sol

                break

            #Aspiration

            elif aspiration and solution_cost(neighbors[j, :-2]) <  solution_cost(best_soln):

                curnt_sol = neighbors[j, :-2].tolist()

                TABU.insert(0, TABU.pop(TABU.index(neighbors[j, -2:].tolist())))

                #Tabu.append(neighbors[j, -2:].tolist())

                if len(TABU) > tabu_length:
                    TABU = TABU[1:]

                    # frequency based
                if not tuple(curnt_sol) in frequency.keys():
                    frequency[tuple(curnt_sol)] = 1  # set key penalty değerini 1 olarak ayarla
                    best_soln = curnt_sol
                    # print("Best sol %s cost: %s @ iter %s" % (best_soln, solution_cost(best_soln), num_iter))
                else:

                    cur_cost= solution_cost(curnt_sol) + frequency[tuple(curnt_sol)] # frekansa göre penalize etme
                    frequency[tuple(curnt_sol)] += 1   # mevcut ziyaretin sıklığını artır

                    if cur_cost <  solution_cost(best_soln):
                        best_soln = curnt_sol

        num_iter -= 1
    printer("Best Solution", best_soln)


if __name__== "__main__":  # programın çalışmaya başladığı ana işlevi çağırma
    print("Part 1:")
    random.seed(0)
    initial_solution = random.sample(range(20), 20)
    printer("Initial", initial_solution)
    print()

    print("Part 4.a:")
    print("1.")
    random.seed(1)
    initial_solution = random.sample(range(20), 20)
    run_tabu(initial_solution, num_iter, neighbors, tabu_length, aspiration = False)
    
    print("2.")
    random.seed(2)
    initial_solution = random.sample(range(20), 20)
    run_tabu(initial_solution, num_iter, neighbors, tabu_length, aspiration = False)

    print("3.")
    random.seed(3)
    initial_solution = random.sample(range(20), 20)
    run_tabu(initial_solution, num_iter, neighbors, tabu_length, aspiration = False)


    print()

    print("Part 4.b:")
    print("Tabu Length: ", tabu_length - 2)
    random.seed(2)
    initial_solution = random.sample(range(20), 20)
    run_tabu(initial_solution, num_iter, neighbors, tabu_length - 2, aspiration = False)

    print("Tabu Length: ", tabu_length + 2)
    random.seed(2)
    initial_solution = random.sample(range(20), 20)
    run_tabu(initial_solution, num_iter, neighbors, tabu_length + 2, aspiration = False)


    print()

    print("Part 4.c:")
    print("With Aspiration")
    random.seed(2)
    initial_solution = random.sample(range(20), 20)
    run_tabu(initial_solution, num_iter, neighbors, tabu_length, aspiration = True)


