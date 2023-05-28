from cmath import pi
from math import cos
import random
import copy
import matplotlib.pyplot as plt 
import numpy as np 
P=100 # 100 pop for the 2nd function, 50 for the first 
N=20 # amount of bits in each indivdual 

Min =-10
Max = 5
gen=1 # the number of the fgenration this is the first genration 
mean_of_gens =[]  # an array to store the mean of each geration to plot it 
smallest_of_gens =[] # an array to store the smallest fit of each geration to plot it 
x_axis=[] # x axis values, which is 50
smallest_fitness = 1000 # intialising the smallest fitness before starting the loop for the genration 
population=[] # creating a population array to store the individuals 
offspring=[] # creating another array to store the offsprings
pop_fit =[]
 

class individual: # this class is just intialising the individuals 
    def __init__(self): 
        self.gene=[0]*N # intialisies all the bits of the genes to 0s
        self.fitness=0 # intialies the fitness for each gene to 0


for i in range(0,P): # that function randmoizes the bits ofevery gene , and for x in range 0,p means from 0 the first indivdual to p the last one 
    tempgene=[] # a temprary array to store the randmised bits
    for y in range(0,N): # for range 0,N means from the first bit in each individual to the last bit which is 12
        tempgene.append(random.uniform(Min,Max))#add random bits to a temporary individual 
    newind=individual()
    newind.gene=tempgene.copy() #after creating an indivisual object then copy the cntents of the temporary gene to it 
    population.append(newind) # add the individual to the populaton 


def test_function(ind): # counts the fitness of the function (by counting the ones) ind stands for indvidual
#Assignment fitness function no. 2
 utility = 0
 for j in range(0,N):
     utility = (ind.gene[j]**2)

 utility2 =0
 for j in range(0,N):
      utility2 += (0.5*ind.gene[j])**2

 utility3 =0
 for j in range(0,N):
     utility3 += (0.5*ind.gene[j])**4

 tot_utility = utility + utility2 +utility3
 return tot_utility


"""    
#Assignemt fitness function no. 1
 utility= 0
 for j in range(0,N-1):
    utility = 100*(((ind.gene[j+1])-(ind.gene[j]**2))**2) + ((1-ind.gene[j])**2)
 return utility
"""

'''
# WS 3 fitness function
 utility=0
 for w in range(0,N):
     utility += ((ind.gene[w]**2) - (10*(cos((2*pi)*ind.gene[w]))))
 utility+= (10*N)
 return utility
 '''
'''
WS1 function
  utility=0
    for w in range(N):
        utility=utility+ind.gene[w] # for each bit inside each indvidual you add it to the utiltiy function 
    return utility
    when you call the test function it will do all of thatagain and give you the answer directly 
'''   
 

for k in range (P): # 
    offspring=[]
    total_fitness=0 
    x_axis.append(k) 
    for y in range(len(population)): 
        population[y].fitness=test_function(population[y]) # each indivdual has intialised fitness from the beginign the self.fitness=0 and now we are going to store the value for each indivdual using the test function 
       # pop_fit.append(population[y].fitness) # uncomment with ranking selection
        if population[y].fitness < smallest_fitness:
            smallest_fitness=population[y].fitness # trying to get the smallest fitness
        total_fitness+=population[y].fitness # calculating the total fitness
        mean_fitness = total_fitness/P
    print(f"Mean of generation {gen} = {mean_fitness}")  #printing the mean fitness         
    print(f"smallest fitness of the genration {gen} = {smallest_fitness}\n") #printing the smallest fitness

    mean_of_gens.append(mean_fitness) # corresponding y axis values for the mean graph
    smallest_of_gens.append(smallest_fitness) # corresponding y axis values for the best graph   

    """
    # Roulette wheel selection 
    def roulette_wheel(population):
     individualas_porob = []
     for x in range(0,P):
        population[x].fitness=test_function(population[x])
        individualas_porob.append(population[x].fitness/ total_fitness)
        #individualas_porob = 1 - np.array(individualas_porob)
     return np.random.choice(population, p=individualas_porob)
   
    tmepoffspring = 0
    for b in range(len(population)):
        tmepoffspring = roulette_wheel(population)
        offspring.append(tmepoffspring)
    
    """
    
   #Tournment selection
    for x in range(0,P): 
        parent1 = random.randint(0,P-1) # choosing any random indivdual in our genration and passing it to the offspring 
        off1 = copy.deepcopy(population[parent1])
        parent2 = random.randint(0,P-1) # choosing any second random indivdual in our genration and passing it to the offspring 
        off2 = copy.deepcopy(population[parent2]) 
        if off1.fitness > off2.fitness: # trying to choose the best individuals to be our new genration, an individual can be choosen twice 
            offspring.append(off2)  # choosing the worst offspring
        else:
            offspring.append(off1)
   

    """
    #ranking selection 
    pop_fit.sort()
    for x in range(50,P):
        del pop_fit[50]

    offspring = pop_fit.copy()
   """

    
    offspring_fitness=0
    for v in range(len(offspring)): 
        offspring[v].fitness=test_function(offspring[v])
        if offspring[v].fitness < smallest_fitness:
            smallest_fitness=offspring[v].fitness # getting the smallest fitness of the offspring 
        offspring_fitness+=offspring[v].fitness # calculating the total of the offsprings 
    

    toff1 = individual() 
    toff2 = individual()
    temp = individual()
    for m in range(0,P,2):
        toff1 = copy.deepcopy(offspring[m]) # coping the first offspring    
        toff2 = copy.deepcopy(offspring[m+1]) # coping the second offspring 
        temp = copy.deepcopy(offspring[m])
        crosspoint = random.randint(1,N) #choose random crossover point between 1-N(50)
        #crosspoint2 = random.randint(1,N) # multipoint crossover 
        #if crosspoint> crosspoint2:
        for l in range (crosspoint, N): #dont forget to switch N to crosspoint 2 if using multipoint 
                    toff1.gene[l] = toff2.gene[l] #making the genes that is stored in toff1 between crosspoint and N = the genes stored in toff 2 
                    toff2.gene[l] = temp.gene[l] #making the genes that is stored in toff2 between crosspoint and N = the genes stored in temp which is the same as toff1
    
       # else:
          #   for l in range (crosspoint2, crosspoint):
         #        toff1.gene[l] = toff2.gene[l] #making the genes that is stored in toff1 between crosspoint and N = the genes stored in toff 2 
          #       toff2.gene[l] = temp.gene[l] #making the genes that is stored in toff2 between crosspoint and N = the genes stored in temp which is the same as toff1
     
    offspring[m] = copy.deepcopy(toff1) # stroing them in array again with the new values 
    offspring[m+1] = copy.deepcopy(toff2)

    MUTRATE=0.3 # 0.3 for the second function, 1st func 0.1
    MUTSTEP = 0.8 # 0.8 for the second function, 1st function 0.2
    temp_population = [] # defining and temprary population array 
    for q in range(0,P):
        newind = individual() # newind equals the indivdual a
        newind.gene = []
        for w in range(0,N):
            gene = offspring[q].gene[w]
            mutprob = random.random() # random.random is random number between 0 and 1 
            if mutprob < MUTRATE: # ask about that line 
               alter = random.uniform(-MUTSTEP, MUTSTEP) # adding or removing from the real value the mutstep 
               gene = gene + alter
               if gene> Max:
                gene = Max # making sure that all genes are within the max
                if gene< Min:
                    gene = Min # making sure that all genes are within the min

            newind.gene.append(gene) # updating the gene in the array 
        temp_population.append(newind) # updating the new indivdiual to the temporary population
        # print(newind.gene)   
    population=[]       
    population=copy.deepcopy(temp_population) # copying the new population after the mutation
    gen+=1 # updating the new genration number 


plt.plot(x_axis,mean_of_gens, label = "Mean fitness") # plotting the mean graph and labeling it  
plt.plot(x_axis,smallest_of_gens,label ="smallest fitness")# plotting the smallest graph and labeling it

plt.xlabel("x - axis - population") # naming the x axis
plt.ylabel('y - axis') # naming the y axis

plt.title('Evolutionary Algorthim') # giving a title to my graph
plt.legend() # show a legend on the plot
plt.show() # function to show the plot
