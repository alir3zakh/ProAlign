#!/usr/bin/env python
import sys
import os
import random
import threading

class parallel_runner(threading.Thread): #this is used to run 2 ncounts in parallel
    def __init__(self,command):
        threading.Thread.__init__(self)
        self.command=command
    def run(self):
        print("Running command in separate thread")
        os.system(self.command)


if len(sys.argv)<5 or len(sys.argv)>10:
    print("Usage: ./GRAALRunner.py alpha net1 net2  result [-s seed]")
    print("The number of nodes in the net1 should be <= number of nodes in net2!")
    print("alpha should be withing 0 and 1")
    print("Due to the inherent randomness in the algorithm it may produce different results")
    print("among different runs. Use -s option to specify the seed for rundom number generator")
    exit(1)

use_seed=False
print_mat=False
print_num=False
adtn_line=" "
seed='0'

L=sys.argv
if  "-s" in L:
    seed=L[L.index("-s")+1]
    adtn_line+=" -s "+seed
if "-c" in L:
    mat_file=L[L.index("-c")+1]
    adtn_line+=" -c "+mat_file
if "-n" in L:
    adtn_line+=" -n"

# rndnum=random.randint(1,100000000)
# dirpath="tmp"+str(rndnum)

dirpath = 'results'
os.system('mkdir -p ' + dirpath)

L1=sys.argv[2].split("/")
L2=sys.argv[3].split("/")

file1 = L1[len(L1)-1]
file2 = L2[len(L2)-1]

if not os.path.exists('./results/{}.res'.format(sys.argv[2])):
    runner1 = parallel_runner("./ncount " + sys.argv[2] + " " + dirpath + "/" + file1 + ".res")
    runner1.start()
    runner1.join()

if not os.path.exists('./results/{}.res'.format(sys.argv[3])):
    runner2 = parallel_runner("./ncount " + sys.argv[3] + " " + dirpath + "/" + file2 + ".res")
    runner2.start()
    runner2.join()



os.system("./GRAAL "+sys.argv[1]+" "+sys.argv[2]+" "+sys.argv[3]+" "+" "+dirpath+"/"+file1+".res.ndump2 "+dirpath+"/"+file2+".res.ndump2 "+sys.argv[4] +adtn_line)
# os.system("rm -f -r "+dirpath)
