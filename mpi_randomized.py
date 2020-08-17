from mpi4py import MPI
import pandas as pd
import numpy as np
import timeit
import math
import gc
import sys

#-------- MPI -------------
comm = MPI.COMM_WORLD #MPI variable
size = comm.size #MPI size of machine cluster
rank = comm.rank #MPI rank of the machine

#------- Variables ---------
filename = sys.argv[1]
tripletfile = sys.argv[2]
direction = sys.argv[3].lower()
k = size

if rank == 0:
	start_time = timeit.default_timer()
	colnames= ['i','j']
	E_s = pd.read_csv(filename, delimiter=' ', names=colnames, header=None).astype('uint64')
	if direction == "undirected":
		E = E_s
		c = E.columns
		E[[c[0],c[1]]]= E[[c[1],c[0]]]
		E_s = pd.concat([E_s, E], ignore_index=True, sort=False)
	elapsed_time1 = timeit.default_timer() - start_time

#-----------graph partitionning and vertices coloring -------------
	start_time = timeit.default_timer()
	V_s = pd.DataFrame().astype('uint64')
	V_s['i'] = pd.concat([E_s['i'],E_s['j']], ignore_index=True)
	V_s = V_s.drop_duplicates(keep='first')
	V_s['C'] = V_s['i'].apply(lambda x: np.random.randint(1, size**(1./3.) + 1))
	V = [E_s[i::k] for i in range(k)]
	elapsed_time2 = timeit.default_timer() - start_time
else:
	V_s = None
	V = None
	E_s = None

start_time = timeit.default_timer()
E_s = comm.scatter(V, root=0)
V_s = comm.bcast(V_s, root=0)
elapsed_time3 = timeit.default_timer() - start_time

del V
gc.collect()

#----------------- Send edges to proxies --------------------#
start_time = timeit.default_timer()
E_s = E_s.merge(V_s, on=['i'], how='inner').astype('uint64')
E_s_proxy = E_s.merge(V_s, left_on=['j'], right_on=['i'], how='inner')
E_s_proxy = E_s_proxy.rename(index=str, columns={'i_x':'i', 'C_x':'i_color', 'C_y': 'j_color'})
E_s_proxy = E_s_proxy.drop(E_s_proxy.columns[[3]], axis=1).astype('uint64')
elapsed_time4 = timeit.default_timer() - start_time

del V_s
del E_s
gc.collect()

#----------------- Assign colors to machines --------------------#
start_time = timeit.default_timer()
colnames= ['machine','color1','color2','color3']
triplet = pd.read_csv(tripletfile,delimiter=' ', names=colnames, header=None)
E_s_local = E_s_proxy.merge(triplet, left_on=['i_color','j_color'], right_on=['color1','color2'], how='inner')[['machine','i','j','i_color','j_color']].append([E_s_proxy.merge(triplet, left_on=['i_color','j_color'],right_on=['color2','color3'], how='inner')[['machine','i','j','i_color','j_color']],E_s_proxy.merge(triplet, left_on=['i_color','j_color'], right_on=['color3','color1'], how='inner')[['machine','i','j','i_color','j_color']]],sort=False).astype('uint64')
grouped = E_s_local.groupby(['machine'])
E_s_local_list = [group.astype('uint64') for _,group in grouped]

local = []
for r in range(size):
        local.append(comm.scatter(E_s_local_list, root=r))

E_s_local = pd.concat(local).drop_duplicates(keep='first').astype('uint64')
E_s_local = E_s_local.drop(E_s_local.columns[[0]],axis=1)
elapsed_time5 = timeit.default_timer() - start_time

del E_s_proxy
gc.collect()

#----------------- Collect edges from proxies --------------------#
start_time = timeit.default_timer()
local_machine = triplet[triplet.machine == (rank+1)].values.tolist()
local_machine = local_machine[0]
E1 = E_s_local[(E_s_local['i'] < E_s_local['j']) & (E_s_local['i_color'] == local_machine[1]) & (E_s_local["j_color"] == local_machine[2])].astype('uint64')
E2 = E_s_local[(E_s_local['i'] < E_s_local['j']) & (E_s_local['i_color'] == local_machine[2]) & (E_s_local["j_color"] == local_machine[3])].astype('uint64')
E3 = E_s_local[(E_s_local['i'] > E_s_local['j']) & (E_s_local['i_color'] == local_machine[3]) & (E_s_local["j_color"] == local_machine[1])].astype('uint64')

del E_s_local
gc.collect()

#----------------- Local triangle enumeration --------------------#
triangles = E1.merge(E2, left_on=['j'], right_on=['i'], how='inner').astype('uint64')
triangles = triangles.drop(triangles.columns[[4,6]],axis=1)
triangles = triangles.rename(index=str, columns={'i_x':'v1','j_x':'v2','j_y':'v3', 'i_color_x':'v1_color', 'j_color_x':'v2_color', 'j_color_y':'v3_color'})
triangles = triangles.merge(E3, left_on=['v1','v3'], right_on=['j','i'], how='inner').astype('uint64')
triangles = triangles.drop(triangles.columns[[6,7,8,9]],axis=1)

elapsed_time6 = timeit.default_timer() - start_time

if rank == 0:
	time = elapsed_time1+elapsed_time2+elapsed_time3+elapsed_time4+elapsed_time5+elapsed_time6
	print("On rank " + str(rank) + " TC = " + str(len(triangles)) + " time local TC = " + str(elapsed_time6) + " total time = " + str(time))
else:
	time = elapsed_time3+elapsed_time4+elapsed_time5+elapsed_time6
	print("On rank " + str(rank) + " TC = " + str(len(triangles)) + " time local TC = " + str(elapsed_time6) + " total time = " + str(time))
