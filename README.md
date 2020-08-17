# Randomized-triangle-enumeration-using-MPI-
This program allows to enumerate triangles in large graphs based on randomized algorithms using MPI on python

# Data set to use

The data set to use need to be in the following manner:
    
v1 <- 1 space -> v2

For example:
```
1 2
3 2
5 6
.
.
etc
```
# Execution

Use the following command to execute the randomized triangle enumeration MPI implementation on python:

```
mpiexec -n machine_count -machinefile machinefile.txt python3 mpi_randomized.py path/to/data_set triplet/triplet8.txt directed/undirected
```

machinefile is a file containing the ip addresses of the cluster machines, here an example of machinefile.txt:
```
192.168.1.2
192.168.1.3
192.168.1.4
192.168.1.5
192.168.1.6
192.168.1.7
192.168.1.8
192.168.1.9
```
The machine_count correspond to the number of machine k involved in the machinefile, in the previous example: machine_count=8
