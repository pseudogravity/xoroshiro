folder to hold files associated with the boinc project of finding a minimum variable set from which all other variables can be derived given a linear system and a table of nonlinear inference rules

C++ usage:
```
./optimizer NameOfConfig NumIters RngSeed [InputFileIfDifferent]
```
e.g
```
./optimizer config-001-from2nextlongs-rand64-int32.npz 10000000 1
```

Python usage:

```
python client.py NameOfConfig NumIters RngSeed [InputFileIfDifferent]
```
e.g.
```
python client.py config-001-from2nextlongs-rand64.npz 1000000 1
```

Saves output in output.txt


