
# mac os

# swig -python trace_skeleton.i
# gcc -O3 -c trace_skeleton.c trace_skeleton_wrap.c -I/usr/local/Cellar/python/3.7.6_1/Frameworks/Python.framework/Versions/3.7/include/python3.7m
# gcc $(python3-config --ldflags) -dynamiclib *.o -o _trace_skeleton.so -I/usr/local/Cellar/python/3.7.6_1/Frameworks/Python.framework/Versions/3.7/lib/libpython3.7m.dylib -undefined dynamic_lookup



# arch
swig -python trace_skeleton.i
# gcc -fpic -c trace_skeleton.c trace_skeleton_wrap.c -I/usr/include/python3.10
gcc -fpic -c trace_skeleton.c trace_skeleton_wrap.c -I/home/brian/.conda/envs/drawings/include/python3.7m
ld -shared trace_skeleton.o trace_skeleton_wrap.o -o _trace_skeleton.so 


# quick tests
# python3 -i -c "import trace_skeleton; trace_skeleton.trace('\0\0\0\1\1\1\0\0\0',3,3); print(trace_skeleton.len_polyline());"
# python3 -i -c "import trace_skeleton; print(trace_skeleton.from_list([0,0,0,1,1,1,0,0,0],3,3))"
# python3 example.py