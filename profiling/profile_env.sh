python -m cProfile -o prof.out profile_env.py
pyprof2calltree -i prof.out -k
