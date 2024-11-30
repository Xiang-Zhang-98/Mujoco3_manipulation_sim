COMMON=-O2 -I${MUJOCO_PATH}include -L${MUJOCO_PATH}bin -std=c++11 -mavx -pthread -Wl,-rpath,'$$ORIGIN'

all:
	# g++ $(COMMON) main.cpp -lmujoco210 -lGL -lglew ${MUJOCO_PATH}bin/libglfw.so.3 -o bin/main
	g++ $(COMMON) -Wall -O3 -fPIC -shared py_module.cpp -lmujoco200 -lGL -lglew ${MUJOCO_PATH}bin/libglfw.so.3 -o bin/mujocosim.so
	g++ $(COMMON) py_module.cpp -lmujoco200 -lGL -lglew ${MUJOCO_PATH}bin/libglfw.so.3 -o bin/mujocosim
	g++ $(COMMON) -Wall -O3 -fPIC -shared py_module_pivoting.cpp -lmujoco200 -lGL -lglew ${MUJOCO_PATH}bin/libglfw.so.3 -o bin/mujocosim_pivoting.so
	g++ $(COMMON) py_module_pivoting.cpp -lmujoco200 -lGL -lglew ${MUJOCO_PATH}bin/libglfw.so.3 -o bin/mujocosim_pivoting
	g++ $(COMMON) -Wall -O3 -fPIC -shared py_module_door.cpp -lmujoco200 -lGL -lglew ${MUJOCO_PATH}bin/libglfw.so.3 -o bin/mujocosim_door.so
	g++ $(COMMON) py_module_door.cpp -lmujoco200 -lGL -lglew ${MUJOCO_PATH}bin/libglfw.so.3 -o bin/mujocosim_door
	# gcc -c -O2 -mavx -I${MUJOCO_PATH}include ${MUJOCO_PATH}include/uitools.c
	# g++ $(COMMON) uitools.o simulate.cpp -lmujoco210 -lGL -lglew ${MUJOCO_PATH}bin/libglfw.so.3 -o bin/simulate
	# rm *.o
