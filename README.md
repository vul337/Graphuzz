# Graphuzz
Graphuzz has four components：
- FUZZER: for FUZZING
- MODEL: for MODEL training
- PREPROCESS: for INSTRUMENTATION and seed ESTIMATION

## FUZZING WORKFLOWS
BUILD PREPROCESS
Make sure that clang (version 11) is installed.
```bash
cd PATH_TO_Graphuzz/PREPROCESS
make
cd PATH_TO_Graphuzz/FUZZER
make
```

Compile target, using the normal OSS-FUZZ target as the example.
```bash
export CFLAGS=""
export CXXFLAGS=""
export CC=PATH_TO_Graphuzz/PREPROCESS/afl-clang-fast
export CXX=PATH_TO_Graphuzz/PREPROCESS/afl-clang-fast++
export OUT=`pwd`/out
export LIB_FUZZING_ENGINE=PATH_TO_Graphuzz/FUZZER/libAFLDriver.a
export AFL_MAP_SIZE=65536
export FUZZER_LIB=PATH_TO_Graphuzz/FUZZER/libAFLDriver.a
export AFL_LLVM_INSTRUMENT=CLASSIC
export CDF_FILE_DIR=`pwd`/CDF
export SRC=`pwd`
mkdir CDF
mkdir out
```
Then, `cd` to the source code directory of the target and follow the `build.sh` from OSS-FUZZ.
After compilation, we will get the graph information of the target in the directory set by `CDF_FILE_PATH`.
We need to aggreagate the `e-CFG` of the target by
```bash
cd $CDF_FILE_PATH
python PATH_TO_Graphuzz/FUZZER/nn_server/exportTrainingSet.py $CDF_FILE_PATH PATH_TO_Graphuzz/FUZZER/nn_server/  
```
Now we will get the `CDFG.pkl` in the directory `PATH_TO_Graphuzz/FUZZER/nn_server/`

Now we can start the fuzzing process.
Start a terminal and run
```bash
cd PATH_TO_Graphuzz/FUZZER/nn_server && python nn_server.py
```
And, start a new terminal and run
```bash
#chmod +x $CurDir/nn_server.sh
#nohup $CurDir/nn_server.sh &
export AFL_NO_AFFINITY=1
export AFL_NO_UI=1
/PATH_TO_Graphuzz/FUZZER/afl-fuzz -i path_to_seeds -o path_to_output -K /tmp/nn_server.sock -- target
```

## TRAINING SET GENERATION
1. Prepare enough seeds (more than 2048 seeds)
2. Run each seed for 30 minutes (as described in our paper), and record the initial bitmap and last bitmap. We provide a program to do this work (`/PATH_TO_Graphuzz/PREPROCESS/afl-evaluator`). Use this program like `./afl-evaluator -I ~/TEST/pcapplusplus/seeds/default/queue -O ~/TEST/pcapplusplus/output_30 -B  ~/TEST/pcapplusplus/FuzzTarget`.
3. Generate TRAINING SET. For example, run `python3 /PATH_TO_Graphuzz/PREPROCESS/Extractor/exportTrainingSet.py THE_DIR_OF_CDFG.PKL THE_DIR_OF_CDFG.PKL ~/TEST/pcapplusplus/output_30 ~/TRAININGSET/trainingset_pcapplusplus`.
4. Normalize the labels. `python preprocess.py ~/TRAININGSET/trainingset_pcapplusplus ~/TRAININGSET/trainingset_pcapplusplus_norm`
5. Sampling multiple software generated training sets and composing a new hybrid training set. For example, `for x in {A,B,C,D}; do ls ~/TRAININGSET/trainingset_${x}_norm | sort -R | head -30000 | xargs -I {} cp ~/TRAININGSET/trainingset_$x_norm/{} ~/trainingset_cross/ ; done`


