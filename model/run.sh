PROCESS_NAME=sc2_backend

# export SC2PF=WineLinux
# export WINE=/usr/bin/wine
export SC2PATH="/home/ivatolm/Projects/betastar/data/game"

pkill -f $PROCESS_NAME
pkill -f Main_Thread
pkill -f SC2_x64.exe
bash -c "exec -a $PROCESS_NAME python model/env/env_back.py &"
