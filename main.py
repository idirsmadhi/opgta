from common.transformations.camera import transform_img, eon_intrinsics
from common.transformations.model import medmodel_intrinsics
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

import cv2 
from tensorflow.keras.models import load_model
from common.tools.lib.parser import parser
import sys




#--------------------------------import Vpilot and GUI

from deepgtav.messages import Start, Stop, Scenario, Commands, frame2numpy, Dataset
from deepgtav.client import Client

print("starting DEEPGTA")

client = Client(ip="localhost", port=8000)
dataset = Dataset(rate=20, frame=[1164,874])

# print("importing vpilotgui")

# import vpilotgui
# from vpilotgui import launch
# from vpilotgui import startCommand

#-------------------------------fin

#-------------------------to detect ctrl+c and close the client
#!/usr/bin/env python
import signal
import sys

def signal_handler(sig, frame):
  client.close()
  print("client closed")
  sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
#-------------------------------------------------

print("before laoding model")
#camerafile = "sample.hevc"
supercombo = load_model('models\supercombo.keras')
print("model loaded")

MAX_DISTANCE = 140.
LANE_OFFSET = 1.8
MAX_REL_V = 10.

LEAD_X_SCALE = 10
LEAD_Y_SCALE = 10

#cap = cv2.VideoCapture(camerafile)

NBFRAME = 1000

def frame_to_tensorframe(frame):                                                                                               
  H = (frame.shape[0]*2)//3                                                                                                
  W = frame.shape[1]                                                                                                       
  in_img1 = np.zeros((6, H//2, W//2), dtype=np.uint8)                                                      
                                                                                                                            
  in_img1[0] = frame[0:H:2, 0::2]                                                                                    
  in_img1[1] = frame[1:H:2, 0::2]                                                                                    
  in_img1[2] = frame[0:H:2, 1::2]                                                                                    
  in_img1[3] = frame[1:H:2, 1::2]                                                                                    
  in_img1[4] = frame[H:H+H//4].reshape((-1, H//2,W//2))                                                              
  in_img1[5] = frame[H+H//4:H+H//2].reshape((-1, H//2,W//2))
  return in_img1

def vidframe2img_yuv_reshaped():
  print("wainting for frame from deepgta")
  message = client.recvMessage()  
  print("received frame from deepgta")
                
  # The frame is a numpy array that can we pass through a CNN for example     
  frame = frame2numpy(message['frame'], (1164,874))
  #ret, frame = cap.read()
  img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
  return frame, img_yuv.reshape((874*3//2, 1164))

def vidframe2frame_tensors():
  frame, img = vidframe2img_yuv_reshaped()
  print("received frame from deepgta")
  imgs_med_model = transform_img(img, from_intr=eon_intrinsics, to_intr=medmodel_intrinsics, yuv=True,
                                    output_size=(512,256))
  f2t = frame_to_tensorframe(np.array(imgs_med_model)).astype(np.float32)/128.0 - 1.0
  return frame, f2t




#----------------------- Vpilolt

# scenario = Scenario(drivingMode=[786603,100.0], weather='EXTRASUNNY',vehicle='blista',time=[12,0],location=[-2573.13916015625, 3292.256103515625, 13.241103172302246]) #manual driving drivingMode=-1
# dataset = Dataset(rate=20, frame=[1164,874])
# client.sendMessage(Start(scenario=scenario, dataset = dataset)) 

#------------------------ fin


# frame_tensors = np.zeros((NBFRAME,6,128,256))
# for i in tqdm(range(NBFRAME)):
#     frame_tensors[i] = vidframe2frame_tensors()[1]

# cap2 = cv2.VideoCapture("sample.hevc")
def startPredict():
  state = np.zeros((1,512))
  desire = np.zeros((1,8))
  for i in tqdm(range(NBFRAME-1)):
    print("-------- ",i)
    if i == 0:
      frame, frame_tensors1 = vidframe2frame_tensors()
    else :
      frame, frame_tensors2 = vidframe2frame_tensors()
      inputs = [np.vstack([frame_tensors1,frame_tensors2])[None], desire, state]
      # inputs = [np.vstack(frame_tensors[i:i+2])[None], desire, state]
      print("before predictio")
      outs = supercombo.predict(inputs)
      print("after predictio")
      #print(outs)
      parsed = parser(outs)
      # Get x and y coordinates of each line
      lll_x, lll_y = parsed["lll"][0], range(0,192)
      rll_x, rll_y = parsed["rll"][0], range(0,192)
      path_x, path_y = parsed["path"][0], range(0,192)

    # Calculate angle of rotation for each line
    #   lll_angle = np.arctan2(np.diff(lll_y), np.diff(lll_x))[0]
    #   rll_angle = np.arctan2(np.diff(rll_y), np.diff(rll_x))[0]
      #path_angle = np.arctan2(np.diff(path_y), np.diff(path_x))[0]

      rll_angle = np.arctan2(rll_y[25]-rll_y[0], rll_x[0]-rll_x[25])
      lll_angle = np.arctan2(lll_y[25]-lll_y[0], lll_x[0]-lll_x[25])
      path_angle = np.arctan2(path_y[25]-path_y[0], path_x[0]-path_x[25])

      

    # Convert angle to degrees
      lll_angle_deg = np.rad2deg(lll_angle)
      rll_angle_deg = np.rad2deg(rll_angle)
      path_angle_deg = np.rad2deg(path_angle)

      diff_x = path_x[0]-path_x[25]
      norm_diff_x = 2.5*diff_x/(max(path_x)-min(path_x))
      client.sendMessage(Commands(0.3,0, float(norm_diff_x )    ))

    # Print angles
      print("Left lane line angle: ", lll_angle_deg)
      print("Right lane line angle: ", rll_angle_deg)
      print("Path line angle: ", path_angle_deg)

    # Plot lines
      plt.clf()
      plt.subplot(1, 2, 1)
      plt.title("Video")
      plt.imshow(frame,aspect="auto")
      plt.subplot(1, 2, 2)
      plt.title("Prediction")
      plt.plot(lll_x, lll_y, "b-", linewidth=1)
      plt.plot(rll_x, rll_y, "r-", linewidth=1)
      plt.plot(path_x, path_y, "g-", linewidth=1)
      plt.gca().invert_xaxis()
      plt.pause(0.001)
plt.show()







      

import tkinter as tk
root = tk.Tk()

#title:
root.title("GTAV OPENPILOT¡!")

##############################################################
#Window Size and Screen Position
window_width = 500
window_height = 800

# get the screen dimension
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# find the center point
center_x = int(screen_width/2 - window_width / 2)
center_y = int(screen_height/2 - window_height / 2)

# set the position of the window to the center of the screen
root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
##############################################################

##############################################################
#Configuring index and weight of rows and colums
root.rowconfigure(0, weight=1)
root.rowconfigure(1, weight=1)
root.rowconfigure(2, weight=1)
root.rowconfigure(3, weight=1)
root.rowconfigure(4, weight=1)
root.rowconfigure(5, weight=1)
root.rowconfigure(6, weight=1)
root.rowconfigure(7, weight=1)
root.rowconfigure(8, weight=1)
root.rowconfigure(9, weight=1)
root.rowconfigure(10, weight=1)
root.rowconfigure(11, weight=1)
root.rowconfigure(12, weight=1)
root.rowconfigure(13, weight=1)
root.rowconfigure(14, weight=1)
root.rowconfigure(15, weight=1)
root.rowconfigure(16, weight=1)
root.rowconfigure(17, weight=1)
root.rowconfigure(18, weight=1)
root.rowconfigure(19, weight=1)
root.rowconfigure(20, weight=1)
root.rowconfigure(21, weight=1)
root.rowconfigure(22, weight=1)
root.rowconfigure(23, weight=1)
root.rowconfigure(24, weight=1)
root.rowconfigure(25, weight=1)



root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=1)
root.columnconfigure(2, weight=0)
root.columnconfigure(3, weight=0)
root.columnconfigure(4, weight=0)
root.columnconfigure(5, weight=0)
root.columnconfigure(6, weight=0)
root.columnconfigure(7, weight=0)
root.columnconfigure(8, weight=0)
root.columnconfigure(9, weight=1)

##############################################################
#Lables
scene = tk.Label(root, text="Scenario:",font=("Courier", 30))
scene.grid(column=5, row=0)

datas = tk.Label(root, text="Dataset:", font=("Courier", 30))
datas.grid(column=5, row=6)

###############################################################
#Scenarios#####################################################
###############################################################

###############################################################
#Location(floats)

locLabel = tk.Label(root, text="Location:", font=("Courier"))
locLabel.grid(column=4, row=1)

locdefault1 = tk.StringVar()
locdefault2 = tk.StringVar()
locdefault3 = tk.StringVar()

#location defaults =[-2573.13916015625, 3292.256103515625, 13.241103172302246]
locdefault1.set("-2573.13916015625")
locdefault2.set("3292.256103515625")
locdefault3.set("13.241103172302246")


locEntry1 = tk.Entry(root, width=5,font=("Courier"), textvariable=locdefault1)#width of the text entry box
locEntry2 = tk.Entry(root, width=5,font=("Courier"), textvariable=locdefault2)
locEntry3 = tk.Entry(root, width=5,font=("Courier"), textvariable=locdefault3)

locEntry1.grid(column=5, row=1)
locEntry2.grid(column=6, row=1)
locEntry3.grid(column=7, row=1)


##############################################################
#Time(ints)
timeLabel1 = tk.Label(root, text="Time:", font=("Courier"))
timeLabel1.grid(column=4, row=2)

timeLabel2 = tk.Label(root, text=":", font=("Courier"))
timeLabel2.grid(column=6, row=2)

timedef1 = tk.StringVar()
timedef2 = tk.StringVar()

timedef1.set("12")
timedef2.set("00")


timeEntry1 = tk.Entry(root, textvariable=timedef1, width=2,font=("Courier"))#width of the text entry box
timeEntry2 = tk.Entry(root, textvariable=timedef2, width=2,font=("Courier"))

timeEntry1.grid(column=5, row=2)
timeEntry2.grid(column=7, row=2)



##############################################################
#Weather Label & Dropdown Menu#

weatherlabel = tk.Label(root, text="Weather:", font=("Courier"))
weatherlabel.grid(column=4, row=3)

weatherchoice = tk.StringVar(root)
weatherchoice.set("SUNNY") #makes the default choice in the dropdown menu

wDropdown = tk.OptionMenu(root, weatherchoice, "SUNNY", "RAINY", "HAZY", "EXTRASUNNY")
wDropdown.grid(column=5, row=3)

##############################################################
#Vehicle(str)
vlabel = tk.Label(root, text="Vehicle:", font=("Courier"))
vlabel.grid(column=4, row=4)

defaultvehicle = tk.StringVar(root)
defaultvehicle.set("blista")
ventry = tk.Entry(root, textvariable=defaultvehicle, width=6,font=("Courier"))
ventry.grid(column=5, row=4)

###############################################################
#Driving Mode(ints)
dmlabel = tk.Label(root, text="Driving Mode:", font=("Courier"))
dmlabel.grid(column=4, row=5)

dmneg1 = tk.StringVar(root)
dmneg1.set(-1)

dmentry1 = tk.Entry(root, textvariable=dmneg1, width=2,font=("Courier"))
dmentry2 = tk.Entry(root, width=4,font=("Courier"))

dmentry1.grid(column=5, row=5)
dmentry2.grid(column=6, row=5)

###############################################################
#DATASETS######################################################
###############################################################

###############################################################
#Rate(int()
rlabel = tk.Label(root, text="Rate:", font=("Courier"))
rlabel.grid(column=4, row=7)

ratedefault = tk.StringVar()
ratedefault.set("20")
#default value = 20
rentry = tk.Entry(root, width=6,font=("Courier"), textvariable=ratedefault)
rentry.grid(column=5, row=7)

###############################################################
#Frame(ints)
framelabel = tk.Label(root, text="Frame:", font=("Courier"))
framelabel.grid(column=4, row=8)

framedefault1 = tk.StringVar()
framedefault1.set("1164")

framedefault2 = tk.StringVar()
framedefault2.set("874")


frameentry1 = tk.Entry(root, width=6,font=("Courier"), textvariable=framedefault1)
frameentry2 = tk.Entry(root, width=6,font=("Courier"), textvariable=framedefault2)

frameentry1.grid(column=5, row=8)
frameentry2.grid(column=6, row=8)

###############################################################
#Vehicles(bool)

vehiclelabel = tk.Label(root, text="Vehicles:", font=("Courier"))
vehiclelabel.grid(column=4, row=9)

vehicleschoice = tk.StringVar(root)
vehicleschoice.set("None") #makes the default choice in the dropdown menu

vehiclesBool = tk.OptionMenu(root, vehicleschoice, "None", "True", "False")
vehiclesBool.grid(column=5, row=9)

###############################################################
#Peds(bool)

pedslabel = tk.Label(root, text="Peds:", font=("Courier"))
pedslabel.grid(column=4, row=10)

pedschoice = tk.StringVar(root)
pedschoice.set("None") #makes the default choice in the dropdown menu

pedsBool = tk.OptionMenu(root, pedschoice, "None", "True", "False")
pedsBool.grid(column=5, row=10)

###############################################################
#TrafficSigns(bool)

trafficslabel = tk.Label(root, text="Traffic Signs:", font=("Courier"))
trafficslabel.grid(column=4, row=11)

trafficschoice = tk.StringVar(root)
trafficschoice.set("None") #makes the default choice in the dropdown menu

trafficsBool = tk.OptionMenu(root, trafficschoice, "None", "True", "False")
trafficsBool.grid(column=5, row=11)

###############################################################
#Direction(floats)
directionlabel = tk.Label(root, text="Direction:", font=("Courier"))
directionlabel.grid(column=4, row=12)

directionentry1 = tk.Entry(root, width=6,font=("Courier"))
directionentry2 = tk.Entry(root, width=6,font=("Courier"))
directionentry3 = tk.Entry(root, width=6,font=("Courier"))

directionentry1.grid(column=5, row=12)
directionentry2.grid(column=6, row=12)
directionentry3.grid(column=7, row=12)

###############################################################
#Reward(ints: [id, p1, p2])
rewardlabel = tk.Label(root, text="Reward:", font=("Courier"))
rewardlabel.grid(column=4, row=13)

rewardentry1 = tk.Entry(root, width=6,font=("Courier"))
rewardentry2 = tk.Entry(root, width=6,font=("Courier"))
rewardentry3 = tk.Entry(root, width=6,font=("Courier"))

rewardentry1.grid(column=5, row=13)
rewardentry2.grid(column=6, row=13)
rewardentry3.grid(column=7, row=13)

###############################################################
#Throttle(bool)

throttlelabel = tk.Label(root, text="Throttle:", font=("Courier"))
throttlelabel.grid(column=4, row=14)

throttlechoice = tk.StringVar(root)
throttlechoice.set("None") #makes the default choice in the dropdown menu

throttleBool = tk.OptionMenu(root, throttlechoice, "None", "True", "False")
throttleBool.grid(column=5, row=14)

###############################################################
#Brake(bool)

brakelabel = tk.Label(root, text="Brake:", font=("Courier"))
brakelabel.grid(column=4, row=15)

brakechoice = tk.StringVar(root)
brakechoice.set("None") #makes the default choice in the dropdown menu

brakeBool = tk.OptionMenu(root, brakechoice, "None", "True", "False")
brakeBool.grid(column=5, row=15)

###############################################################
#Steering(bool)

steeringlabel = tk.Label(root, text="Steering:", font=("Courier"))
steeringlabel.grid(column=4, row=16)

steeringchoice = tk.StringVar(root)
steeringchoice.set("None") #makes the default choice in the dropdown menu

steeringBool = tk.OptionMenu(root, steeringchoice, "None", "True", "False")
steeringBool.grid(column=5, row=16)

###############################################################
#Speed(bool)

speedlabel = tk.Label(root, text="Speed:", font=("Courier"))
speedlabel.grid(column=4, row=17)

speedchoice = tk.StringVar(root)
speedchoice.set("None") #makes the default choice in the dropdown menu

speedBool = tk.OptionMenu(root, speedchoice, "None", "True", "False")
speedBool.grid(column=5, row=17)


###############################################################
#Yaw Rate(bool)

yawlabel = tk.Label(root, text="Yaw Rate:", font=("Courier"))
yawlabel.grid(column=4, row=18)

yawchoice = tk.StringVar(root)
yawchoice.set("None") #makes the default choice in the dropdown menu

yawBool = tk.OptionMenu(root, yawchoice, "None", "True", "False")
yawBool.grid(column=5, row=18)


###############################################################
#Driving Mode(bool)

#dm2 because dm is already for the driving mode in scenario.
dm2label = tk.Label(root, text="Driving Mode:", font=("Courier"))
dm2label.grid(column=4, row=19)

dm2choice = tk.StringVar(root)
dm2choice.set("None") #makes the default choice in the dropdown menu

dm2Bool = tk.OptionMenu(root, dm2choice, "None", "True", "False")
dm2Bool.grid(column=5, row=19)



###############################################################
#Location(bool)

#loc2 because loc is already for the location in scenario.

loc2label = tk.Label(root, text="Location:", font=("Courier"))
loc2label.grid(column=4, row=20)

loc2choice = tk.StringVar(root)
loc2choice.set("None") #makes the default choice in the dropdown menu

loc2Bool = tk.OptionMenu(root, loc2choice, "None", "True", "False")
loc2Bool.grid(column=5, row=20)


###############################################################
#Speed(bool)
#time2 because time is already for the time in scenario.


time2label = tk.Label(root, text="Time:", font=("Courier"))
time2label.grid(column=4, row=21)

time2choice = tk.StringVar(root)
time2choice.set("None") #makes the default choice in the dropdown menu

time2Bool = tk.OptionMenu(root, time2choice, "None", "True", "False")
time2Bool.grid(column=5, row=21)


###############################################################
#Start and Stop functions 
def strtobool(TorF: str) -> bool:
    if TorF == "True":
        return True
    elif TorF == "False":
        return False
    else:
        return None

def intify(inputparam : str):
    if inputparam == "":
        return None
        
    return int(inputparam) 

def floatify(inputparam: str): 
    if inputparam == "":
        return None

    return float(inputparam)

def intify_list(inputparam: list): 
    
    for value in inputparam:
        if intify(value) == None:
            return None
    intified = []
    
    for value in inputparam:
        intified.append(intify(value))

    return intified

def floatify_list(inputparam: list): 
    
    for value in inputparam:
        if floatify(value) == None:
            return None
    floatified = []
    
    for value in inputparam:
        floatified.append(floatify(value))

    return floatified


def printingdicts():

    if intify(dmentry1.get()) == -1:
        drivingmode = -1
    else:
        drivingmode = floatify_list([(dmentry1.get()), (dmentry2.get())])

    scenario_dict = {
        "location" : floatify_list([(locEntry1.get()), (locEntry2.get()), (locEntry3.get())]),
        "time" : intify_list([(timeEntry1.get()), (timeEntry2.get())]),
        "weather" : weatherchoice.get(),
        "vehicle" : ventry.get(),
        "drivingmode" : drivingmode
    }

    dataset_dict = {
        "rate" : intify(rentry.get()),
        "frame" : intify_list([(frameentry1.get()), (frameentry2.get())]),
        "vehicles" : strtobool(vehicleschoice.get()),
        "peds" : strtobool(pedschoice.get()),
        "trafficsigns" : strtobool(trafficschoice.get()),
        "direction" : floatify_list([(directionentry1.get()), (directionentry2.get()), (directionentry3.get())]),
        "reward" : intify_list([(rewardentry1.get()), (rewardentry2.get()), (rewardentry3.get())]),
        "throttle" : strtobool(throttlechoice.get()),
        "brake" : strtobool(brakechoice.get()),
        "steering" : strtobool(steeringchoice.get()),
        "speed" : strtobool(speedchoice.get()),
        "yawrate" : strtobool(yawchoice.get()),
        "drivingmode" : strtobool(dm2choice.get()),
        "location" : strtobool(loc2choice.get()),
        "time" : strtobool(time2choice.get())
    }    

    for each in scenario_dict:
        print(scenario_dict[each])
    
    for each in dataset_dict:
        print(dataset_dict[each])

    return None

def startCommand():
    if intify(dmentry1.get()) == -1:
        drivingmode = -1
    else:
        drivingmode = floatify_list([(dmentry1.get()), (dmentry2.get())])

    scenario_dict = {
        "location" : floatify_list([(locEntry1.get()), (locEntry2.get()), (locEntry3.get())]),
        "time" : intify_list([(timeEntry1.get()), (timeEntry2.get())]),
        "weather" : weatherchoice.get(),
        "vehicle" : ventry.get(),
        "drivingmode" : drivingmode
    }

    dataset_dict = {
        "rate" : intify(rentry.get()),
        "frame" : intify_list([(frameentry1.get()), (frameentry2.get())]),
        "vehicles" : strtobool(vehicleschoice.get()),
        "peds" : strtobool(pedschoice.get()),
        "trafficsigns" : strtobool(trafficschoice.get()),
        "direction" : floatify_list([(directionentry1.get()), (directionentry2.get()), (directionentry3.get())]),
        "reward" : intify_list([(rewardentry1.get()), (rewardentry2.get()), (rewardentry3.get())]),
        "throttle" : strtobool(throttlechoice.get()),
        "brake" : strtobool(brakechoice.get()),
        "steering" : strtobool(steeringchoice.get()),
        "speed" : strtobool(speedchoice.get()),
        "yawrate" : strtobool(yawchoice.get()),
        "drivingmode" : strtobool(dm2choice.get()),
        "location" : strtobool(loc2choice.get()),
        "time" : strtobool(time2choice.get())
    }    
    
    # Configures the information that we want DeepGTAV to generate and send to us. 

    # See deepgtav/messages.py to see what options are supported
    dataset = Dataset(
        rate=dataset_dict["rate"], 
        frame=dataset_dict["frame"], 
        vehicles=dataset_dict["vehicles"], 
        peds=dataset_dict["peds"], 
        trafficSigns=dataset_dict["trafficsigns"], 
        direction=dataset_dict["direction"], 
        reward=dataset_dict["reward"], 
        throttle=dataset_dict["throttle"], 
        brake=dataset_dict["brake"], 
        steering=dataset_dict["steering"], 
        speed=dataset_dict["speed"], 
        yawRate=dataset_dict["yawrate"], 
        drivingMode=dataset_dict["drivingmode"], 
        location=dataset_dict["location"], 
        time=dataset_dict["time"]
        )
    # Send the Start request to DeepGTAV.
    scenario = Scenario(
        location=scenario_dict["location"],
        time=scenario_dict["time"],
        weather=scenario_dict["weather"],
        vehicle=scenario_dict["vehicle"], 
        drivingMode=scenario_dict["drivingmode"]
        )

        
    # Driving style is set to normal, with a speed of 15.0 mph. All other scenario options are random.
    print("sending scenario")
    client.sendMessage(Start(scenario=scenario, dataset = dataset))
    print("scenario sent")
    startPredict()


def stopCommand(): 
    client.sendMessage(Stop())

def onclosing(): 
    if not(client is None):
        print("closing client")
        client.close()
    root.destroy()
    

###############################################################
#Start & Stop Buttons
startButton = tk.Button(root, text = "Start", command = startCommand)
startButton.grid(column=4,row=25)

stopButton = tk.Button(root, text = "Stop", command = stopCommand)
stopButton.grid(column=6,row=25)
##############################################################


#root.protocol("WM_DELETE_WINDOW", onclosing)
#root.protocol()
root.mainloop() #needed to keep window open

print("mainloop done")