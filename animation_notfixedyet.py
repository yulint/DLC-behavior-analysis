### Animation
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation

fig, ax = plt.subplots(figsize=(8,7))

pause = 1/35 # should match fps of camera

for t in range(len(male_nose_interpolated_lfilt_x)):
    if t == 0:
        points1, = ax.plot(male_nose_interpolated_lfilt_x, male_nose_interpolated_lfilt_y, marker='o', linestyle='None', color = 'blue')
        points2, = ax.plot(male_tail_interpolated_lfilt_x, male_tail_interpolated_lfilt_y, marker='o', linestyle='None', color = 'black')
        points3, = ax.plot(male_right_ear_interpolated_lfilt_x, male_right_ear_interpolated_lfilt_y, marker='o', linestyle='None', color = 'cyan')
        points4, = ax.plot(male_left_ear_interpolated_lfilt_x, male_left_ear_interpolated_lfilt_y, marker='o', linestyle='None', color = 'lightblue')
        
        points5, = ax.plot(female_nose_interpolated_lfilt_x, female_nose_interpolated_lfilt_y, marker='o', linestyle='None', color = 'red')
        points6, = ax.plot(female_tail_interpolated_lfilt_x, female_tail_interpolated_lfilt_y, marker='o', linestyle='None', color = 'pink')
        points7, = ax.plot(female_right_ear_interpolated_lfilt_x, female_right_ear_interpolated_lfilt_y, marker='o', linestyle='None', color = 'purple')
        points8, = ax.plot(female_left_ear_interpolated_lfilt_x, female_left_ear_interpolated_lfilt_y, marker='o', linestyle='None', color = 'salmon')        
        
        ax.set_xlim(500, 2000) 
        ax.set_ylim(0, 1000) 
    else:
        x = male_nose_interpolated_lfilt_x[t]
        y =male_nose_interpolated_lfilt_y[t]
        points1.set_data(x, y)
        
        x = male_tail_interpolated_lfilt_x[t]
        y = male_tail_interpolated_lfilt_y[t]
        points2.set_data(x, y)
        
        x = male_right_ear_interpolated_lfilt_x[t]
        y = male_right_ear_interpolated_lfilt_y[t]
        points3.set_data(x, y)
        
        x = male_left_ear_interpolated_lfilt_x[t]
        y = male_left_ear_interpolated_lfilt_y[t]
        points4.set_data(x, y)
        
        
        
        x = female_nose_interpolated_lfilt_x[t]
        y = female_nose_interpolated_lfilt_y[t]
        points5.set_data(x, y)
        
        x = female_tail_interpolated_lfilt_x[t]
        y = female_tail_interpolated_lfilt_y[t]
        points6.set_data(x, y)
        
        x = female_right_ear_interpolated_lfilt_x[t]
        y = female_right_ear_interpolated_lfilt_y[t]
        points7.set_data(x, y)
        
        x = female_left_ear_interpolated_lfilt_x[t]
        y = female_left_ear_interpolated_lfilt_y[t]
        points8.set_data(x, y)
        
    plt.pause(pause)

male_nose_interpolated_lfilt_x




def animate2(i):

    ax1.clear()
    ax1.plot(male_nose_interpolated_lfilt_x[i], male_nose_interpolated_lfilt_x[i])

# ---

#test_data=np.array([[3, 7],[1, 2],[8, 11],[5, -12],[20, 25], [-3, 30], [2,2], [17, 17]])

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)

# create animation
animation.FuncAnimation(fig, animate2, frames=range(1, len(male_nose_interpolated_lfilt_x)), interval=1/30, repeat=False)

# start animation
HTML(anim.to_html5_video())

fig, ax = plt.subplots()

ax.set_xlim(500, 2000) 
ax.set_ylim(0, 1000) 

line, = ax.plot([], [], lw=20)

#ax.set_xlim(( 0, 2))
#ax.set_ylim((-2, 2))

#line, = ax.plot([], [], lw=20)


def init():
    line.set_data([], [])
    return (line,)

# animation function. This is called sequentially
def animate(i):
    x = np.array(male_nose_interpolated_lfilt_x)[i]
    y = np.array(male_nose_interpolated_lfilt_y)[i]
    line.set_data(x, y)
# animation function. This is called sequentially
#    x = np.linspace(0, 2, 1000)
#    y = np.sin(2 * np.pi * (x - 0.01 * i))
#    line.set_data(x, y)
    return (line,)

# call the animator. blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=500, interval=20, blit=True)

HTML(anim.to_html5_video())


data_x = np.array(mother_earR_x)

data = np.array([mother_earR_x, mother_earR_y])

data

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Create initial data
data = np.array([mother_earR_x, mother_earR_y])

# Create figure and axes
fig = plt.figure()
ax = plt.axes(xlim=(0, 200), ylim=(0, 10))

# Create initial objects
line, = ax.plot([], [], 'r-')
annotation = ax.annotate('A0', xy=(data[0][0], data[1][0]))
annotation.set_animated(True)

# Create the init function that returns the objects
# that will change during the animation process
def init():
    return line, annotation

# Create the update function that returns all the
# objects that have changed
def update(num):
    newData = np.array([[1 + num, 2 + num / 2, 3, 4 - num / 4, 5 + num],
                        [7, 4, 9 + num / 3, 2, 3]])
    line.set_data(newData)
    # This is not working i 1.2.1
    # annotation.set_position((newData[0][0], newData[1][0]))
    annotation.xytext = (newData[0][0], newData[1][0])
    return line, annotation

anim = animation.FuncAnimation(fig, update, frames=25, init_func=init,
                               interval=200, blit=True)
HTML(anim.to_html5_video())
