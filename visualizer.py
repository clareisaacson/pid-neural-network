import Tkinter
import time
import math

### VISUALIZER CONSTANTS ###
WIDTH = 1200 # width of window
HEIGHT = 700 # height of window
CART_Y = HEIGHT//2
CART_WIDTH = 100
CART_HEIGHT = 25
CART_COLOR = 'green'
SCALE = 250 # should be (WIDTH/2) / max(cartPosition) -- if you want it to span entire window
TIME_STEP = 0.01

# Create and open tkinter window
window = Tkinter.Tk()
CANVAS = Tkinter.Canvas(window, width = WIDTH, height = HEIGHT)
CANVAS.pack()

### VISUALIZER ###

def run_visualizer(rod_length, rod_angles, cart_positions, ball_diameter = 40):
    """
    Run the visualizer on a given example.

    :param rod_length: rod length in meters
    :type rod_length: float
    :param rod_angles: sequence of angles by time step
    :type rod_angles: float list
    :param cart_positions: sequence of cart positions along the horizontal
                           by time step
    :type cart_positions: float list
    :param ball_diameter: ball diameter in pixels
    :type ball_diameter: int
    """
    rod_length = (rod_length*250)
    for t in range(len(cart_positions)):
        angle = rod_angles[t]
        pos = -cart_positions[t]
        CANVAS.delete(Tkinter.ALL)
        draw_cart(pos)
        draw_rod(pos, angle, ball_diameter, rod_length)
        CANVAS.update()
        time.sleep(TIME_STEP)
    window.destroy()

def draw_cart(center_x):
    """
    Draws cart at given horizontal position

    :param center_x: Cart position, in pixels
    :type center_x: int
    :return type: None
    """
    pos = WIDTH//2 + center_x * SCALE
    x0 = pos-(CART_WIDTH//2)
    x1 = pos+(CART_WIDTH//2)
    y0 = CART_Y - CART_HEIGHT//2
    y1 = CART_Y + CART_HEIGHT//2
    CANVAS.create_rectangle(x0,y0,x1,y1,fill = CART_COLOR)


def draw_rod(center_x, angle, ball_diameter, rod_length):
    """
    Draws rod at from given point ant given angle

    :param center_x: Cart position/start of rod
    :type center_x: int
    :param angle: Rod angle from horizontal, radians
    :type angle: float
    :param ball_diameter: Ball diameter in pixels
    :type ball_diameter: int
    :param rod_length: Rod length in pixels
    :type rod_length: int
    :return type: None
    """
    (h,w) = translate_angle(angle, rod_length)
    x0 = WIDTH//2 + center_x * SCALE
    y0 = CART_Y - CART_HEIGHT//2
    x1 = x0 - w
    y1 = y0 - h
    CANVAS.create_line(x0,y0,x1,y1,width = 5)
    draw_ball(x1,y1,ball_diameter)

def draw_ball(x, y, ball_diameter):
    """
    Draws ball on end of rod

    :param x: horizontal center of ball
    :type x: int
    :param y: vertical center of ball
    :type y: int
    :param ball_diameter: ball diameter in pixels
    :type ball_diameter: int
    :return type: None
    """
    radius = ball_diameter//2
    CANVAS.create_oval(x-radius,y-radius,x+radius,y+radius,fill='grey')

def translate_angle(angle, rod_length):
    """
    Translates rod into height and width legs of a right trangle where 
    rod length is the hypotenuse.

    :param angle: rod angle in radians
    :type angle: float
    :param rod_length: rod length in pixels
    :type rod_length: int
    :return: (height, width) vertical and horizontal legs of right triangle
    :return type: tuple (int, int)
    """
    angle -= math.pi/2 # find angle to the horizontal
    h = math.sin(angle)*rod_length
    w = math.cos(angle)*rod_length
    return (h,w)

#### TEST CASE ####
from visualizer_sample import angles, positions 
if __name__ == "__main__":
    a = angles.split()
    a = [float(x) for x in a]

    p = positions.split()
    p = [float(x) for x in p]

    run_visualizer(.5,a,p)

